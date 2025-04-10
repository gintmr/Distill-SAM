import os
import argparse
import random
import sys
from collections import defaultdict, deque
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
from PIL import Image
import cv2
from eval_tools import calculate_pa_iou, overlay_mask_on_image,overlay_point_on_image, combine_visualize_results

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms
from mobile_sam.utils.transforms import ResizeLongestSide
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import segmentation_models_pytorch as smp
from transformers.models.maskformer.modeling_maskformer import dice_loss, sigmoid_focal_loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from eval_tools import init_model
from mobile_sam import sam_model_registry
from torch.cuda.amp import autocast

NUM_WORKERS=4
NUM_GPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))



def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def all_gather(data, gather_batch_size=500):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    
    torch.cuda.empty_cache()
    
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # 将数据分批
    data_batches = [data[i:i + gather_batch_size] for i in range(0, len(data), gather_batch_size)]
    gathered_batches = []

    for batch in data_batches:
        # 序列化当前批次的数据
        buffer = pickle.dumps(batch)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")

        # 获取当前批次的大小
        local_size = torch.LongTensor([tensor.numel()]).to("cuda")
        size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)

        # 准备接收张量
        tensor_list = [torch.ByteTensor(size=(max_size,)).to("cuda") for _ in range(world_size)]

        # 填充当前批次的数据
        if local_size != max_size:
            padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
            tensor = torch.cat((tensor, padding), dim=0)

        # 执行 all_gather
        dist.all_gather(tensor_list, tensor)

        # 反序列化收集到的数据
        batch_data_list = []
        for size, t in zip(size_list, tensor_list):
            buffer = t.cpu().numpy().tobytes()[:size]
            batch_data_list.append(pickle.loads(buffer))

        # 合并批次数据
        gathered_batches.append(batch_data_list)

    # 将所有批次的数据合并为一个列表
    gathered_data = [item for batch in gathered_batches for sublist in batch for item in sublist]
    return gathered_data

class MobileSAMFintuner(pl.LightningModule):
    mask_threshold: float = 0.5
    def __init__(
            self,
            model_type,
            checkpoint_path,
            freeze_image_encoder=False,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=False,
            batch_size=1,
            learning_rate=1e-5,
            weight_decay=1e-4,
            train_dataset=None,
            val_dataset=None,
            metrics_interval=10,
            multimask=False,
            use_bbox=False,
    ):
        super(MobileSAMFintuner, self).__init__()

        self.model_type = model_type
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.model.to(device=self.device)
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.multimask = multimask
        # 初始化训练指标，使用defaultdict和deque来存储训练指标，设置最大长度为metrics_interval
        self.train_metric = defaultdict(lambda: deque(maxlen=metrics_interval))

        self.use_bbox = use_bbox
        self.metrics_interval = metrics_interval


    def forward(self, imgs, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size, coco_image_names):
        device = imgs.device
        imgs.to("cpu") #g CHW
        
        #g 从输入的L,L图像 => （1024,1024）model.image_encoder.img_size
        input_images = torch.stack([self.model.preprocess(imgs[i,:,:,:]) for i in range(self.batch_size)], dim=0)
        input_images.to(device)
        
        try:
            # G input_images已完成缩放
            features = self.model.image_encoder(input_images)
            num_masks = sum([len(b) for b in bboxes])

            loss_focal = loss_dice = loss_iou = accurare_masks = 0.
            BS_IoU = []
            BS_pa = []
            tp, fp, fn, tn = [], [], [], []
            if self.multimask:
                num_masks *= 3

            for feature, bbox, label, center_point, point_label, target_label, original_input_size_item, resized_input_size_item, coco_image_name, img in \
                    zip(features, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size, coco_image_names, imgs):
                '''
                original_input_size_item: ori阶段图像尺寸 => H0,W0
                resized_input_size_item: middle阶段图像尺寸 => H1,W1(未填充成正方形)
                img: 填充成正方形(resized_input_size_item的长边)
                '''
                if self.use_bbox:
                    bbox_input=bbox
                else:
                    bbox_input=None


                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=(center_point, point_label),  #
                    boxes=bbox_input,
                    masks=None,
                )
                torch.cuda.empty_cache()


                # Predict masks
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=feature.unsqueeze(0),
                    # image_embeddings=feature, #G 此处是predictor中的写法
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=self.multimask,
                )
                torch.cuda.empty_cache()
                
                # Upscale the masks to the original image resolution
                masks = F.interpolate(
                    low_res_masks,
                    (self.model.image_encoder.img_size, self.model.image_encoder.img_size),
                    mode="bilinear",
                    align_corners=False,
                ) ## self.model.image_encoder.img_size在vit_t设置下为1024
                torch.cuda.empty_cache()
                del low_res_masks
                

                resized_height, resized_width = img.shape[1], img.shape[2]
                masks = masks[..., : resized_height, : resized_width]
                # masks = F.interpolate(masks, original_input_size_item, mode="bilinear", align_corners=False)
                torch.cuda.empty_cache()


                b,c,h,w=masks.shape
                if self.multimask:
                    label = torch.repeat_interleave(label.unsqueeze(1), masks.shape[1], dim=1).view(b*3,-1,h,w)
                    masks = masks.view(b*3,-1,h,w)
                    iou_predictions = iou_predictions.reshape(-1,1)
                else:
                    label = label.unsqueeze(1)
                    #!------------设置metric-size，减少计算量------------
                # metric_size_item = (int(original_input_size_item[0]) // 2, int(original_input_size_item[1]) // 2)
                # label = F.interpolate(label.float(), metric_size_item, mode="bilinear", align_corners=False)
                # label = label.int()
                # masks = F.interpolate(masks, metric_size_item, mode="bilinear", align_corners=False)
                # assert masks.shape == label.shape, f"Masks shape {masks.shape} and label shape {label.shape} do not match"
                    #!------------设置metric-size，减少计算量------------
                

                #G ------- 计算IoU + 可视化输出--------G#
                pred_masks = masks.squeeze(1)
                iou_label = label.squeeze(1)
                pa_list, iou_list = calculate_pa_iou(pred_masks, iou_label)
                
                ext_name = coco_image_name.split('.')[-1]
                step = self.current_epoch
                training_visual_path = f"/data2/wuxinrui/RA-L/MobileSAM/training_visual/{step}"
                if not os.path.exists(training_visual_path):
                    os.makedirs(training_visual_path, exist_ok=True)
                #G 1% to save images
                if random.random() < 0.01:
                    output_path_pred = os.path.join(training_visual_path, f"{step}_{coco_image_name.replace(ext_name, '_pred.jpg')}")
                    output_path_GT = os.path.join(training_visual_path, f"{step}_{coco_image_name.replace(ext_name, '_GT.jpg')}")
                    output_path_point = os.path.join(training_visual_path, f"{step}_{coco_image_name.replace(ext_name, '_point.jpg')}")
                    output_path_combined = os.path.join(training_visual_path, f"{step}_{coco_image_name.replace(ext_name, '_combined.jpg')}")


                    pred_mask_array = overlay_mask_on_image(pred_masks, output_path_pred, image_array=img, array_out=True)
                    GT_mask_array = overlay_mask_on_image(iou_label, output_path_GT, image_array=img, array_out=True)
                    point_array = overlay_point_on_image(center_point, output_path_point, image_array=img, array_out=True)
                    combine_visualize_results(pred_mask_array, GT_mask_array, point_array, output_path_combined)
                    

                    #G check to avoid too many images in the folder
                    image_files = [f for f in os.listdir(training_visual_path) if f.endswith(('_pred.jpg', '_GT.jpg', '_point.jpg'))]
                    if len(image_files) > 30:
                        #G randomly select three file prefixes
                        random_prefixes = random.sample(set(f.split('_GT.jpg')[0] for f in image_files), 3)
                        for prefix in random_prefixes:
                            for suffix in ('_pred.jpg', '_GT.jpg', '_point.jpg'):
                                file_to_delete = os.path.join(training_visual_path, f"{prefix}{suffix}")
                                if os.path.exists(file_to_delete):
                                    os.remove(file_to_delete)
                #G ------- 计算IoU + 可视化输出--------G#

                
                single_IoU = np.mean(iou_list)
                single_pa = np.mean(pa_list)
                BS_IoU.append(single_IoU)
                BS_pa.append(single_pa)
                    
                masks.to("cpu")
                label.to("cpu")
                torch.cuda.empty_cache()
                
                batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                    masks,
                    label, 
                    mode='binary',
                    threshold=self.mask_threshold,
                )
                #G 输入尺寸 => [batch_size, channels, height, width]
                
                batch_iou = smp.metrics.iou_score(batch_tp, batch_fp, batch_fn, batch_tn)
                #G IoU = TP / (TP + FP + FN)
                
                # Compute the loss            

                masks = F.interpolate(masks, scale_factor=0.5, mode="bilinear")
                label = F.interpolate(label.float(), scale_factor=0.5, mode="nearest")
                masks = masks.squeeze(1).flatten(1)
                label = label.flatten(1)
                
                loss_focal += sigmoid_focal_loss(masks, label.float(), num_masks, alpha=0.6, gamma=2.5)
                #G more information on https://kimi.moonshot.cn/chat/cvf7v67f2enav567m6h0

                loss_dice += dice_loss(masks, label.float(), num_masks)
                #G more information on https://kimi.moonshot.cn/chat/cvf7v67f2enav567m6h0
                
                del masks
                loss_iou += F.mse_loss(iou_predictions, batch_iou, reduction='sum') / num_masks
                #G the meaning of prediction_iou refers to https://kimi.moonshot.cn/chat/cvf7v67f2enav567m6h0
                
                del iou_predictions, batch_iou
                tp.append(batch_tp)
                fp.append(batch_fp)
                fn.append(batch_fn)
                tn.append(batch_tn)
                
                accurare_masks += (batch_tp + batch_tn).sum().item() / (batch_tp + batch_fp + batch_fn + batch_tn).sum().item()
                del batch_tp, batch_fn, batch_fp, batch_tn
            accuracy = accurare_masks / num_masks

            penalty_coefficient = 0.5 * (1 + single_IoU)
            av_BS_IoU = np.mean(BS_IoU)
            av_BS_pa = np.mean(BS_pa)
            del BS_IoU, BS_pa

            return {
                'loss': (5 * loss_focal + loss_dice + loss_iou) / penalty_coefficient,  # SAM default loss
                'loss_focal': loss_focal,
                'loss_dice': loss_dice,
                'loss_iou': loss_iou,
                'acc': accuracy,
                'av_BS_IoU': av_BS_IoU,
                'av_BS_pa': av_BS_pa,
                'tp': torch.cat(tp),
                'fp': torch.cat(fp),
                'fn': torch.cat(fn),
                'tn': torch.cat(tn),
            }
        except Exception as e:
            # 获取图片名称
            img_names = [coco_image_name for coco_image_name in coco_image_names]
            print(f"Error occurred while processing images: {img_names}")
            print(f"Error details: {str(e)}")
            # 返回一个标记为无效的结果
            raise e
            # return {'valid': False}


    def training_step(self, batch, batch_nb):
        imgs, bboxes, labels, center_points, point_labels, img_name,category_ids ,original_input_size,resized_input_size, coco_image_names = batch

        with autocast():
            outputs = self(imgs, bboxes, labels, center_points, point_labels,category_ids,original_input_size,resized_input_size, coco_image_names)

        if not outputs.get('valid', True):
            print("Skipping invalid batch")
            return None  # 跳过当前批次
        
        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.train_metric[metric].append(outputs[metric])
        # aggregate step metics
        step_metrics = [torch.cat(list(self.train_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
        del step_metrics
        metrics = {
            "loss": outputs["loss"],
            #G this is the core target function. The other metrics are just for monitoring and logging.
            "loss_focal": outputs["loss_focal"],
            "loss_dice": outputs["loss_dice"],
            "acc": outputs["acc"],
            "av_BS_IoU": outputs["av_BS_IoU"],
            "av_BS_pa": outputs["av_BS_pa"],
            "loss_iou": outputs["loss_iou"],
            "train_per_mask_iou": per_mask_iou,
        }
        self.log_dict(metrics, prog_bar=True, rank_zero_only=True)
        del outputs
        torch.cuda.empty_cache()
        return metrics

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) 
        #G weight_decay -> L2 regularization hyparam, avoid overfitting, recommend 1e-4
        
        def warmup_step_lr_builder(warmup_steps, milestones, gamma):
            def warmup_step_lr(steps):
                # 如果步数小于预热步数，则学习率缩放为步数加1除以预热步数
                if steps < warmup_steps:
                    lr_scale = (steps + 1.) / float(warmup_steps)
                # 否则，学习率缩放为1
                # 否则，学习率缩放为1
                else:
                    # 遍历所有里程碑
                    lr_scale = 1.
                        # 如果步数大于等于里程碑乘以估计的步数，则学习率缩放乘以gamma
                    for milestone in sorted(milestones):
                        if steps >= milestone * self.trainer.estimated_stepping_batches:
                # 返回学习率缩放
                            lr_scale *= gamma
                return lr_scale

            return warmup_step_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            warmup_step_lr_builder(400, [0.66667, 0.86666], 0.1)
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "step",
                'frequency': 1,
            }
        }

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=True)
        return train_loader
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False)
        return val_loader


    def validation_step(self, batch, batch_nb):
        imgs, bboxes, labels, center_points, point_labels, img_name,category_ids ,original_input_size,resized_input_size, coco_image_names = batch

        with autocast():
            outputs = self(imgs, bboxes, labels, center_points, point_labels,category_ids,original_input_size,resized_input_size, coco_image_names)

        if not outputs.get('valid', True):
            print("Skipping invalid batch")
            return None  # 跳过当前批次
        
        torch.cuda.empty_cache()
        outputs.pop('loss_focal')
        outputs.pop('loss_dice')
        outputs.pop('loss_iou')
        outputs.pop('acc')
        return outputs

    def validation_epoch_end(self, outputs):
        if NUM_GPUS > 1:
            outputs = all_gather(outputs, gather_batch_size=500)## 存放多个字典的列表
            # the outputs are a list of lists, so flatten it
        
        len_val = len(outputs)
        losses = sum([output['loss'].cpu() for output in outputs])
        average_loss = losses / len_val
        
        IoUs = sum([output['av_BS_IoU'] for output in outputs])
        average_IoU = IoUs / len_val
        tp, fp, fn, tn = 0, 0, 0, 0
        for output in outputs:
            tp += output['tp'].to('cpu')
            fp += output['fp'].to('cpu')
            fn += output['fn'].to('cpu')
            tn += output['tn'].to('cpu')
        
        precision = tp / (tp + fp + 1e-16)
        recall = tp / (tp + fn + 1e-16)
        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-16)
        metrics = {
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_acc": accuracy,
            "average_loss": average_loss,
            "average_IoU": average_IoU
        }
        self.log_dict(metrics)
        return metrics