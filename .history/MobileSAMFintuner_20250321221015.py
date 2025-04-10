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
from shapely.geometry import Point, Polygon
from torch.cuda.amp import autocast, GradScaler
import logging
from torchmetrics import JaccardIndex
from torch.profiler import profile, record_function, ProfilerActivity

scaler = GradScaler()
NUM_WORKERS=8
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
            learning_rate=1e-4,
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
        # self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
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
        self.train_metric = defaultdict(lambda: deque(maxlen=metrics_interval))
        self.use_bbox = use_bbox
        self.metrics_interval = metrics_interval

    def forward(self, imgs, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size):
        device = imgs.device
        imgs.to("cpu")
        input_images = torch.stack([self.model.preprocess(imgs[i,:,:,:]) for i in range(self.batch_size)], dim=0)
        input_images.to(device)
        # input_images已完成缩放
        features = self.model.image_encoder(input_images)
        num_masks = sum([len(b) for b in bboxes])

        loss_focal = loss_dice = loss_iou = loss_classification = accurare_masks = 0.
        predictions = []
        tp, fp, fn, tn = [], [], [], []
        if self.multimask:
            num_masks *= 3

        for feature, bbox, label, center_point, point_label, target_label, original_input_size_item, resized_input_size_item in \
                zip(features, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size):
            # Perform the apply coords and bboxes. This is very important !!!!!
            # center_point = self.transform.apply_coords_torch(center_point, original_input_size_item) ## 不可通过此处的transform进行缩放，因为center_point应该从原图resize到我们自定义尺寸下的坐标.(由于将缩放后的图像填充放在左上角,因此无需transform 到 此处self.model.image_encoder.img_size尺寸下)
            # bbox = self.transform.apply_boxes_torch(bbox, original_input_size_item)

            if self.use_bbox:
                bbox_input=bbox
            else:
                bbox_input=None
            
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(center_point, point_label),  #
                boxes=bbox_input,
                masks=None,
            )
            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=feature.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.multimask,
            )
            del sparse_embeddings, sparse_embeddings
            
            # Upscale the masks to the original image resolution
            masks = F.interpolate(
                low_res_masks,
                (self.model.image_encoder.img_size, self.model.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            ) ## self.model.image_encoder.img_size在vit_t设置下为1024
            del low_res_masks
            
            resized_height, resized_width = resized_input_size_item[0], resized_input_size_item[1]

            masks = masks[..., : resized_height, : resized_width]
            masks = F.interpolate(masks, original_input_size_item, mode="bilinear", align_corners=False)
            predictions.append(masks)

            # prediction_dict = {
            #     'masks': masks,
            #     'iou_predictions': iou_predictions,
            #     'low_res_masks': low_res_masks,
            # }

            b,c,h,w=masks.shape
                        
            if self.multimask:
                label = torch.repeat_interleave(label.unsqueeze(1), masks.shape[1], dim=1).view(b*3,-1,h,w)
                masks = masks.view(b*3,-1,h,w)
                iou_predictions = iou_predictions.reshape(-1,1)
            else:
                label = label.unsqueeze(1)
                
            label = F.interpolate(label.float(), original_input_size_item, mode="bilinear", align_corners=False)
            label = label.int()

            assert masks.shape == label.shape, f"Masks shape {masks.shape} and label shape {label.shape} do not match"

            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                masks,
                label, # label.unsqueeze(1),
                mode='binary',
                threshold=self.mask_threshold,
            )

            batch_iou = smp.metrics.iou_score(batch_tp, batch_fp, batch_fn, batch_tn)
            # Compute the loss            
            masks = masks.squeeze(1).flatten(1)
            label = label.flatten(1)
            loss_focal += sigmoid_focal_loss(masks, label.float(), num_masks)
            loss_dice += dice_loss(masks, label.float(), num_masks)
            del masks
            loss_iou += F.mse_loss(iou_predictions, batch_iou, reduction='sum') / num_masks
            del iou_predictions
            tp.append(batch_tp)
            fp.append(batch_fp)
            fn.append(batch_fn)
            tn.append(batch_tn)
            accurare_masks += (batch_tp + batch_tn).sum().item() / (batch_tp + batch_fp + batch_fn + batch_tn).sum().item()
            del batch_tp, batch_fn, batch_tn, batch_iou
        accuracy = accurare_masks / num_masks
    
        return {
            'loss': 20. * loss_focal + loss_dice + loss_iou,  # SAM default loss
            'loss_focal': loss_focal,
            'loss_dice': loss_dice,
            'loss_iou': loss_iou,
            'loss_classification': loss_classification,
            'predictions': predictions,
            'acc': accuracy,
            'tp': torch.cat(tp),
            'fp': torch.cat(fp),
            'fn': torch.cat(fn),
            'tn': torch.cat(tn),
        }


    def training_step(self, batch, batch_nb):
        imgs, bboxes, labels, center_points, point_labels, img_name,category_ids ,original_input_size,resized_input_size = batch

        with autocast():
            outputs = self(imgs, bboxes, labels, center_points, point_labels,category_ids,original_input_size,resized_input_size )

        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.train_metric[metric].append(outputs[metric])
        # aggregate step metics
        step_metrics = [torch.cat(list(self.train_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
        metrics = {
            "loss": outputs["loss"],
            "loss_focal": outputs["loss_focal"],
            "loss_dice": outputs["loss_dice"],
            "loss_classification": outputs["loss_classification"],
            "acc": outputs["acc"],
            "loss_iou": outputs["loss_iou"],
            "train_per_mask_iou": per_mask_iou,
        }
        self.log_dict(metrics, prog_bar=True, rank_zero_only=True)
        del outputs
        torch.cuda.empty_cache()
        # logging.info(f"Training step {batch_nb}")
        # logging.info(f"Training metrics: {metrics}")
        return metrics
    
    

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        def warmup_step_lr_builder(warmup_steps, milestones, gamma):
            def warmup_step_lr(steps):
                if steps < warmup_steps:
                    lr_scale = (steps + 1.) / float(warmup_steps)
                else:
                    lr_scale = 1.
                    for milestone in sorted(milestones):
                        if steps >= milestone * self.trainer.estimated_stepping_batches:
                            lr_scale *= gamma
                return lr_scale

            return warmup_step_lr
        milestones = [int(m * self.trainer.estimated_stepping_batches) for m in [0.66667, 0.86666]]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            warmup_step_lr_builder(250, [0.66667, 0.86666], 0.1)
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
    
    

# def training_step(self, batch, batch_nb):
#     imgs, bboxes, labels, center_points, point_labels, img_name, category_ids, original_input_size, resized_input_size = batch

#     with autocast():
#         outputs = self(imgs, bboxes, labels, center_points, point_labels, category_ids, original_input_size, resized_input_size)

#     # 只存储累积的指标值
#     for metric in ['tp', 'fp', 'fn', 'tn']:
#         self.train_metric[metric].append(outputs[metric].sum())

#     # 每隔 10 个批次计算一次 IoU
#     if batch_nb % 10 == 0:
#         step_metrics = [torch.stack(list(self.train_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
#         per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
#         metrics = {
#             "loss": outputs["loss"],
#             "train_per_mask_iou": per_mask_iou,
#         }
#         self.log_dict(metrics, prog_bar=True, rank_zero_only=True)

#     # 手动释放显存
#     del outputs
#     torch.cuda.empty_cache()

#     return {"loss": outputs["loss"]}



    # def validation_step(self, batch, batch_nb):
    #     imgs, bboxes, labels, center_points, point_labels, img_name, category_ids ,original_input_size, resized_input_size = batch
    #     outputs = self(imgs, bboxes, labels, center_points, point_labels, category_ids, original_input_size, resized_input_size)
    #     torch.cuda.empty_cache()
    #     outputs.pop("predictions")
    #     outputs.pop('loss_focal')
    #     outputs.pop('loss_dice')
    #     outputs.pop('loss_iou')
    #     outputs.pop('loss_classification')
    #     outputs.pop('acc')
    #     return outputs

    # def validation_epoch_end(self, outputs):
    #     if NUM_GPUS > 1:
    #         outputs = all_gather(outputs, gather_batch_size=500)## 存放多个字典的列表
    #         # the outputs are a list of lists, so flatten it
        
    #     len_val = len(outputs)
    #     losses = sum([output['loss'].to('cpu') for output in outputs])
    #     average_loss = losses / len_val
            
    #     tp, fp, fn, tn = 0, 0, 0, 0
    #     for output in outputs:
    #         tp = output['tp'].to('cpu')
    #         fp = output['fp'].to('cpu')
    #         fn = output['fn'].to('cpu')
    #         tn = output['tn'].to('cpu')
    #     precision = tp / (tp + fp + 1e-16)
    #     recall = tp / (tp + fn + 1e-16)
    #     f1 = 2 * precision * recall / (precision + recall + 1e-16)
    #     accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-16)
    #     metrics = {
    #         "val_precision": precision,
    #         "val_recall": recall,
    #         "val_f1": f1,
    #         "val_acc": accuracy,
    #         "average_loss": average_loss
    #     }
    #     logging.info(f"Testing metrics: {metrics}")
    #     self.log_dict(metrics)
    #     return metrics