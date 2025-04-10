from .DistillFintuner import AbstractDistillFinetuner
import torch
import torch.nn.functional as F
from .Data_Argument.img_size_arg import random_resize
from transformers.models.maskformer.modeling_maskformer import dice_loss, sigmoid_focal_loss
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from torch.utils.checkpoint import checkpoint
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from eval_tools import calculate_pa_iou
import numpy as np

def feature_distillation_loss(T_features, S_features, T_layers_features, S_layers_features, RATE, reduction='mean', alpha=0.5, beta=0.1):
    """
    计算教师模型和学生模型特征的蒸馏损失。

    参数:
        T_features (torch.Tensor): 教师模型的特征输出。
        S_features (torch.Tensor): 学生模型的特征输出。
        T_layers_features (dict): 教师模型的各层特征输出。
        S_layers_features (dict): 学生模型的各层特征输出。
        reduction (str): 损失的归约方式，可选 'mean' 或 'sum'。
        alpha (float): MSE 损失的权重。
        beta (float): 余弦相似度损失的权重。

    返回:
        torch.Tensor: 蒸馏损失。
    """
    total_loss = 0.0
    # 计算 MSE 损失
    mse_loss = F.mse_loss(S_features, T_features.detach(), reduction=reduction)

    # 计算余弦相似度损失
    T_features_norm = F.normalize(T_features.detach(), p=2, dim=1)
    S_features_norm = F.normalize(S_features, p=2, dim=1)
    cosine_similarity = F.cosine_similarity(S_features_norm, T_features_norm, dim=1)
    cosine_loss = 1 - cosine_similarity.mean()

    total_loss = alpha * mse_loss + beta * cosine_loss
    
    
    temperature = max(3.0 - 2*(RATE), 1.0)
    
    # 阶段感知权重调整
    progress = RATE
    if progress < 0.5:  # 前期侧重特征对齐
        A,B = 0.7, 0.1
    else:  # 后期加强注意力对齐
        A,B = 0.4, 0.1
    
    for T_layers_feature, S_layers_feature in zip(T_layers_features.values(), S_layers_features.values()):
        # 计算 MSE 损失
        if S_layers_feature.shape[-1:] != T_layers_feature.shape[-1:]:
            #G interporlate需要输入四维张量，这里将三维张量扩充为四维张量
            S_layers_feature = S_layers_feature.unsqueeze(-1)  # 形状变为 [1, 4096, 160, 1]
            T_layers_feature = T_layers_feature.unsqueeze(-1)  # 形状变为 [1, 4096, 128, 1]

            # 使用 F.interpolate 进行插值
            S_layers_feature = F.interpolate(S_layers_feature, size=T_layers_feature.shape[-2:], mode='bilinear', align_corners=False)

            # 将四维张量还原为三维
            S_layers_feature = S_layers_feature.squeeze(-1)  
            T_layers_feature = T_layers_feature.squeeze(-1)  
            
        layers_mse_loss = F.mse_loss(S_layers_feature/temperature, T_layers_feature.detach()/temperature, reduction=reduction) * (temperature ** 2)
        layers_cosine_loss = 1 - F.cosine_similarity(F.normalize(S_layers_feature, p=2, dim=1), F.normalize(T_layers_feature.detach(), p=2, dim=1), dim=1).mean()
        
        total_loss += A*layers_mse_loss + B*layers_cosine_loss

    return total_loss


class Imgencoder_Distill(AbstractDistillFinetuner):
    mask_threshold: float = 0.5
    def __init__(
            self,
            T_model,
            S_model,
            checkpoint_path,
            freeze_image_encoder=False,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=False,
            batch_size=1,
            learning_rate=1e-5,
            weight_decay=1e-4, #G avoid overfitting
            train_dataset=None,
            val_dataset=None,
            metrics_interval=20,
            multimask=False,
            use_bbox=False,
            max_steps=10000,
            distill_weight=0.75  # 新增参数：蒸馏权重
    ):
        super(Imgencoder_Distill, self).__init__(
            T_model=T_model,
            S_model=S_model,
            checkpoint_path=checkpoint_path,
            freeze_image_encoder=freeze_image_encoder,
            freeze_prompt_encoder=freeze_prompt_encoder,
            freeze_mask_decoder=freeze_mask_decoder,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            metrics_interval=metrics_interval,
            multimask=multimask,
            use_bbox=use_bbox
        )
        self.distill_weight = distill_weight  # 蒸馏权重
        self.max_steps = max_steps
        
    def forward(self, imgs, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size, coco_image_names):
        
        device = imgs.device
        imgs.to("cpu")
        
        assert self.T_model.image_encoder.img_size == self.S_model.image_encoder.img_size
        
        # imgs = random_resize(imgs)
        
        # self.T_model = FSDP(self.T_model)
        
        
        input_images = torch.stack([self.T_model.preprocess(imgs[i,:,:,:]) for i in range(self.batch_size)], dim=0) #G padding
        input_images.to(device)
        try:
            with autocast():
                T_features, T_layers_features = self.T_model.image_encoder(input_images)
                S_features, S_layers_features = self.S_model.image_encoder(input_images)

                RATE = self.current_epoch / self.max_steps
                distill_loss = feature_distillation_loss(T_features, S_features, T_layers_features, S_layers_features, RATE)
                
                num_masks = sum([len(b) for b in bboxes])

                loss_focal = loss_dice = loss_iou = accurare_masks = 0.
                predictions = []
                tp, fp, fn, tn = [], [], [], []
                if self.multimask:
                    num_masks *= 3

                for T_feature, S_feature, bbox, label, center_point, point_label, target_label, original_input_size_item, resized_input_size_item in \
                        zip(T_features, S_features, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size):
                    if self.use_bbox:
                        bbox_input=bbox
                    else:
                        bbox_input=None
                    
                    sparse_embeddings, dense_embeddings = self.T_model.prompt_encoder(
                        points=(center_point, point_label),  #
                        boxes=bbox_input,
                        masks=None,
                    )

                    # Predict masks
                    low_res_masks, iou_predictions = self.T_model.mask_decoder(
                        image_embeddings=S_feature.unsqueeze(0),
                        image_pe=self.T_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=self.multimask,
                    )
                    
                    # Upscale the masks to the original image resolution
                    masks = F.interpolate(
                        low_res_masks,
                        (self.T_model.image_encoder.img_size, self.T_model.image_encoder.img_size),
                        mode="bilinear",
                        align_corners=False,
                    ) ## self.T_model.image_encoder.img_size在vit_t设置下为1024
                    
                    resized_height, resized_width = resized_input_size_item[0], resized_input_size_item[1]

                    masks = masks[..., : resized_height, : resized_width]
                    masks = F.interpolate(masks, original_input_size_item, mode="bilinear", align_corners=False)
                    
                    # masks = masks > self.T_model.mask_threshold
                    #G 返回二值化后的数据

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

                    pred_masks = masks.squeeze(1)
                    iou_label = label.squeeze(1)
                    pa_list, iou_list = calculate_pa_iou(pred_masks, iou_label)
                    av_BS_IoU = np.mean(iou_list)
                    av_BS_pa = np.mean(pa_list)

                    del pred_masks, iou_label
                    
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
                    masks = masks.squeeze(1).flatten(1)
                    label = label.flatten(1)
                    loss_focal += sigmoid_focal_loss(masks, label.float(), num_masks, alpha=0.6, gamma=2.5)
                    #G more information on https://kimi.moonshot.cn/chat/cvf7v67f2enav567m6h0

                    loss_dice += dice_loss(masks, label.float(), num_masks)
                    #G more information on https://kimi.moonshot.cn/chat/cvf7v67f2enav567m6h0
                    
                    loss_iou += F.mse_loss(iou_predictions, batch_iou, reduction='sum') / num_masks
                    #G the meaning of prediction_iou refers to https://kimi.moonshot.cn/chat/cvf7v67f2enav567m6h0
                    
                    tp.append(batch_tp)
                    fp.append(batch_fp)
                    fn.append(batch_fn)
                    tn.append(batch_tn)

                    accurare_masks += (batch_tp + batch_tn).sum().item() / (batch_tp + batch_fp + batch_fn + batch_tn).sum().item()
                accuracy = accurare_masks / num_masks
            return {
                'loss': 20 * loss_focal + loss_dice + loss_iou + self.distill_weight * distill_loss,  # SAM default loss
                'loss_focal': loss_focal,
                'loss_dice': loss_dice,
                'loss_iou': loss_iou,
                'distill_loss': distill_loss,
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
            raise e
        
    def training_step(self, batch, batch_nb):
        imgs, bboxes, labels, center_points, point_labels, img_name,category_ids ,original_input_size,resized_input_size, coco_image_names = batch

        outputs = self(imgs, bboxes, labels, center_points, point_labels,category_ids,original_input_size,resized_input_size, coco_image_names)
        
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
            "distill_loss": outputs['distill_loss'],
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
