from .DistillFintuner import AbstractDistillFinetuner
import torch
import torch.nn.functional as F
from DistillFinetune.Data_Argument.img_arg import random_resize
def feature_distillation_loss(T_features, S_features, reduction='mean', alpha=0.5, beta=0.01):
    """
    计算教师模型和学生模型特征的蒸馏损失。

    参数:
        T_features (torch.Tensor): 教师模型的特征输出。
        S_features (torch.Tensor): 学生模型的特征输出。
        reduction (str): 损失的归约方式，可选 'mean' 或 'sum'。
        alpha (float): MSE 损失的权重。
        beta (float): 余弦相似度损失的权重。

    返回:
        torch.Tensor: 蒸馏损失。
    """
    # 计算 MSE 损失
    mse_loss = F.mse_loss(S_features, T_features, reduction=reduction)

    # 计算余弦相似度损失
    T_features_norm = F.normalize(T_features, p=2, dim=1)
    S_features_norm = F.normalize(S_features, p=2, dim=1)
    cosine_similarity = F.cosine_similarity(S_features_norm, T_features_norm, dim=1)
    cosine_loss = 1 - cosine_similarity.mean()

    # 总蒸馏损失
    total_loss = alpha * mse_loss + beta * cosine_loss

    return total_loss


class Imgencoder_Distill(AbstractDistillFinetuner):
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
            distill_weight=0.5  # 新增参数：蒸馏权重
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
        
    def forward(self, imgs, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size, coco_image_names):

        device = imgs.device
        imgs.to("cpu")
        
        assert self.T_model.image_encoder.img_size == self.S_model.image_encoder.img_size
        
        
        
        
        input_images = torch.stack([self.T_model.preprocess(imgs[i,:,:,:]) for i in range(self.batch_size)], dim=0) #G padding
        input_images.to(device)
        try:
            T_features = self.T_model.image_encoder(input_images)
            S_features = self.S_model.image_encoder(input_images)

            distill_loss = feature_distillation_loss(T_features, S_features)
            
            loss_dict = {
                "loss": distill_loss,

            }
            return loss_dict
            
        except Exception as e:
            # 获取图片名称
            img_names = [coco_image_name for coco_image_name in coco_image_names]
            print(f"Error occurred while processing images: {img_names}")
            print(f"Error details: {str(e)}")
            raise e
        
    def training_step(self, batch, batch_nb):
        imgs, bboxes, labels, center_points, point_labels, img_name,category_ids ,original_input_size,resized_input_size, coco_image_names = batch

        outputs = self(imgs, bboxes, labels, center_points, point_labels,category_ids,original_input_size,resized_input_size, coco_image_names)
        
        metrics = {
            "loss": outputs["loss"]
        }
        self.log_dict(metrics, prog_bar=True, rank_zero_only=True)
        del outputs
        torch.cuda.empty_cache()
        return metrics