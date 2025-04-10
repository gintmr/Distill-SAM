from DistillFintuner import AbstractDistillFinetuner

class Imgencoder_Distil(AbstractDistillFinetuner):
    def __init__(
            self,
            T_model,
            S_model,
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
            distill_weight=0.5  # 新增参数：蒸馏权重
    ):
        super(Imgencoder_Distil, self).__init__(
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
        """
        前向传播逻辑，包含图像编码器的蒸馏。
        """
        device = imgs.device