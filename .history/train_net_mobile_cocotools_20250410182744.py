import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
from PIL import Image
import cv2
from sahi.utils.coco import Coco
from sahi.utils.cv import get_bool_mask_from_coco_segmentation
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mobile_sam.utils.transforms import ResizeLongestSide
from eval_tools import get_bool_mask_from_segmentation, random_croods_in_mask, clean_checkpoint_path
from MobileSAMFintuner import MobileSAMFintuner
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
# module.py
from torch.profiler import profile, record_function, ProfilerActivity
import random
from Datasets.coco import Coco2MaskDataset


NUM_GPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
DEVICE = 'cuda'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['INFERENCE_MODE'] = "train"

# torch.cuda.set_per_process_memory_fraction(0.9, device=0)
# torch.cuda.set_per_process_memory_fraction(0.9, device=1)  
# torch.cuda.set_per_process_memory_fraction(0.9, device=2)  
# torch.cuda.set_per_process_memory_fraction(0.9, device=3)  



def main():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--train_data", default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/train_list", type=str, required=False, help="path to the data root")
    # parser.add_argument("--train_anno", default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/train-taxonomic_cleaned.json", type=str, required=False, help="path to the annotation file")

    # parser.add_argument("--val_data", default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/val_list", type=str, required=False, help="path to the data root")
    # parser.add_argument('--val_anno', default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/val-taxonomic_cleaned.json", )
    
    
    parser.add_argument("--train_data", default="/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC/images/train", type=str, required=False, help="path to the data root")
    parser.add_argument("--train_anno", default="/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC/annotations/train.json", type=str, required=False, help="path to the annotation file")

    parser.add_argument("--val_data", default="/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC/images/val", type=str, required=False, help="path to the data root")
    parser.add_argument('--val_anno', default="/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC/annotations/val.json", )
    
    # parser.add_argument("--train_data", default="/data2/wuxinrui/Datasets/COCO/images/train2017", type=str, required=False, help="path to the data root")
    # parser.add_argument("--train_anno", default="/data2/wuxinrui/Datasets/COCO/annotations/instances_train2017_sampled_3.json", type=str, required=False, help="path to the annotation file")

    # parser.add_argument("--val_data", default="/data2/wuxinrui/Datasets/COCO/images/val2017", type=str, required=False, help="path to the data root")
    # parser.add_argument('--val_anno', default="/data2/wuxinrui/Datasets/COCO/annotations/instances_val2017_sampled_2000.json", type=str, required=False, help="path to the annotation file")
    

    parser.add_argument("--model_type", default='vit_t', type=str, required=False, help="model type")
    # parser.add_argument("--checkpoint_path", default="weights/mobile_sam.pt", type=str, required=False, help="path to the checkpoint")
    parser.add_argument("--checkpoint_path", default="/data2/wuxinrui/RA-L/MobileSAM/trained_models/wxr_modified_mobilesam/random_point_v2.pth", type=str, required=False, help="path to the checkpoint")
    parser.add_argument("--freeze_image_encoder", default=True, action="store_true", help="freeze image encoder")
    parser.add_argument("--freeze_prompt_encoder", default=False, action="store_true", help="freeze prompt encoder")
    parser.add_argument("--freeze_mask_decoder", default=False, action="store_true", help="freeze mask decoder")
    # 添加一个名为multimask的参数，类型为布尔型，默认值为False，当该参数被指定时，其值为True，用于生成多掩码
    parser.add_argument("--multimask", action="store_true", help="generate multi masks")
    # 添加一个名为use_bbox的参数，类型为布尔型，默认值为False，当该参数被指定时，其值为True，用于生成多掩码
    parser.add_argument("--use_bbox", default=True, help="generate multi masks")
    parser.add_argument("--use_centerpoint", action="store_true", help="use only one center point", default=False)
    
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--save_topk", type=int, default=3, help="save top K models")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    parser.add_argument("--steps", type=int, default=200000, help="number of steps")
    parser.add_argument("--num_points", type=int, default=6, help="number of random points")
    parser.add_argument("--length", type=int, default=50, help="the length of the chosen masks")

    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--metrics_interval", type=int, default=1000, help="interval for logging metrics")
    
    parser.add_argument("--output_dir", type=str, default="./trained_models/wxr_modified_mobilesam", help="path to save the model")
    
    parser.add_argument("--model_name", type=str, default="final_model.pth", help="model name to save the model")
    parser.add_argument('--every_n_train_steps', default=500)
    
    parser.add_argument("--log_dir", default="./metrics_logs/wxr_modified_mobilesam")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # load the dataset
    train_dataset = Coco2MaskDataset(data_root=args.train_data, annotation_path=args.train_anno, image_size=args.image_size,
                                     length=args.length,num_points=args.num_points,use_centerpoint=args.use_centerpoint)
    val_dataset = Coco2MaskDataset(data_root=args.val_data, annotation_path=args.val_anno, image_size=args.image_size,
                                   length=args.length, num_points=args.num_points,use_centerpoint=args.use_centerpoint)
    # create the model
    train_checkpoint_path = clean_checkpoint_path(args.checkpoint_path, train=True)
    model = MobileSAMFintuner(
        args.model_type,
        train_checkpoint_path,
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_prompt_encoder=args.freeze_prompt_encoder,
        freeze_mask_decoder=args.freeze_mask_decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        metrics_interval=args.metrics_interval,
        multimask=args.multimask,
        use_bbox=args.use_bbox,
    )

    # # model.load_state_dict(checkpoint)
    # val_dataloader = DataLoader(
    # val_dataset,
    # batch_size=args.batch_size,
    # shuffle=True,
    # )
    # train_datloader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #         )

    # 定义回调函数列表
    callbacks = [
        # 学习率监控器，每一步记录一次学习率
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{step}-{val_per_mask_iou:.4f}',
            save_last=True,  # 保存最新的模型
            save_top_k=args.save_topk,  # 保存最好的模型数量
            monitor="val_per_mask_iou",  # 监控的指标
            mode="max",  # 监控指标的最大值
            save_weights_only=True,  # 只保存模型权重
            every_n_train_steps=args.every_n_train_steps,  # 每训练args.metrics_interval步保存一次模型
        ),
    ]
    logger = TensorBoardLogger(save_dir=args.log_dir, name="mobile_sam_finetune")
    trainer = pl.Trainer(
        strategy='ddp' if NUM_GPUS > 1 else None,
        accelerator=DEVICE,
        devices=NUM_GPUS,
        precision=16,
        callbacks=callbacks,
        max_epochs=-1,
        logger=logger,
        max_steps=args.steps,
        val_check_interval=args.metrics_interval,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        profiler="simple",
        # accumulate_grad_batches=4
    )
    
    # trainer.validate(model=model, dataloaders=val_dataloader)
    trainer.fit(model)
    torch.save(model.model.state_dict(), os.path.join(args.output_dir, args.model_name))

if __name__ == "__main__":
    main()
