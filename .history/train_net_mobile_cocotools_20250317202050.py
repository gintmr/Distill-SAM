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
from mobile_sam.utils.transforms import ResizeLongestSide
from eval_tools import get_bool_mask_from_segmentation, random_croods_in_mask
from MobileSAMFintuner import MobileSAMFintuner
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
# module.py
import logging
# 使用 filename 参数配置日志文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="/data2/wuxinrui/RA-L/MobileSAM/train_mobilesam.log"  # 指定日志文件路径
)

logging.info("ceshi")

NUM_GPUS = 4
DEVICE = 'cuda'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# torch.cuda.set_per_process_memory_fraction(0.9, device=0)
# torch.cuda.set_per_process_memory_fraction(0.9, device=1)  
# torch.cuda.set_per_process_memory_fraction(0.9, device=2)  
# torch.cuda.set_per_process_memory_fraction(0.9, device=3)  

class Coco2MaskDataset(Dataset):

    def __init__(self, data_root, image_size=1024, annotation_path=None, length=150, num_points=4, use_centerpoint=False):
        self.coco = COCO(annotation_path)  # 使用 pycocotools 加载 COCO 数据集
        self.data_root = data_root
        self.image_size = image_size
        self.transform = ResizeLongestSide(self.image_size)
        self.length = length
        self.num_points = num_points
        self.use_centerpoint = use_centerpoint
        self.imgIds = self.coco.getImgIds()[:24]
        
    
    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, index):
        try:
            # 使用 pycocotools 获取图像信息
            img_id = self.imgIds[index]
            img_info = self.coco.loadImgs(img_id)[0]
            coco_image_name = img_info["file_name"]
            image_path = os.path.join(self.data_root, coco_image_name)
            image = np.array(Image.open(image_path).convert("RGB"))

            original_height, original_width = image.shape[0], image.shape[1]
            input_image = self.transform.apply_image(image)
            resized_height, resized_width = input_image.shape[0], input_image.shape[1]

            original_input_size = [original_height, original_width]
            resized_input_size = [resized_height, resized_width]

            annIds = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(annIds)

            bboxes = []
            masks = []
            center_point_labels = []
            center_points = []
            combined_points = []
            combined_point_labels = []
            category_ids = []
            count_label = 1

            for annotation in annotations[:self.length]:
                x, y, w, h = annotation["bbox"]

                bbox = [x, y, x + w, y + h]

                # 获取掩码
                mask = self.coco.annToMask(annotation)
                mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)

                category_ids.append([annotation["category_id"]])

                # 随机生成点
                points, num_points = random_croods_in_mask(mask=mask, num_croods=self.num_points)
                center_points.append([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
                bboxes.append(bbox)
                masks.append(mask)
                combined_point_labels.append([count_label] * num_points)
                combined_points.append(points)
                center_point_labels.append([count_label])

            center_points = np.stack(center_points, axis=0)
            combined_points = np.stack(combined_points, axis=0)
            category_ids = np.stack(category_ids, axis=0)

            if self.use_centerpoint:
                given_points = center_points
                point_labels = center_point_labels
            else:
                given_points = combined_points
                point_labels = combined_point_labels

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            point_labels = np.stack(point_labels, axis=0)

            input_image_torch = torch.tensor(input_image)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()

            return (
                input_image_torch,
                torch.tensor(bboxes),
                torch.tensor(masks).long(),
                torch.tensor(given_points),
                torch.tensor(point_labels),
                coco_image_name,
                torch.tensor(category_ids),
                original_input_size,
                resized_input_size,
            )
        except Exception as e:
            print("Error in loading image: ", coco_image_name)
            print("Error: ", e)
            return self.__getitem__((index+1) % len(self))
        
        
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks, center_points, point_labels, img_name, category_ids ,original_input_size, resized_input_size= zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks, center_points, point_labels, img_name, category_ids, original_input_size, resized_input_size


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_data", default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/train_list", type=str, required=False, help="path to the data root")
    parser.add_argument("--train_anno", default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/train-taxonomic_cleaned.json", type=str, required=False, help="path to the annotation file")

    parser.add_argument("--val_data", default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/val_list", type=str, required=False, help="path to the data root")
    parser.add_argument('--val_anno', default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/val-taxonomic_cleaned.json", )
    
    
    # parser.add_argument("--train_data", default="/data2/wuxinrui/Datasets/COCO/images/train2017", type=str, required=False, help="path to the data root")
    # parser.add_argument("--train_anno", default="/data2/wuxinrui/Datasets/COCO/annotations/instances_train2017_sampled_5000.json", type=str, required=False, help="path to the annotation file")

    # parser.add_argument("--val_data", default="/data2/wuxinrui/Datasets/COCO/images/val2017", type=str, required=False, help="path to the data root")
    # parser.add_argument('--val_anno', default="/data2/wuxinrui/Datasets/COCO/annotations/instances_val2017_sampled_2000.json", type=str, required=False, help="path to the annotation file")
    

    
    parser.add_argument("--model_type", default='vit_t', type=str, required=False, help="model type")
    parser.add_argument("--checkpoint_path", default="weights/mobile_sam.pt", type=str, required=False, help="path to the checkpoint")


    parser.add_argument("--freeze_image_encoder", default=False, action="store_true", help="freeze image encoder")
    parser.add_argument("--freeze_prompt_encoder", default=True, action="store_true", help="freeze prompt encoder")
    parser.add_argument("--freeze_mask_decoder", default=True, action="store_true", help="freeze mask decoder")
    
    parser.add_argument("--multimask", action="store_true", help="generate multi masks")
    parser.add_argument("--use_bbox", action="store_true", help="generate multi masks")
    parser.add_argument("--use_centerpoint", action="store_true", help="use only one center point")
    
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--save_topk", type=int, default=3, help="save top K models")
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--steps", type=int, default=10000, help="number of steps")
    parser.add_argument("--num_points", type=int, default=3, help="number of random points")
    parser.add_argument("--length", type=int, default=150, help="the length of the chosen masks")
    
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--metrics_interval", type=int, default=None, help="interval for logging metrics")
    
    parser.add_argument("--output_dir", type=str, default="./trained_models/mobile", help="path to save the model")
    
    parser.add_argument("--model_name", type=str, default="final_model.pth", help="model name to save the model")
    parser.add_argument('--every_n_train_steps', default=5000)


    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # load the dataset
    train_dataset = Coco2MaskDataset(data_root=args.train_data, annotation_path=args.train_anno, image_size=args.image_size,
                                     length=args.length,num_points=args.num_points,use_centerpoint=args.use_centerpoint)
    val_dataset = Coco2MaskDataset(data_root=args.val_data, annotation_path=args.val_anno, image_size=args.image_size,
                                   length=args.length, num_points=args.num_points,use_centerpoint=args.use_centerpoint)
    # create the model
    model = MobileSAMFintuner(
        args.model_type,
        args.checkpoint_path,
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
    
   
    # model.load_state_dict(checkpoint)
    val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    )
    train_datloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
            )

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
    
    trainer = pl.Trainer(
        strategy='ddp' if NUM_GPUS > 1 else None,
        accelerator=DEVICE,
        devices=NUM_GPUS,
        precision=16,
        callbacks=callbacks,
        max_epochs=-1,
        max_steps=args.steps,
        val_check_interval=10,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )
    
    # trainer.validate(model=model, dataloaders=val_dataloader)
    trainer.fit(model)
    torch.save(model.model.state_dict(), os.path.join(args.output_dir, args.model_name))

if __name__ == "__main__":
    main()
