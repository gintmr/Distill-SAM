
import os
import argparse

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
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.profiler import profile, record_function, ProfilerActivity
import random


class Coco2MaskDataset(Dataset):

    def __init__(self, data_root, image_size=None, annotation_path=None, length=150, num_points=5, use_centerpoint=False):
        self.coco = COCO(annotation_path)  # 使用 pycocotools 加载 COCO 数据集
        self.data_root = data_root
        assert image_size is not None, "image_size must be specified"
        self.image_size = image_size
        self.transform = ResizeLongestSide(self.image_size) #自定义最长边
        self.length = length #G the number of masks to load
        self.num_points = num_points
        self.use_centerpoint = use_centerpoint
        self.imgIds = self.coco.getImgIds()[:]
        
        self.augmentations = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度和对比度
            T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),  # 随机高斯模糊
        ])
    
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        ## 将resize后的图像padding至image_encoder接受的长度
        # Normalize colors

        h, w = x.shape[:2]
        padh = self.image_size - h
        padw = self.image_size - w

        x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)  # 假设 x 是 (H, W, C) 格式
        return x
    
    def preprocess_mask(self, x):
        """Normalize pixel values and pad to a square input."""
        ## 将resize后的图像padding至image_encoder接受的长度
        # Normalize colors

        h, w = x.shape[:2]
        padh = self.image_size - h
        padw = self.image_size - w

        x = np.pad(x, ((0, padh), (0, padw)), mode='constant', constant_values=0)  # 假设 x 是 (H, W, C) 格式
        return x
        
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
            
            input_image = self.transform.apply_image(image) ## 根据设置的最长边resize图像
            
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

                # resize掩码
                mask = self.coco.annToMask(annotation)
                ## 保持原来的尺寸
                category_ids.append([annotation["category_id"]])
                points, num_points = random_croods_in_mask(mask=mask, num_croods=self.num_points) ## points的坐标顺序与mask相同,W\H
                
                mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)
                mask = self.preprocess_mask(mask)
                # points = self.transform.apply_coords_torch(points, original_input_size)
                # bbox = self.transform.apply_boxes_torch(bbox, original_input_size)
                
                ## bbox需要resize,同理center_points需要resize, points也需要resize; 但是mask不resize
                
                
                center_points.append([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
                bboxes.append(bbox)
                masks.append(mask)
                
                #G assert mask has the same size as original_input_size
                # assert mask.shape == (original_input_size[0], original_input_size[1])
                combined_point_labels.append([count_label] * num_points)
                combined_points.append(points)
                center_point_labels.append([count_label])

            combined_points = self.transform.apply_coords_torch(combined_points, original_input_size)
            center_points = self.transform.apply_coords_torch(center_points, original_input_size)
            bboxes = self.transform.apply_boxes_torch(bboxes, original_input_size)
            
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

            # 将输入图像转换为torch张量
            input_image = self.preprocess(input_image)
            input_image_torch = torch.tensor(input_image)
            
            #g 将张量的维度从HWC转换为CHW
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
            
            prompt_type = random.choice(['1_point', '3_points', '5_points'])
            
            if prompt_type == "1_point":
                given_points = given_points[:, :1]
                point_labels = point_labels[:, :1]
            elif prompt_type == "3_points":
                given_points = given_points[:, :3]
                point_labels = point_labels[:, :3]
            elif prompt_type == "5_points":
                given_points = given_points[:, :5]
                point_labels = point_labels[:, :5]


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
                str(coco_image_name),
            )
        except Exception as e:
            print("Error in loading image: ", coco_image_name)
            print("Error: ", e)
            return self.__getitem__((index+1) % len(self))
        
        
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks, center_points, point_labels, img_name, category_ids ,original_input_size, resized_input_size, coco_image_names = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks, center_points, point_labels, img_name, category_ids, original_input_size, resized_input_size, coco_image_names
