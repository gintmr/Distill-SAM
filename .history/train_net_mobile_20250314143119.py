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
from sahi.utils.coco import Coco
from sahi.utils.cv import get_bool_mask_from_coco_segmentation
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
F

from transformers.models.maskformer.modeling_maskformer import dice_loss, sigmoid_focal_loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


NUM_WORKERS = 0  # https://github.com/pytorch/pytorch/issues/42518
NUM_GPUS = 2
DEVICE = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
from shapely.geometry import Point, Polygon


def Random_Points_in_Polygon(polygon, number):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    # 获取进程总数
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

# coco mask style dataloader
class Coco2MaskDataset(Dataset):
    def __init__(self, data_root, image_size=1024, annotation_path=None, length=150, num_points=4, use_centerpoint=False):
        self.data_root = data_root
        self.image_size = image_size
        self.use_centerpoint = use_centerpoint
        self.length = length
        self.coco = Coco.from_coco_dict_or_path(annotation_path)
        self.num_points = num_points
        self.transform = ResizeLongestSide(self.image_size)

    def __len__(self):
        return len(self.coco.images)

    def __getitem__(self, index):
        coco_image = self.coco.images[index]
        coco_image_name = coco_image.file_name
        image = np.array(Image.open(os.path.join(self.data_root, self.split, coco_image.file_name)).convert("RGB"))
        original_height, original_width = np.shape(image)[0], np.shape(image)[1]
        input_image = self.transform.apply_image(image)
        resized_height, resized_width = np.shape(input_image)[0], np.shape(input_image)[1]
        original_input_size =tuple((original_height,original_width))
        resized_input_size = tuple((resized_height, resized_width))
        bboxes = []
        masks = []
        center_point_labels = []
        center_points = []
        combined_points = []
        combined_point_labels = []
        category_ids = []
        count_label=1
        for annotation in coco_image.annotations[:self.length]:
            zzq_points=[]
            x, y, w, h = annotation.bbox
            # get scaled bbox in xyxy format
            bbox = [x , y , (x + w), (y + h) ]
            # mask = get_bool_mask_from_coco_segmentation(annotation.segmentation, original_width, original_height).astype(np.uint8)
            mask = get_bool_mask_from_coco_segmentation(annotation.segmentation, original_width,original_height)
            mask = cv2.resize(mask, (resized_width,resized_height), interpolation=cv2.INTER_LINEAR) # (width, height)
            mask = (mask > 0.5).astype(np.uint8)
            mask_polygon = Polygon(np.reshape(np.array(annotation.segmentation[0]), (-1, 2)))
            category_ids.append([annotation.category_id])
            points = Random_Points_in_Polygon(mask_polygon, self.num_points)
            for zzq_j in range(self.num_points):
                zzq_points.append(([points[zzq_j].xy[0][0], points[zzq_j].xy[1][0]]))
            center_points.append([[(bbox[0]+bbox[2])/2.0,(bbox[1]+bbox[3])/2.0]])
            bboxes.append(bbox)
            masks.append(mask)
            combined_point_labels.append([count_label]*self.num_points)
            combined_points.append(zzq_points)
            center_point_labels.append([count_label])

        center_points = np.stack(center_points,axis=0)
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
        input_image_torch=torch.tensor(input_image)
        input_image_torch=input_image_torch.permute(2, 0, 1).contiguous()
        return input_image_torch, torch.tensor(bboxes), torch.tensor(masks).long(), torch.tensor(given_points), torch.tensor(point_labels), \
               coco_image_name, torch.tensor(category_ids),original_input_size, resized_input_size
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks, center_points, point_labels, img_name, category_ids ,original_input_size, resized_input_size= zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks, center_points, point_labels, img_name, category_ids, original_input_size, resized_input_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="path to the data root")
    parser.add_argument("--model_type", type=str, required=True, help="model type", choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to the checkpoint")
    parser.add_argument("--annotation_name", type=str, default="annotations.json", help="the annotation file name")
    parser.add_argument("--freeze_image_encoder", action="store_true", help="freeze image encoder")
    parser.add_argument("--freeze_prompt_encoder", action="store_true", help="freeze prompt encoder")
    parser.add_argument("--freeze_mask_decoder", action="store_true", help="freeze mask decoder")
    parser.add_argument("--multimask", action="store_true", help="generate multi masks")
    parser.add_argument("--use_bbox", action="store_true", help="generate multi masks")
    parser.add_argument("--use_centerpoint", action="store_true", help="use only one center point")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--save_topk", type=int, default=3, help="save top K models")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    parser.add_argument("--steps", type=int, default=10000, help="number of steps")
    parser.add_argument("--num_points", type=int, default=4, help="number of random points")
    parser.add_argument("--length", type=int, default=150, help="the length of the chosen masks")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--metrics_interval", type=int, default=5000, help="interval for logging metrics")
    parser.add_argument("--output_dir", type=str, default="./exp/debug_erase", help="path to save the model")
    parser.add_argument("--model_name", type=str, default="erased_vit_b_points.pth", help="model name to save the model")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # load the dataset
    train_dataset = Coco2MaskDataset(data_root=args.data_root, split="train", image_size=args.image_size,annotation_name=args.annotation_name,
                                     length=args.length,num_points=args.num_points,use_centerpoint=args.use_centerpoint)
    val_dataset = Coco2MaskDataset(data_root=args.data_root, split="val", image_size=args.image_size,annotation_name=args.annotation_name,
                                   length=args.length,num_points=args.num_points,use_centerpoint=args.use_centerpoint)
    # create the model
    model = SAMFinetuner(
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

    # 定义回调函数列表
    callbacks = [
        # 学习率监控器，每一步记录一次学习率
        LearningRateMonitor(logging_interval='step'),
        # 模型检查点，每训练args.metrics_interval步保存一次模型，保存最新的模型和最好的模型
        ModelCheckpoint(
            dirpath=args.output_dir,  # 模型保存路径
            filename='{step}-{val_per_mask_iou:.4f}',  # 模型保存文件名，包含训练步数和验证集的mask_iou
            save_last=True,  # 保存最新的模型
            save_top_k=args.save_topk,  # 保存最好的模型数量
            monitor="val_per_mask_iou",  # 监控的指标
            mode="max",  # 监控指标的最大值
            save_weights_only=True,  # 只保存模型权重
            every_n_train_steps=args.metrics_interval,  # 每训练args.metrics_interval步保存一次模型
        ),
    ]
    
    trainer = pl.Trainer(
        strategy='ddp' if NUM_GPUS > 1 else None,
        accelerator=DEVICE,
        devices=NUM_GPUS,
        precision=32,
        callbacks=callbacks,
        max_epochs=-1,
        max_steps=args.steps,
        val_check_interval=args.metrics_interval,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )
    trainer.fit(model)
    torch.save(model.model.state_dict(), os.path.join(args.output_dir, args.model_name))

if __name__ == "__main__":
    main()
