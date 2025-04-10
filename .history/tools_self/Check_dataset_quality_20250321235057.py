import json
from pycocotools.coco import COCO   
import os


anno_path = "/data2/wuxinrui/Datasets/COCO/annotations/instances_train2017_sampled_5000.json"
image_folder_path = "data2/wuxinrui/Datasets/COCO/images/train2017"


def check_dataset_quality(anno_path, image_folder_path):
    coco = COCO(anno_path)
    imgIds = coco.getImgIds()[:] ## 得到的是anno文件中对应的图像id
    for imgId in imgIds:
        img_info = coco.loadImgs(imgId)[0]
        img_name = img_info['file_name']
        img_path = image_folder_path + "/" + img_name
        