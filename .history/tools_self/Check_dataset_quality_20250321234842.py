import json
from pycocotools.coco import COCO


anno_path = "/data2/wuxinrui/Datasets/COCO/annotations/instances_train2017_sampled_5000.json"
image_folder_path = "data2/wuxinrui/Datasets/COCO/images/train2017"

coco = COCO(anno_path)
imgIds = coco.getImgIds()[:] ## 得到的是anno文件中对应的图像id
print(imgIds)