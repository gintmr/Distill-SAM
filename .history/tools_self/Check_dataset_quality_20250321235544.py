import json
from pycocotools.coco import COCO   
import os
import cv2

anno_path = "/data2/wuxinrui/Datasets/COCO/annotations/instances_train2017_sampled_5000.json"
image_folder_path = "data2/wuxinrui/Datasets/COCO/images/train2017"


def check_dataset_valid(anno_path, image_folder_path):
    coco = COCO(anno_path)
    imgIds = coco.getImgIds()[:] ## 得到的是anno文件中对应的图像id
    for imgId in imgIds:
        img_info = coco.loadImgs(imgId)[0]
        img_name = img_info['file_name']
        img_path = os.path.join(image_folder_path, img_name)
        if not os.path.exists(img_path):
             print(img_path, "not exists")
        else:
            img = cv2.imread(img_path)
            if img is None:
                print(img_path, "not read")
            else:
                h, w, c = img.shape
                if h <= 0 or w <= 0:
                    print(img_path, "invalid image size")
                else:
                    annIds = coco.getAnnIds(imgIds=imgId)
                    anns = coco.loadAnns(annIds)
                    if len(anns) == 0:
                        print(img_path, "no annotation")
                    else:
                        for ann in anns:
                            if ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
                                print(img_path, "invalid bbox")
                                break
                            if ann['area'] <= 0:
                                print(img_path, "invalid area")
                                break
                            if ann['category_id'] <= 0:
                                print(img_path, "invalid category_id")
                                break
                            if ann['iscrowd'] not in [0, 1]:
                                print(img_path, "invalid iscrowd")
                                break
                            if ann['segmentation'] is not None and len(ann['segmentation']) > 0:
                                if isinstance(ann['segmentation'][0], list):
                                    for seg in ann['segmentation']:
                                        if len(seg) % 2 != 0:
                                            print(img_path, "invalid segmentation")
                                            break
                                else:
                                    if len(ann['segmentation']) % 2 != 0:
                                        print(img_path, "invalid segmentation")
                                        break
                            if ann['keypoints'] is not None and len(ann['keypoints']) > 0:
                                if len(ann['keypoints']) % 3 != 0:
                                    print(img_path, "invalid keypoints")
                                    break
                                for kp in ann['keypoints']:
                                    if kp < 0 or kp > 1:
                                        print(img_path, "invalid keypoints")
                                        break
                        print(img_path, "checked")


        