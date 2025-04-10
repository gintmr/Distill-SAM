import os
import cv2
import numpy as np
import pycocotools.mask as mask_util
import json
from pycocotools.coco import COCO

image_paths = ["Apogon_atradorsatus###382793077.jpeg", "Apogon_atradorsatus###422122967.jpg", "Apogon_atradorsatus###444682610.jpg","Seriola_dumerili###185196588.jpg"]  ## 需要可视化的图片名列表

def get_ann_ids(annotation_file, image_id, category_id):
    """
    获取指定图像的注释 ID。
    :param json_path: COCO 数据集的 JSON 文件路径
    :param image_id: 图像 ID
    :return: 注释 ID 列表
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
        annotations = data['annotations']
        for anno in annotations:
            if anno['image_id'] == image_id and anno['category_id'] == category_id:
                print(anno['id'])
                return anno['id']
        
def get_masks(coco, image_id):
    """
    根据 COCO 对象、image_id 和 category_id 生成掩码。
    :param coco: COCO 对象
    :param image_id: 图像 ID
    :param category_id: 类别 ID（可选）
    :return: 二值掩码
    """
    
    masks = []
    
    image_info = coco.loadImgs(image_id)[0]
    width, height = image_info["width"], image_info["height"]
    image_name = image_info['file_name']
    image_id = image_info['id']
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
        annotations = data['annotations']
        for anno in annotations:
            if anno['image_id'] == image_id:
                ##--##
                #RLE格式掩码 ~ 其他格式自行拓展
                ##--##
                rle = anno['segmentation']
                mask = np.array(mask_util.decode(rle), dtype=np.uint8)
                mask[mask != 0] = 255
                masks.append(mask)
    
    return masks

def overlay_mask_on_image(image_path, mask, output_path, mask_color=(178, 102, 255), alpha=0.5):
    """
    将掩码叠加到原图上并保存。
    :param image_path: 原图路径
    :param mask: 掩码
    :param output_path: 输出路径
    :param mask_color: 掩码颜色 (B, G, R)
    :param alpha: 掩码透明度 (0 到 1)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    mask_image = np.zeros_like(image)
    mask_image[mask > 0] = mask_color

    overlay = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)
    cv2.imwrite(output_path, overlay)
    print(f"Saved: {output_path}")


def overlay_masks_on_image(image_path, masks, output_path, mask_color=(178, 102, 255), alpha=0.5):
    """
    将多个掩码叠加到原图上并保存。
    :param image_path: 原图路径
    :param masks: 掩码列表
    :param output_path: 输出路径
    :param mask_color: 掩码颜色 (B, G, R)
    :param alpha: 掩码透明度 (0 到 1)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return
    # 合并所有掩码
    combined_mask = np.zeros_like(image[:, :, 0], dtype="uint8")
    for mask in masks:
        combined_mask[mask > 0] = 1

    # 创建掩码图像
    mask_image = np.zeros_like(image)
    mask_image[combined_mask > 0] = mask_color

    # 将掩码叠加到原图上
    overlay = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)
    cv2.imwrite(output_path, overlay)
    print(f"Saved: {output_path}")
    
    
def main(annotation_file, image_folder, output_folder):
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()

    os.makedirs(output_folder, exist_ok=True)

    for img_id in img_ids:
        image_info = coco.loadImgs(img_id)[0]
        if image_info["file_name"] in image_paths:
            image_path = os.path.join(image_folder, image_info["file_name"])
            output_path = os.path.join(output_folder, image_info["file_name"])

            masks = get_masks(coco, img_id)
            
            overlay_masks_on_image(image_path, masks, output_path, mask_color=(178, 102, 255), alpha=0.5)

if __name__ == "__main__":
    annotation_file = "/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC/annotations/train.json"  # 替换为 COCO JSON 文件路径
    image_folder = "/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC/images/train"                  # 替换为 COCO 图像文件夹路径
    output_folder = "./GT-visual-results"                         # 替换为输出文件夹路径

    main(annotation_file, image_folder, output_folder)