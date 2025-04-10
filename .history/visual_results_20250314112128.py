import os
import cv2
import json
import numpy as np
import argparse
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
import pycocotools.mask as mask_util
from tqdm import tqdm
import logging
import pycocotools.mask as mask_utils
from eval_tools import calculate_pa_iou, init_mask_generator
from eval_tools import init_model, inference_image


input_folder = "/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/test_list"
output_folder = "/data2/wuxinrui/RA-L/MobileSAM/visual_results"


def overlay_masks_on_image(image, masks, output_path):
    overlay = np.zeros_like(image)
    
    for mask in masks:
        segmentation = mask["segmentation"].astype(np.uint8) * 255  # 将布尔掩码转换为255的掩码
        color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)  # 随机生成颜色
        overlay[segmentation > 0] = color  # 将掩码区域填充为随机颜色

    # 将掩码叠加到原始图像上
    alpha = 0.5  # 设置透明度
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # 保存结果
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")


os.makedirs(output_folder, exist_ok=True)

mask_generator, mask_predictor = init_model(model_type="vit_t", sam_checkpoint="./weights/mobile_sam.pt", device="cuda", generator=False, predictor=True)
# 遍历文件夹中的所有图片
for filename in os.listdir(input_folder)[:10]:
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        
        
        # 将掩码叠加到原始图像上并保存
        overlay_masks_on_image(image, masks, output_path)

print("All images processed and saved.")