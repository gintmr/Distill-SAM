from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
import os
import argparse


argparse = argparse.ArgumentParser()
argparse.add_argument("--img_folder", type=str, default="/data2/wuxinrui/Datasets/UIIS/UDW/train")

model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

args = argparse.parse_args()
img_folder = args.img_folder
img_list = os.listdir(img_folder)

for img_name in img_list:
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)
    mask_generator = SamAutomaticMaskGenerator(mobile_sam)
    masks = mask_generator.generate(img)
    print(masks)
