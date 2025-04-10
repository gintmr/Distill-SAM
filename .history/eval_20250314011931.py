import os
import cv2
import json
import numpy as np
import argparse
import torch
from mobile_sam import sam_model_registry, SamPredictor
import pycocotools.mask as mask_util
from tqdm import tqdm
import logging
import pycocotools.mask as mask_utils
from eval_tools import calculate_pa_iou, inference_image, init_model

# 配置日志
log_filename = "mobilesam_inaturalist.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="vit_t", help="model type", choices=["vit_t"])
    parser.add_argument("--checkpoint_path", type=str, default="./weights/mobile_sam.pt", help="path to the checkpoint")
    parser.add_argument("--test_img_path", type=str, default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/test_list", help="the test image path")
    parser.add_argument("--label_path", type=str, default="/data2/wuxinrui/Projects/ICCV/jsons_for_salient_instance_segmentation/test_1_prompts.json", help="the test json path")
    parser.add_argument("--output_dir", type=str, default="/data2/wuxinrui/Projects/ICCV/evaluate_fm/datasets/outputs", help="path to save the output")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    parser.add_argument("--prompt", type=str, default="bbox", help="which kind of prompt",choices=['point','bbox'])
    
    args = parser.parse_args()
    prompt=args.prompt
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.info("Using device: " + device)
    logging.info(f"Devices num is {torch.cuda.device_count()}")

    with open(args.label_path, 'r') as f:
        json_data = json.load(f)
        annotations = json_data['annotations']

    img_files = [f for f in os.listdir(args.test_img_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    bbox_pred_masks = []
    point_pred_masks = []

    bbox_pa_list, bbox_iou_list, point_pa_list, point_iou_list = [], [], [], []
    
    mask_generator, mask_predictor = init_model(model_type=args.model_type, sam_checkpoint=args.checkpoint_path, device=device, generator=False, predictor=True)
    
    
    for img_file in tqdm(img_files):
        
        img_path = os.path.join(args.test_img_path, img_file)
        
        results_dict = inference_image(img_path, annotations, mask_generator, mask_predictor, device=device)
        
        predictor_results = results_dict['predictor_results']
        image_masks = results_dict['image_masks']
        

        bbox_pred_masks = predictor_results['bbox']['masks']
        point_pred_masks = predictor_results['point']['masks']
        
        
        pa_list, iou_list = calculate_pa_iou(bbox_pred_masks, image_masks)
        bbox_iou_list.append(iou_list[i] for i in range(len(iou_list)))
        bbox_pa_list.append(pa_list[i] for i in range(len(pa_list)))
        
        pa_list, iou_list = calculate_pa_iou(point_pred_masks, image_masks)
        point_iou_list.append(iou_list[i] for i in range(len(iou_list)))
        point_pa_list.append(pa_list[i] for i in range(len(pa_list)))
        
    bbox_avg_iou = np.mean(bbox_iou_list)
    bbox_avg_pa = np.mean(bbox_pa_list)
    
    point_avg_iou = np.mean(point_iou_list)
    point_avg_pa = np.mean(point_pa_list)

    print(f"bbox_avg_iou: {bbox_avg_iou}")
    print(f"bbox_avg_pa: {bbox_avg_pa}")
    
    print(f"point_avg_iou: {point_avg_iou}")
    print(f"point_avg_pa: {point_avg_pa}")

if __name__ == "__main__":
    main()