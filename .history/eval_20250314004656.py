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
from eval_tools import calculate_pa_iou, inference_image, 

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
    
    for img_file in tqdm(img_files):
        
        generator_results, predictor_results, image_annotations, img_name, img_h, img_w = inference_image(img_file, annotations, generator=False, predictor=True, bbox_prompt=True, point_prompt=True, model_type=args.model_type, sam_checkpoint=args.checkpoint_path, device=device)
        
        img_mask_bbox = np.zeros([img_h, img_w])
        img_mask_point = np.zeros([img_h, img_w])
        
        bbox_results = predictor_results['bbox']
        point_results = predictor_results['point']

        bbox_results_masks = bbox_results['masks']
        point_results_masks = point_results['masks']

        bbox_pred_masks.append(bbox_results_masks[0])
        point_pred_masks.append(point_results_masks[0])
        
        
        img_mask_bbox[bbox_results_masks[0]] = 255
        fortran_ground_truth_binary_mask = np.asfortranarray(bbox_results_masks[0])
        compressed_rle = mask_util.encode(fortran_ground_truth_binary_mask)
        pred.append(compressed_rle)
        ori_segmentation = item_anno["segmentation"]
        ori_mask = np.array(mask_util.decode(ori_segmentation), dtype=np.float32)
        ori_masks.append(ori_mask)
        
    pa_list, iou_list = calculate_pa_iou(pred_masks, ori_masks)
    # print(pa_list)
    # print(iou_list)
    
    ious.extend(iou_list)
    pas.extend(pa_list)

    avg_iou = np.mean(ious)
    avg_pa = np.mean(pas)

    logging.info(f"Average IoU: {avg_iou}")
    logging.info(f"Average PA: {avg_pa}")
    print(f"Average IoU: {avg_iou}")
    print(f"Average PA: {avg_pa}")




if __name__ == "__main__":
    main()