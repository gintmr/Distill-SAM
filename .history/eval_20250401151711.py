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
from eval_tools import calculate_pa_iou, inference_image, init_model, overlay_mask_on_image, clean_checkpoint_path

# 配置日志
log_filename = "mobilesam_inaturalist.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
os.environ['INFERENCE_MODE'] = "test"
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="vit_t", help="model type", choices=["vit_t","tiny_msam"])
    parser.add_argument("--checkpoint_path", type=str, default="/data2/wuxinrui/RA-L/MobileSAM/trained_models/Distilled_encoder/Distilled.pth", help="path to the checkpoint")
    # parser.add_argument("--checkpoint_path", type=str, default="/data2/wuxinrui/RA-L/MobileSAM/trained_models/standard_mimc/final_model.pth", help="path to the checkpoint")
    parser.add_argument("--test_img_path", type=str, default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/test_list", help="the test image path")
    parser.add_argument("--label_path", type=str, default="/data2/wuxinrui/Projects/ICCV/jsons_for_salient_instance_segmentation/test_5_prompts.json", help="the test json path")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    parser.add_argument("--visualize_mask_path", default="/data2/wuxinrui/RA-L/MobileSAM/modified_mobilesam_results")
    
    args = parser.parse_args()


    visualize_mask_path = args.visualize_mask_path
    
    if os.path.exists(visualize_mask_path):
        for file in os.listdir(visualize_mask_path):
            os.remove(os.path.join(visualize_mask_path, file))
    else:
        os.makedirs(visualize_mask_path)
        
    logging.info("Using device: " + device)
    logging.info(f"Devices num is {torch.cuda.device_count()}")
    logging.info(f"args: {args}")
    
    with open(args.label_path, 'r') as f:
        json_data = json.load(f)
        annotations = json_data['annotations']

    img_files = [f for f in os.listdir(args.test_img_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    bbox_pred_masks = []
    point_pred_masks = []

    bbox_pa_list, bbox_iou_list, point_pa_list, point_iou_list = [], [], [], []
    temp_checkpoint_path = clean_checkpoint_path(args.checkpoint_path)
    mask_generator, mask_predictor = init_model(model_type=args.model_type, sam_checkpoint=temp_checkpoint_path, device=device, generator=False, predictor=True)
    
    
    for img_file in tqdm(img_files[:50], miniters=50):
        
        img_path = os.path.join(args.test_img_path, img_file)
        ext_name = img_file.split('.')[-1]
        
        results_dict = inference_image(img_path, annotations, mask_generator, mask_predictor, device=device, bbox_prompt=True, point_prompt=True)
        
        predictor_results = results_dict['predictor_results']
        image_masks = results_dict['image_masks']
        

        bbox_pred_masks = predictor_results['bbox']['masks']
        point_pred_masks = predictor_results['point']['masks']
        
        output_path_bbox = os.path.join(visualize_mask_path, img_file.replace(ext_name, "_bbox_pred.png"))
        output_path_point = os.path.join(visualize_mask_path, img_file.replace(ext_name, "_point_pred.png"))
        
        overlay_mask_on_image(bbox_pred_masks, output_path_bbox, image_path=img_path)
        overlay_mask_on_image(point_pred_masks, output_path_point, image_path=img_path)
        
        
        pa_list, iou_list = calculate_pa_iou(bbox_pred_masks, image_masks)
        bbox_iou_list.extend(iou_list[i] for i in range(len(iou_list)))
        bbox_pa_list.extend(pa_list[i] for i in range(len(pa_list)))
        
        pa_list, iou_list = calculate_pa_iou(point_pred_masks, image_masks)
        point_iou_list.extend(iou_list[i] for i in range(len(iou_list)))
        point_pa_list.extend(pa_list[i] for i in range(len(pa_list)))
        
        
    bbox_avg_iou = np.mean(bbox_iou_list)
    bbox_avg_pa = np.mean(bbox_pa_list)
    point_avg_iou = np.mean(point_iou_list)
    point_avg_pa = np.mean(point_pa_list)

    print(f"bbox_avg_iou: {bbox_avg_iou}")
    print(f"bbox_avg_pa: {bbox_avg_pa}")
    print(f"point_avg_iou: {point_avg_iou}")
    print(f"point_avg_pa: {point_avg_pa}")
    
    logging.info(f"bbox_avg_iou: {bbox_avg_iou}")
    logging.info(f"bbox_avg_pa: {bbox_avg_pa}")
    logging.info(f"point_avg_iou: {point_avg_iou}")
    logging.info(f"point_avg_pa: {point_avg_pa}")
    
if __name__ == "__main__":
    main()