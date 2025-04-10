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
    parser.add_argument("--prompt", type=str, default="point", help="which kind of prompt",choices=['point','bbox'])
    
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
    ious = []
    pas = []
    
    mobile_sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    mask_predictor = SamPredictor(mobile_sam)
    
    for img_file in tqdm(img_files):
        now_img_anno = [annotation for annotation in annotations if annotation['image_name'] == img_file]
        img_path = os.path.join(args.test_img_path, img_file)

        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Failed to load image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.uint8(img)
            height,width=np.shape(img)[0],np.shape(img)[1]
        except:
            logging.info("Image truncated: " + img_file)
            continue
        
        mask_predictor.set_image(img)
        
        pred = []
        gt = []
        size = [height,width]
        ori_masks = []
        pred_masks = []

        for item_anno in now_img_anno:
            
            img_mask = np.zeros([height, width])
            ori_segmentation = item_anno["segmentation"]
            gt.append(ori_segmentation)
            ori_mask = np.array(mask_utils.decode(ori_segmentation), dtype=np.float32)
            ori_masks.append(ori_mask)
            if prompt=="bbox":
                input_box = np.array([item_anno['bbox'][0], item_anno['bbox'][1], item_anno['bbox'][0]+item_anno['bbox'][2], item_anno['bbox'][1]+item_anno['bbox'][3]])
                masks, _, _ = mask_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
            elif prompt=="point":
                points,labels=[],[]
                coordinates=item_anno['coords']
                for item_coords in coordinates:
                    points.append([item_coords[0], item_coords[1]])
                    labels.append(1)
                input_point = np.array(points)
                input_label = np.array(labels)
                masks, _, _ = mask_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )
                pred_masks.append(masks[0])
            else:
                print("not implemented")


            img_mask[masks[0]] = 255
            fortran_ground_truth_binary_mask = np.asfortranarray(masks[0])
            compressed_rle = mask_util.encode(fortran_ground_truth_binary_mask)

            pred.append(compressed_rle)

            
            ori_segmentation = item_anno["segmentation"]
            ori_mask = np.array(mask_util.decode(ori_segmentation), dtype=np.float32)
            ori_masks.append(ori_mask)
            
        pa_list, iou_list = calculate_pa_iou(pred_masks, ori_masks)
        ious.extend(iou_list)
        pas.extend(pa_list)

    # 计算平均值
    avg_iou = np.mean(ious)
    avg_pa = np.mean(pas)

    logging.info(f"Average IoU: {avg_iou}")
    logging.info(f"Average PA: {avg_pa}")
    print(f"Average IoU: {avg_iou}")
    print(f"Average PA: {avg_pa}")

def calculate_pa_iou(pred_masks, ori_masks):
    pa_list = []
    iou_list = []
    for pred_mask, ori_mask in zip(pred_masks, ori_masks):
        # 计算PA
        TP = np.sum((pred_mask == True) & (ori_mask == True))
        FP = np.sum((pred_mask == True) & (ori_mask == False))
        TN = np.sum((pred_mask == False) & (ori_mask == False))
        FN = np.sum((pred_mask == False) & (ori_mask == True))
        pa = (TP + TN) / (TP + TN + FP + FN)
        pa_list.append(pa)

        # 计算IoU
        intersection = np.sum((pred_mask == True) & (ori_mask == True))
        union = np.sum((pred_mask == True) | (ori_mask == True))
        iou = intersection / union if union != 0 else 0
        iou_list.append(iou)

    return pa_list, iou_list

if __name__ == "__main__":
    main()