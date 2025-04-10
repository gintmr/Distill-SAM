import os
import cv2
import json
import numpy as np
import argparse
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pycocotools.mask as mask_util
from tqdm import tqdm
import logging

# 配置日志
log_filename = "mobilesam_inaturalist.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="vit_t", help="model type", choices=["vit_t"])
    parser.add_argument("--checkpoint_path", type=str, default="./weights/mobile_sam.pt", help="path to the checkpoint")
    parser.add_argument("--test_img_path", type=str, default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/test_list", help="the test image path")
    parser.add_argument("--label_path", type=str, default="/data2/wuxinrui/Projects/ICCV/jsons_for_salient_instance_segmentation/test_1_prompts.json", help="the test json path")
    parser.add_argument("--output_dir", type=str, default="/data2/wuxinrui/Projects/ICCV/evaluate_fm/datasets/outputs", help="path to save the output")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: " + device)

    # 初始化模型
    mobile_sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    mask_predictor = SamPredictor(mobile_sam)
    
    # 加载标注文件
    with open(args.label_path, 'r') as f:
        json_data = json.load(f)
        annotations = json_data['annotations']

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    img_files = [f for f in os.listdir(args.test_img_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    ious = []
    pas = []

    for img_file in tqdm(img_files):
        now_img_anno = [annotation for annotation in annotations if annotation['image_name'] == img_file]
        img_path = os.path.join(args.test_img_path, img_file)
        logging.info(f"Processing image: {img_path}")

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Failed to load image: {img_path}")
            continue

        # 转换为RGB格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 使用MobileSAM生成掩码
        mask_predictor.set_image(img)
        masks, _, _ = mask_predictor.predict(point_coords=point_coords, point_labels=point_labels,)

        # 初始化掩码列表
        pred_masks = []
        ori_masks = []

        for item_anno in now_img_anno:
            # 解码标注掩码
            ori_segmentation = item_anno["segmentation"]
            ori_mask = np.array(mask_util.decode(ori_segmentation), dtype=np.float32)
            ori_masks.append(ori_mask)

            # 找到与标注最匹配的预测掩码
            best_match_mask = None
            best_match_iou = 0.0
            for mask_info in generated_masks:
                pred_mask = mask_info['segmentation']
                # 计算预测掩码和原始分割之间的交并比（IoU）
                iou = mask_util.iou([pred_mask], [ori_segmentation], [False])[0][0]
                # 如果交并比大于最佳匹配交并比
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_mask = pred_mask

            if best_match_mask is not None:
                pred_masks.append(best_match_mask)
            else:
                pred_masks.append(np.zeros_like(ori_mask))  # 如果没有匹配的掩码，添加空掩码

        # 计算PA和IoU
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