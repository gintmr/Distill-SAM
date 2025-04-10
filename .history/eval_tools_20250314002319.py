import numpy as np
import cv2
import os
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pycocotools.mask as mask_utils

def init_model(generator=False, predictor=False, model_type="vit_t", sam_checkpoint='./weights/mobile_sam.pt', device="cuda"):
    mask_generator = None
    mask_predictor = None
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    if generator:
        mask_generator = SamAutomaticMaskGenerator(mobile_sam)
    if predictor:
        mask_predictor = SamPredictor(mobile_sam)
    return mask_generator, mask_predictor


def calculate_pa_iou(pred_masks, ori_masks):
    pa_list = []
    iou_list = []
    for pred_mask, ori_mask in zip(pred_masks, ori_masks):
        TP = np.sum((pred_mask == True) & (ori_mask == True))
        FP = np.sum((pred_mask == True) & (ori_mask == False))
        TN = np.sum((pred_mask == False) & (ori_mask == False))
        FN = np.sum((pred_mask == False) & (ori_mask == True))
        pa = (TP + TN) / (TP + TN + FP + FN)
        pa_list.append(pa)

        intersection = np.sum((pred_mask == True) & (ori_mask == True))
        union = np.sum((pred_mask == True) | (ori_mask == True))
        iou = intersection / union if union != 0 else 0
        iou_list.append(iou)

    return pa_list, iou_list

def preprocess_image(image_path):
    '''
    return
        image: np.array
    '''
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def inference_image(image_path, annotations, prompt_type, generator=False, predictor=False):
    image_name = os.path.basename(image_path)
    image_annotations = [anno for anno in annotations if anno['image_name'] == image_name]

    image = preprocess_image(image_path)
    
    if image is None:
        print(f"!!!!!!!!!!!!!{image_path} is None")
        return None, None, None

    height, width = image.shape[:2]
    
    mask_generator, mask_predictor = init_model(generator=generator, predictor=predictor)

    generator_masks = mask_generator.generate(image) if generator else None
    
    
    mask_predictor.set_image(image) if predictor else None
    

    predicted_masks = []
    ground_truth_masks = []

    for anno in image_annotations:
        gt_mask = mask_util.decode(anno['segmentation'])
        ground_truth_masks.append(gt_mask)

        if prompt_type == "bbox":
            input_box = np.array([
                anno['bbox'][0], anno['bbox'][1],
                anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]
            ])
            masks, _, _ = mask_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False
            )
        elif prompt_type == "point":
            points = np.array(anno['coords'])
            labels = np.ones(len(points), dtype=np.int32)
            masks, _, _ = mask_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False
            )
        else:
            logging.error(f"Unsupported prompt type: {prompt_type}")
            return None, None, None

        predicted_masks.append(masks[0])

    return predicted_masks, ground_truth_masks, image_name

