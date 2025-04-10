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


def inference_image(image_path, annotations=None, mask_generator=None, mask_predictor=None, device="cuda", bbox_prompt=False, point_prompt=False):
    
    image_name = os.path.basename(image_path)
    image = preprocess_image(image_path)
    image_array = image
    
    image_masks = []
    image_annotations = None
    if annotations:
        image_annotations = [anno for anno in annotations if anno['image_name'] == image_name]
        for anno in image_annotations:
            gt_mask = np.array((mask_utils.decode(anno['segmentation'])), dtype=np.float32)
            image_masks.append(gt_mask)


    if image is None:
        print(f"!!!!!!!!!!!!!{image_path} is None")
        return None, None, None
    height, width = image.shape[:2]

    generator_results = mask_generator.generate(image) if mask_generator else None
    
    
    if mask_predictor:
        
        predictor_results = {
            'bbox': {
                'masks': [],
                'scores': [],
                'logits': []
            },
            'point': {
               'masks': [],
               'scores': [],
               'logits': [],
            }
        }
        
        mask_predictor.set_image(image)
        for anno in image_annotations:
            # start inference
            if bbox_prompt:
                input_box = np.array([
                        anno['bbox'][0], anno['bbox'][1],
                        anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]
                    ])
                masks, scores, logits = mask_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False
                )
                predictor_results['bbox']['masks'].append(masks[0])
                predictor_results['bbox']['scores'].append(scores)
                predictor_results['bbox']['logits'].append(logits)


            if point_prompt:
                points = np.array(anno['coords'])
                labels = np.ones(len(points), dtype=np.int32)
                masks, scores, logits = mask_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False
                )
                predictor_results['point']['masks'].append(masks[0])
                predictor_results['point']['scores'].append(scores)
                predictor_results['point']['logits'].append(logits)
            # end inference
    else:
        predictor_results = None
    
    results_dict = {
        'generator_results': generator_results,
        'predictor_results': predictor_results,
        'image_annotations': image_annotations,
        'image_masks': image_masks,
        'image_name': image_name,
        'height': height,
        'width': width,
        'image_array': image_array
    }
    return results_dict

