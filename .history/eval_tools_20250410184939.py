import numpy as np
import cv2
import os
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pycocotools.mask as mask_utils
from segment_anything.build_sam import sam_model_registry as SAM_model_registry
import matplotlib.pyplot as plt

def combine_visualize_results(pred_mask_array, GT_mask_array, point_array, save_path):
    """
    将三张图像（预测掩码图、真实掩码图和点标记图）绘制在一起并保存到文件。

    参数:
        pred_mask_array (numpy.ndarray): 预测掩码图的数组，形状为 (H, W, 3)，BGR格式。
        GT_mask_array (numpy.ndarray): 真实掩码图的数组，形状为 (H, W, 3)，BGR格式。
        point_array (numpy.ndarray): 点标记图的数组，形状为 (H, W, 3)，BGR格式。
        save_path (str): 保存图像的路径。
    """


    # 创建一个图形窗口
    plt.figure(figsize=(15, 5))

    # 绘制预测掩码图
    plt.subplot(1, 3, 1)  # 1行3列的第1个
    plt.imshow(pred_mask_array)
    plt.title("Predicted Mask")
    plt.axis("off")  # 关闭坐标轴

    # 绘制真实掩码图
    plt.subplot(1, 3, 2)  # 1行3列的第2个
    plt.imshow(GT_mask_array)
    plt.title("Ground Truth Mask")
    plt.axis("off")  # 关闭坐标轴

    # 绘制点标记图
    plt.subplot(1, 3, 3)  # 1行3列的第3个
    plt.imshow(point_array)
    plt.title("Points Overlay")
    plt.axis("off")  # 关闭坐标轴

    # 自动调整子图参数，确保图像之间没有重叠
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 关闭图形窗口，释放资源

# 示例调用
# 假设 pred_mask_array, GT_mask_array, point_array 已经通过相关函数生成
# save_images(pred_mask_array, GT_mask_array, point_array, save_path="output_images/combined_image.png")



def init_model(generator=False, predictor=False, model_type="vit_t", sam_checkpoint='./weights/mobile_sam.pt', device="cuda", ori_SAM=False):

    mask_generator = None
    mask_predictor = None
    if ori_SAM:
        mobile_sam = SAM_model_registry[model_type](checkpoint=sam_checkpoint)
    else:
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
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().detach().numpy() > 0
    if isinstance(ori_masks, torch.Tensor):
        ori_masks = ori_masks.cpu().detach().numpy() > 0

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



def calculate_metrics(pred_masks, ori_masks):
    pa_list = []
    iou_list = []
    dice_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    mae_list = []
    mse_list = []

    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().detach().numpy() > 0
    if isinstance(ori_masks, torch.Tensor):
        ori_masks = ori_masks.cpu().detach().numpy() > 0

    for pred_mask, ori_mask in zip(pred_masks, ori_masks):
        TP = np.sum((pred_mask == True) & (ori_mask == True))
        FP = np.sum((pred_mask == True) & (ori_mask == False))
        TN = np.sum((pred_mask == False) & (ori_mask == False))
        FN = np.sum((pred_mask == False) & (ori_mask == True))

        # Pixel Accuracy (PA)
        pa = (TP + TN) / (TP + TN + FP + FN)
        pa_list.append(pa)

        # Intersection over Union (IoU)
        intersection = np.sum((pred_mask == True) & (ori_mask == True))
        union = np.sum((pred_mask == True) | (ori_mask == True))
        iou = intersection / union if union != 0 else 0
        iou_list.append(iou)

        # Dice Similarity Coefficient (Dice)
        dice = (2 * intersection) / (np.sum(pred_mask) + np.sum(ori_mask)) if (np.sum(pred_mask) + np.sum(ori_mask)) != 0 else 0
        dice_list.append(dice)

        # Precision
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        precision_list.append(precision)

        # Recall
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        recall_list.append(recall)

        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_list.append(f1)

        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(pred_mask.astype(np.float32) - ori_mask.astype(np.float32)))
        mae_list.append(mae)

        # Mean Squared Error (MSE)
        mse = np.mean((pred_mask.astype(np.float32) - ori_mask.astype(np.float32)) ** 2)
        mse_list.append(mse)

    return {
        "PA": pa_list,
        "IoU": iou_list,
        "Dice": dice_list,
        "Precision": precision_list,
        "Recall": recall_list,
        "F1 Score": f1_list,
        "MAE": mae_list,
        "MSE": mse_list
    }
    
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
    '''
    in_args:
        image_path - path to current image
        annotations - list of annotations for current image (which has point/bbox prompts for predicting)
        mask_generator - mask generator for generating masks (init in the init_model function)
        mask_predictoy - mask predictor for predicting masks (init in the init_model function)
        device - device to run the model on
        bbox_prompt - whether to use bbox prompt for predicting
        point_prompt - whether to use point prompt for predicting
    out_args:
        results_dict - dict of results for current image by prompts
    '''
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

    
    
    
def get_bool_mask_from_segmentation(segmentation, height, width) -> np.ndarray:
    """
    Convert a segmentation mask to a boolean mask.
    """
    size = (height, width)
    if "size" not in segmentation:
        points = [np.array(point).reshape(-1, 2).round().astype(int) for point in segmentation]
        bool_mask = np.zeros(size)
        bool_mask = cv2.fillPoly(bool_mask, points, (1.0,))
        bool_mask.astype(bool)
        return bool_mask
    else:
        rle = segmentation
        mask = np.array(mask_utils.decode(rle), dtype=np.uint8)
        bool_mask = mask.astype(bool)
        return bool_mask
        
def random_croods_in_mask(mask, num_croods=1):
    '''
    generate croods in mask where > 0
    mask shape: H,W
    '''
    mask_T = mask.T
    #g       mask_T -> W,H
    croods_to_chose = np.argwhere(mask_T > 0)
    
    if len(croods_to_chose) < num_croods:
        return croods_to_chose, len(croods_to_chose)
    
    selected_croods = croods_to_chose[np.random.choice(len(croods_to_chose), num_croods, replace=False)]
    
    return selected_croods, num_croods

def overlay_point_on_image(center_points, output_path, image_path=None, image_array=None, array_out=False):
    """
    将点叠加到原图上并保存。
    :param image_path: 原图路径
    :param center_points: [N,m,2]，代表一张图N个mask，每个mask选取m个点
    :param output_path: 输出路径
    """
    if image_array is not None:
        image = image_array
    if image_path:
        image = cv2.imread(image_path) # H,W,C

    if image is None:
        print(f"Error: Unable to read image at {image_path} OR image_array is None")
        return
    else:
        dims = image.shape
        if len(dims) == 3:
            if dims[0] in [1, 3, 4]:
                # print("检测到 CHW 格式，正在转换为 HWC...")
                image = image.permute((1, 2, 0))
            elif dims[2] in [1, 3, 4]:
                image = image
                # print("图像已经是 HWC 格式，无需转换。")
            else:
                raise ValueError("无法确定图像的维度顺序。")
        elif len(dims) == 2:
            # print("检测到灰度图，自动添加通道维度...")
            image = image[..., np.newaxis]
        else:
            raise ValueError("图像维度不正确。")
        
    center_points = center_points.cpu().detach().numpy()
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        image = np.ascontiguousarray(image)
    for mask_points in center_points:
        random_color = (
                int(np.random.randint(0, 256)),  # B
                int(np.random.randint(0, 256)),  # G
                int(np.random.randint(0, 256))   # R
            )
        for point in mask_points:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, random_color, -1)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # print(f"Saved: {output_path}")
    if array_out:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    

def overlay_mask_on_image(mask, output_path, image_path=None, image_array=None, mask_color=(178, 102, 255), alpha=0.5, array_out=False, save_img=True):
    """
    将掩码叠加到原图上并保存。
    :param image_path: 原图路径
    :param mask: 掩码
    :param output_path: 输出路径
    :param mask_color: 掩码颜色 (B, G, R)
    :param alpha: 掩码透明度 (0 到 1)
    """
    if image_array is not None:
        image = image_array
    if image_path:
        image = cv2.imread(image_path) # H,W,C

    if image is None:
        print(f"Error: Unable to read image at {image_path} OR image_array is None")
        return
    else:
        dims = image.shape
        if len(dims) == 3:
            if dims[0] in [1, 3, 4]:
                # print("检测到 CHW 格式，正在转换为 HWC...")
                image = image.permute((1, 2, 0))
            elif dims[2] in [1, 3, 4]:
                image = image
                # print("图像已经是 HWC 格式，无需转换。")
            else:
                raise ValueError("无法确定图像的维度顺序。")
        elif len(dims) == 2:
            # print("检测到灰度图，自动添加通道维度...")
            image = image[..., np.newaxis]
        else:
            raise ValueError("图像维度不正确。")
    
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
        
    overlay = image.copy()
    mask_image = np.zeros_like(image)
    if isinstance(mask, list) or len(mask.shape) == 3:
        
        colors = np.random.randint(0, 256, size=(len(mask), 3), dtype=np.uint8)
        # mask_3channel = mask[0][:, :, np.newaxis].repeat(3, axis=2)
        for mask_s, mask_color in zip(mask, colors):
            
            pad_height = max(0, mask_image.shape[0] - mask_s.shape[0])
            pad_width = max(0, mask_image.shape[1] - mask_s.shape[1])
            mask_s = np.pad(mask_s,((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
            mask_image[mask_s > 0] = mask_color
    else:

        mask_image[mask > 0] = mask_color
        
    overlay = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)) if save_img else None
        # print(f"Saved: {output_path}")
    if array_out:
        return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)



def overlay_masks_on_image(image, masks, output_path):
    overlay = np.zeros_like(image)
    for mask in masks:
        if mask is dict:
            segmentation = mask["segmentation"].astype(np.uint8) * 255
            color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
            overlay[segmentation > 0] = color
        else:
            segmentation = mask.astype(np.uint8) * 255
            color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
            overlay[segmentation > 0] = color 
    alpha = 0.5
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")
    
    
    
def clean_checkpoint_path(check_point_path, train=False):
    if train:
        temp_path = '/data2/wuxinrui/RA-L/MobileSAM/weights/temp_weights/train.pth'
    else:
        temp_path = '/data2/wuxinrui/RA-L/MobileSAM/weights/temp_weights/test.pth'
    check_point = torch.load(check_point_path, map_location='cuda')
    if ".ckpt" in check_point_path:
        check_point = check_point['state_dict']
    temp_check_point = {}
    for k, v in check_point.items():
        if "model." in k:
            temp_check_point[k.replace("model.", "")] = v
        else:
            temp_check_point[k] = v
    torch.save(temp_check_point, temp_path)

    return temp_path