import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
anno_path = "/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC/annotations/train.json"
image_folder_path = "/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC/images/train"

cocoed_anno = COCO(anno_path)

def resize_image(image, max_size=640):
    """
    等比例缩放图像，最长边不超过 max_size
    """
    h, w = image.shape[:2]
    scale = min(max_size / max(h, w), 1.0)  # 计算缩放比例
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_image, scale

def resize_annotations(coco, annotations, scale, seg_type):
    """
    等比例缩放标注中的分割掩码、标注框和关键点
    """
    
    for ann in annotations:
        if 'bbox' in ann:
            ann['bbox'] = [int(x * scale) for x in ann['bbox']]
        
        if 'segmentation' in ann:
            if seg_type == 'poly':
                ann['segmentation'] = [[int(x * scale) for x in seg] for seg in ann['segmentation']]
            elif seg_type == 'rle':
                mask = coco.annToMask(ann=ann)
                resized_mask = cv2.resize(mask, (int(mask.shape[1] * scale), int(mask.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
                rle = maskUtils.encode(np.asfortranarray(resized_mask))
                ann['segmentation'] = {
                'size': rle['size'],
                'counts': rle['counts'].decode('utf-8')
                }
        if 'keypoints' in ann:
            ann['keypoints'] = [int(x * scale) for x in ann['keypoints']]
    
    return annotations

def process_images_and_annotations(cocoed_anno, image_folder_path, output_image_folder, output_anno_path, max_size=640, seg_type="rle"):
    """
    处理所有图像和标注，保存缩放后的图像和更新后的标注
    """
    
    if not os.path.exists(output_image_folder):
        os.mkdir(output_image_folder)
    
    coco = cocoed_anno
    imgIds = coco.getImgIds()[:]
    new_images = []
    new_annotations = []

    for imgId in tqdm(imgIds):
        # 加载图像信息
        img_info = coco.loadImgs(imgId)[0]
        img_name = img_info['file_name']
        img_path = os.path.join(image_folder_path, img_name)
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: {img_path} not read.")
            continue
        
        resized_image, scale = resize_image(image, max_size)
        
        new_img_name = os.path.splitext(img_name)[0] + ".jpg"  # 统一保存为 JPG 格式
        new_img_path = os.path.join(output_image_folder, new_img_name)
        cv2.imwrite(new_img_path, resized_image)
        
        # 更新图像信息
        img_info['file_name'] = new_img_name
        img_info['height'], img_info['width'] = resized_image.shape[:2]
        new_images.append(img_info)
        
        # 加载并缩放标注
        annIds = coco.getAnnIds(imgIds=imgId)
        annotations = coco.loadAnns(annIds)
        resized_annotations = resize_annotations(coco, annotations, scale, seg_type)
        new_annotations.extend(resized_annotations)

    # 保存更新后的标注文件
    new_anno = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.loadCats(coco.getCatIds())
    }
    with open(output_anno_path, 'w') as f:
        json.dump(new_anno, f, indent=4)
    
    print(f"Resized images saved to {output_image_folder}")
    print(f"Updated annotations saved to {output_anno_path}")

def get_dataset_quality(cocoed_anno, image_folder_path):
    coco = cocoed_anno
    imgIds = coco.getImgIds()[:] ## 得到的是anno文件中对应的图像id

    H_W_infos = []

    for imgId in tqdm(imgIds):
        img_info = coco.loadImgs(imgId)[0]
        img_name = img_info['file_name']
        img_path = os.path.join(image_folder_path, img_name)
        image_array = cv2.imread(img_path)
        if image_array is None:
            print(img_path, "not read")
        else:
            h, w, c = image_array.shape
            H_W_info = [h, w]
            H_W_infos.append(H_W_info)

    return H_W_infos

def analyze_and_visualize(H_W_infos, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 将长宽信息转换为 NumPy 数组
    H_W_infos = np.array(H_W_infos)
    heights = H_W_infos[:, 0]
    widths = H_W_infos[:, 1]
    S = heights * widths
    # 计算统计信息
    mean_S = np.mean(S)
    std_S = np.std(S)
    min_S = np.min(S)
    max_S = np.max(S)
    mean_height = np.mean(heights)
    mean_width = np.mean(widths)
    std_height = np.std(heights)
    std_width = np.std(widths)
    min_height = np.min(heights)
    min_width = np.min(widths)
    max_height = np.max(heights)
    max_width = np.max(widths)

    # 保存统计信息到文本文件
    stats_file = os.path.join(output_folder, "dataset_quality_stats.txt")
    with open(stats_file, "w") as f:
        f.write("Dataset Quality Statistics:\n")
        f.write(f"Mean Height: {mean_height:.2f}\n")
        f.write(f"Mean Width: {mean_width:.2f}\n")
        f.write(f"Std Dev Height: {std_height:.2f}\n")
        f.write(f"Std Dev Width: {std_width:.2f}\n")
        f.write(f"Min Height: {min_height}\n")
        f.write(f"Min Width: {min_width}\n")
        f.write(f"Max Height: {max_height}\n")
        f.write(f"Max Width: {max_width}\n")
        f.write(f"Mean S: {mean_S:.2f}\n")
        f.write(f"Std Dev S: {std_S:.2f}\n")
        f.write(f"Min S: {min_S}\n")
        f.write(f"Max S: {max_S}\n")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(heights, bins=50, color='blue', alpha=0.7)
    plt.title("Height Distribution")
    plt.xlabel("Height")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(widths, bins=50, color='green', alpha=0.7)
    plt.title("Width Distribution")
    plt.xlabel("Width")
    plt.ylabel("Frequency")

    # 保存可视化图像
    plot_file = os.path.join(output_folder, "dataset_quality_plots.png")
    plt.savefig(plot_file)
    plt.close()

    print(f"Statistics saved to {stats_file}")
    print(f"Plots saved to {plot_file}")
    
H_W_infos = get_dataset_quality(cocoed_anno, image_folder_path)    
analyze_and_visualize(H_W_infos, "/data2/wuxinrui/RA-L/MobileSAM/tools_self/visual_datasets")

def check_dataset_valid(cocoed_anno, image_folder_path):
    coco = cocoed_anno
    imgIds = coco.getImgIds()[:] ## 得到的是anno文件中对应的图像id
    for imgId in imgIds:
        img_info = coco.loadImgs(imgId)[0]
        img_name = img_info['file_name']
        img_path = os.path.join(image_folder_path, img_name)
        if not os.path.exists(img_path):
             print(img_path, "not exists")
        else:
            img = cv2.imread(img_path)
            if img is None:
                print(img_path, "not read")
            else:
                h, w, c = img.shape
                if h <= 0 or w <= 0:
                    print(img_path, "invalid image size")
                else:
                    annIds = coco.getAnnIds(imgIds=imgId)
                    anns = coco.loadAnns(annIds)
                    if len(anns) == 0:
                        print(img_path, "no annotation")
                    else:
                        for ann in anns:
                            if ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
                                print(img_path, "invalid bbox")
                                break
                            if ann['area'] <= 0:
                                print(img_path, "invalid area")
                                break
                            if ann['category_id'] <= 0:
                                print(img_path, "invalid category_id")
                                break
                            if ann['iscrowd'] not in [0, 1]:
                                print(img_path, "invalid iscrowd")
                                break
                            if ann['segmentation'] is not None and len(ann['segmentation']) > 0:
                                if isinstance(ann['segmentation'][0], list):
                                    for seg in ann['segmentation']:
                                        if len(seg) % 2 != 0:
                                            print(img_path, "invalid segmentation")
                                            break
                                else:
                                    if len(ann['segmentation']) % 2 != 0:
                                        print(img_path, "invalid segmentation")
                                        break
                            if ann['keypoints'] is not None and len(ann['keypoints']) > 0:
                                if len(ann['keypoints']) % 3 != 0:
                                    print(img_path, "invalid keypoints")
                                    break
                                for kp in ann['keypoints']:
                                    if kp < 0 or kp > 1:
                                        print(img_path, "invalid keypoints")
                                        break
                        print(img_path, "checked")


