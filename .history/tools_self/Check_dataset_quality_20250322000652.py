import json
from pycocotools.coco import COCO   
import os
import cv2
import numpy as np

anno_path = "/data2/wuxinrui/Datasets/COCO/annotations/instances_train2017_sampled_5000.json"
image_folder_path = "data2/wuxinrui/Datasets/COCO/images/train2017"

cocoed_anno = COCO(anno_path)

def get_dataset_quality(cocoed_anno, image_folder_path):
    coco = cocoed_anno
    imgIds = coco.getImgIds()[:] ## 得到的是anno文件中对应的图像id

    H_W_infos = []

    for imgId in imgIds:
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

    # 绘制高度和宽度的分布图
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


        