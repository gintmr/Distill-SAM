import numpy as np
import cv2
from mobile_sam.utils.transforms import ResizeLongestSide
from torch.nn import functional as F
import torch

def random_resize(imgs):
    size_list = [128, 256, 368, 512, 720, 896, 1024]

    resized_imgs = []
    for img in imgs:
        target_size = np.random.choice(size_list)
        transform = ResizeLongestSide(target_size)
        img_np = img.permute(1, 2, 0).numpy() 
        img = transform.apply_image(img_np)
        
        resized_h, resized_w = img.shape[:2]
        deltah = 1024 - resized_h
        deltaw = 1024 - resized_w
        pad_l = np.random.randint(0, deltah)
        pad_t = np.random.randint(0, deltaw)
        pad_r = deltah - pad_l
        pad_b = deltaw - pad_t
        img = torch.from_numpy(img)
        img = F.pad(img, (pad_l, pad_r, pad_t, pad_b)) ## 左边填0，右边填padw。上0，下padh
        resized_imgs.append(img.to(imgs.device))
    resized_imgs = torch.stack(resized_imgs, dim=0)
    return resized_imgs