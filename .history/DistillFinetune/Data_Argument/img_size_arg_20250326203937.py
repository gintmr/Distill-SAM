import numpy as np
import cv2
from mobile_sam.utils.transforms import ResizeLongestSide
from torch.nn import functional as F
import torch
import numpy as np
import torch
import torch.nn.functional as F

def random_resize(imgs):
    size_list = [128, 256, 368, 512, 720, 896, 1024]
    bs, c, h_original, w_original = imgs.shape  # 获取原始尺寸
    resized_imgs = []
    
    for img in imgs:
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        valid_sizes = [s for s in size_list if s <= max(h_original, w_original)]
        target_size = np.random.choice(valid_sizes or [max(h_original, w_original)])
        
        transform = ResizeLongestSide(target_size)
        resized_img = transform.apply_image(img_np)  # (H, W, 3)
        
        h_new, w_new = resized_img.shape[:2]
        padh = h_original - h_new
        padw = w_original - w_new
        
        pad_top = np.random.randint(0, padh + 1) if padh > 0 else 0
        pad_bottom = padh - pad_top
        
        pad_left = np.random.randint(0, padw + 1) if padw > 0 else 0
        pad_right = padw - pad_left
        
        img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1)  # (C, H, W)
        padded_img = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom))
        
        resized_imgs.append(padded_img.to(imgs.device))

    return torch.stack(resized_imgs, dim=0)  # (B, C, H, W)