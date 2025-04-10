import numpy as np
import cv2
from mobile_sam.utils.transforms import ResizeLongestSide

def random_resize(imgs):
    size_list = [128, 256, 368, 512, 720, 896, 1024]
    target_size = np.random.choice(size_list)
    transform = ResizeLongestSide(target_size)
    for img in imgs:        
        ori_h, ori_w =  img[-2:]
        padh = ori_h - target_size
        padw = ori_w - target_size
        x = F.pad(x, (0, padw, 0, padh)) ## 左边填0，右边填padw。上0，下padh
        return x