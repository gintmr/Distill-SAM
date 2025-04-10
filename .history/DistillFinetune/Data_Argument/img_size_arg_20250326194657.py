import numpy as np
import cv2

def random_resize(imgs):
    size_list = [128, 256, 368, 512, 720, 896, 1024]
    target_size = np.random.choice(size_list)
    
    for img in imgs:
        img = cv2.resize(img, (target_size, target_size))
        
    return imgs