from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
img_path = "/data2/wuxinrui/Datasets/UIIS/UDW/train/L_5.jpg"
img = cv2.imread(img_path)
mask_generator = SamAutomaticMaskGenerator(mobile_sam)
masks = mask_generator.generate(img)
'''
输出一个列表，每一位对应一个字典
字典键值：

sgementation => h,w e.g. 256*256
bbox
area
predicred_iou
stability_scores
crop_box
'''


print((masks))
