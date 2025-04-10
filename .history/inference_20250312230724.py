from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

# predictor = SamPredictor(mobile_sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)
img_path = "/data2/wuxinrui/Datasets/UIIS/UDW/train/L_5.jpg"
img = cv2.imread(img_path)
mask_generator = SamAutomaticMaskGenerator(mobile_sam)
masks = mask_generator.generate(img)
