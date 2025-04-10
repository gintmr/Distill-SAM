from mobile_sam import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(Sam)
masks = mask_generator.generate("/data2/wuxinrui/Datasets/UIIS/UDW/train/L_5.jpg")