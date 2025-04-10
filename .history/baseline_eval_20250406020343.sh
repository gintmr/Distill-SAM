#! 👇全都加上下采样


export INFERENCE_MODE=train
#g 原生SAM 1,3,5,bbox
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/sam_vit_h_4b8939.pth --ori_SAM True

#g MObileSAM 1,3,5,bbox
