#! 👇全都加上下采样


export INFERENCE_MODE=train
#g 原生SAM 1,3,5,bbox
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --label_num 1 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/ori_SAM

python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --label_num 3 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/ori_SAM

python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --label_num 5 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/ori_SAM



export INFERENCE_MODE=train
#g MObileSAM 1,3,5,bbox
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt --ori_SAM False --label_num 1 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/mobilesam

python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt --ori_SAM False --label_num 3 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/mobilesam

python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt --ori_SAM False --label_num 5 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/mobilesam



export INFERENCE_MODE=train
#g modified_mobilesam
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/trained_models/wxr_modified_mobilesam/random_point_v2_bbox.pth --ori_SAM False --label_num 1 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/temp

python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/trained_models/wxr_modified_mobilesam/random_point_v2_bbox.pth --ori_SAM False --label_num 3 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/temp

python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/trained_models/wxr_modified_mobilesam/random_point_v2_bbox.pth --ori_SAM False --label_num 5 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/temp