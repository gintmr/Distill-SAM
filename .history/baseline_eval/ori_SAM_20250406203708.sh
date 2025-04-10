# #! 👇全都加上下采样

export CUDA_VISIBLE_DEVICES=0
export INFERENCE_MODE=train
#g 原生SAM 1,3,5,bbox
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --label_num 1 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/ori_SAM

python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --label_num 3 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/ori_SAM

python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --label_num 5 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/ori_SAM



# export INFERENCE_MODE=train
# #g MObileSAM 1,3,5,bbox
# python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt --ori_SAM False --label_num 1 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/mobilesam

# python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt --ori_SAM False --label_num 3 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/mobilesam

# python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt --ori_SAM False --label_num 5 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/mobilesam



# export INFERENCE_MODE=train
# #g modified_mobilesam
# python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/trained_models/wxr_modified_mobilesam/random_point_v2_bbox.pth --ori_SAM False --label_num 1 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/temp

# python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/trained_models/wxr_modified_mobilesam/random_point_v2_bbox.pth --ori_SAM False --label_num 3 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/temp

# python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/trained_models/wxr_modified_mobilesam/random_point_v2_bbox.pth --ori_SAM False --label_num 5 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/temp



#!/bin/bash

# 设置第一组代码运行在 GPU 0 上
export CUDA_VISIBLE_DEVICES=0
export INFERENCE_MODE=train
echo "Running original SAM with label_num 1 on GPU 0"
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --model_type vit_h --label_num 1 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/ori_SAM &

# 设置第二组代码运行在 GPU 1 上
export CUDA_VISIBLE_DEVICES=1
export INFERENCE_MODE=train
echo "Running MobileSAM with label_num 1 on GPU 1"
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt --ori_SAM False --label_num 1 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/mobilesam &

# 设置第三组代码运行在 GPU 2 上
export CUDA_VISIBLE_DEVICES=2
export INFERENCE_MODE=train
echo "Running modified MobileSAM with label_num 1 on GPU 2"
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/trained_models/wxr_modified_mobilesam/random_point_v2_bbox.pth --ori_SAM False --label_num 1 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/temp &

# 等待第一轮运行完成
wait

# 设置第一组代码运行在 GPU 0 上
export CUDA_VISIBLE_DEVICES=0
echo "Running original SAM with label_num 3 on GPU 0"
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --model_type vit_h --label_num 3 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/ori_SAM &

# 设置第二组代码运行在 GPU 1 上
export CUDA_VISIBLE_DEVICES=1
echo "Running MobileSAM with label_num 3 on GPU 1"
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt --ori_SAM False --label_num 3 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/mobilesam &

# 设置第三组代码运行在 GPU 2 上
export CUDA_VISIBLE_DEVICES=2
echo "Running modified MobileSAM with label_num 3 on GPU 2"
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/trained_models/wxr_modified_mobilesam/random_point_v2_bbox.pth --ori_SAM False --label_num 3 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/temp &

# 等待第二轮运行完成
wait

# 设置第一组代码运行在 GPU 0 上
export CUDA_VISIBLE_DEVICES=0
echo "Running original SAM with label_num 5 on GPU 0"
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --model_type vit_h --label_num 5 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/ori_SAM &

# 设置第二组代码运行在 GPU 1 上
export CUDA_VISIBLE_DEVICES=1
echo "Running MobileSAM with label_num 5 on GPU 1"
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt --ori_SAM False --label_num 5 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/mobilesam &

# 设置第三组代码运行在 GPU 2 上
export CUDA_VISIBLE_DEVICES=2
echo "Running modified MobileSAM with label_num 5 on GPU 2"
python /data2/wuxinrui/RA-L/MobileSAM/eval.py --checkpoint_path /data2/wuxinrui/RA-L/MobileSAM/trained_models/wxr_modified_mobilesam/random_point_v2_bbox.pth --ori_SAM False --label_num 5 --visualize_mask_path /data2/wuxinrui/RA-L/MobileSAM/baseline_eval/temp &

