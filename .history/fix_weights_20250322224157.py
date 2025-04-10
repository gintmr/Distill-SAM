import torch
import os
# 检查文件是否存在
checkpoint_path = '/data2/wuxinrui/RA-L/MobileSAM/trained_models/new_mimc/last.ckpt'
assert os.path.exists(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

try:
    # 尝试在 CPU 上加载文件
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Checkpoint loaded successfully.")
except RuntimeError as e:
    print(f"Error loading checkpoint: {e}")
    print("Please check the file integrity and PyTorch version compatibility.")