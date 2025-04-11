import torch

pruned_state = torch.load("/data2/wuxinrui/RA-L/MobileSAM/init_weights.pth")
complex_state_dict = torch.load("/data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt")

def load_matched_weights(pruned_state, complex_state_dict):
    """智能权重加载函数"""
    matched_keys = []
    
    # 第一轮：精确匹配
    for key in pruned_state:
        if key in complex_state_dict: 
            if pruned_state[key].shape == complex_state_dict[key].shape: # 形状匹配
                pruned_state[key] = complex_state_dict[key] # 直接赋值
                matched_keys.append(key)
    
    # 第二轮：维度兼容匹配（处理embed_dim变化）
    for key in pruned_state:
        if key not in matched_keys and "weight" in key:
            if key in complex_state_dict:
                # 卷积层权重适配
                if len(pruned_state[key].shape) == 4: # Conv2d权重
                    min_cin = min(pruned_state[key].shape[1], complex_state_dict[key].shape[1])
                    min_cout = min(pruned_state[key].shape[0], complex_state_dict[key].shape[0])
                    pruned_state[key][:min_cout, :min_cin] = complex_state_dict[key][:min_cout, :min_cin]
                    matched_keys.append(key)
                # 线性层权重适配        
                elif len(pruned_state[key].shape) == 2: # Linear层
                    min_dim2 = min(pruned_state[key].shape[1], complex_state_dict[key].shape[1])
                    min_dim1 = min(pruned_state[key].shape[0], complex_state_dict[key].shape[0])
                    pruned_state[key][:min_dim1, :min_dim2] = complex_state_dict[key][:min_dim1, :min_dim2]
                    matched_keys.append(key)

    print("Matched keys:", matched_keys)
    torch.save(pruned_state, "/data2/wuxinrui/RA-L/MobileSAM/weights/weights_prune_init")
    
load_matched_weights(pruned_state, complex_state_dict)
