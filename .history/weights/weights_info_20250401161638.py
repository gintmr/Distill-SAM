import torch

def print_model_info(state_dict):
    """
    打印模型的参数量和内存占用信息。
    :param state_dict: 模型的 state_dict，包含权重参数。
    """
    module_params = {}  # 用于存储每个模块的参数量
    module_memory = {}  # 用于存储每个模块的内存占用

    # 遍历 state_dict 中的每个参数
    for key, param in state_dict.items():
        # 获取模块名称（假设键名以模块名开头）
        module_name = key.split('.')[0]

        # 计算参数量
        num_elements = param.numel()

        # 计算内存占用（字节）
        if param.dtype.is_floating_point:
            bytes_per_element = torch.finfo(param.dtype).bits // 8
        elif param.dtype.is_signed:
            bytes_per_element = torch.iinfo(param.dtype).bits // 8
        else:
            bytes_per_element = torch.iinfo(param.dtype).bits // 8

        memory_size_bytes = num_elements * bytes_per_element

        # 累加到对应模块
        if module_name not in module_params:
            module_params[module_name] = 0
            module_memory[module_name] = 0
        module_params[module_name] += num_elements
        module_memory[module_name] += memory_size_bytes

    # 打印每个模块的参数量和内存占用
    print("模块参数量和内存占用：")
    for module_name in module_params:
        param_count = module_params[module_name]
        memory_mb = module_memory[module_name] / (1024 * 1024)
        print(f"模块 {module_name}:")
        print(f"  参数量: {param_count}")
        print(f"  内存占用: {memory_mb:.2f} MB")

    # 计算总参数量和总内存占用
    total_params = sum(module_params.values())
    total_memory_bytes = sum(module_memory.values())
    total_memory_mb = total_memory_bytes / (1024 * 1024)

    # 打印总参数量和总内存占用
    print("\n总参数量和内存占用：")
    print(f"总参数量: {total_params}")
    print(f"总内存占用: {total_memory_mb:.2f} MB")


# 加载模型的 state_dict
state_dict_path = '/data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt'  # 替换为你的 .pt 文件路径
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))

# 打印模型信息
print_model_info(state_dict)