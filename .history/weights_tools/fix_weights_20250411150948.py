import torch

# checkpoint = torch.load('/data2/wuxinrui/RA-L/MobileSAM/trained_models/new_mimc/last.ckpt', map_location="cuda")

# #G normal-checkpoint:
# normal_checkpoint = torch.load("/data2/wuxinrui/RA-L/MobileSAM/weights/mobile_sam.pt")


# new_checkpoint = {k.replace("model.", ""): v for k, v in checkpoint.items()}
# # new_checkpoint = {f"model.{k}": v for k, v in checkpoint.items()}

# torch.save(new_checkpoint, "/data2/wuxinrui/RA-L/MobileSAM/trained_models/new_mimc/last.ckpt")


checkpoint = torch.load("/data2/wuxinrui/RA-L/MobileSAM/trained_models/Distilled_encoder/last-v5.ckpt", map_location="cuda")
# print(checkpoint.keys())
state_dict = checkpoint['state_dict']
save_checkpoint = {}
for k, v in state_dict.items():
    if "S_model." in k and "image_encoder" in k:
        save_checkpoint[k.replace("S_model.", "")] = v
    elif "T_model." in k:
        if "prompt_encoder" or "mask_decoder" in k:
            save_checkpoint[k.replace("T_model.", "")] = v

# torch.save(save_checkpoint, "/data2/wuxinrui/RA-L/MobileSAM/trained_models/Distilled_encoder/Distilled.pth")