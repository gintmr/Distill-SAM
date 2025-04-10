import torch

checkpoint = torch.load('/data2/wuxinrui/RA-L/MobileSAM/trained_models/new_mimc/last.ckpt', map_location="cuda")

new_checkpoint = {k.replace("model.", ""): v for k, v in checkpoint.items()}
# new_checkpoint = {f"model.{k}": v for k, v in checkpoint.items()}

torch.save(new_checkpoint, "/data2/wuxinrui/RA-L/MobileSAM/trained_models/new_mimc/last.ckpt")