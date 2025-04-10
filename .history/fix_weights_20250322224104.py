import torch

checkpoint = torch.load('/data2/wuxinrui/RA-L/MobileSAM/trained_models/new_mimc/final_model.pth', map_location="cpu")

new_checkpoint = {f"model.{k}": v for k, v in checkpoint.items()}

torch.save(new_checkpoint, "/data2/wuxinrui/RA-L/MobileSAM/trained_models/new_mimc/final_model.pth")