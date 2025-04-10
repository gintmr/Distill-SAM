import torch

checkpoint = torch.load('/data2/wuxinrui/RA-L/MobileSAM/trained_models/mobile/erased_vit_b_points.pth', map_location="cuda")

# new_checkpoint = {f"model.{k}": v for k, v in checkpoint.items()}
new_checkpoint = checkpoint
if "pytorch-lightning_version" not in new_checkpoint:
    new_checkpoint["pytorch-lightning_version"] = "1.9.0"
# merge_checkpoint = {**checkpoint, **new_checkpoint}
torch.save(new_checkpoint, "/data2/wuxinrui/RA-L/MobileSAM/weights/fixed_mobile_sam.pt")