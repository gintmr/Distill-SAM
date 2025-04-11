import os
import torch
from mobile_sam import sam_model_registry
os.environ['INFERENCE_MODE'] = "train"

model = sam_model_registry["tiny_msam"]

torch.save(model.state_dict(), "init_weights.pth")