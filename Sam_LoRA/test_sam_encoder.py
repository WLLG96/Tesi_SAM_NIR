import os
import torch
from segment_anything import sam_model_registry
from sam_lora import LoRA_Sam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "..", "checkpoints", "sam_vit_b_01ec64.pth")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)
print("checkpoint:", CKPT_PATH)

sam = sam_model_registry["vit_b"](checkpoint=CKPT_PATH)
lora_sam = LoRA_Sam(sam, 4).to(device)

x = torch.randn(1, 3, 1024, 1024, device=device)

with torch.no_grad():
    feats = lora_sam.sam.image_encoder(x)

print("feature type:", type(feats))
if isinstance(feats, (list, tuple)):
    for i, f in enumerate(feats):
        print(f"feat[{i}] shape:", f.shape)
else:
    print("feature shape:", feats.shape)