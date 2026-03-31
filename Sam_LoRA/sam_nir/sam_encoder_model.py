import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

SAM_LORA_DIR = os.path.join(os.path.dirname(__file__), "..", "Sam_LoRA")
if SAM_LORA_DIR not in sys.path:
    sys.path.append(SAM_LORA_DIR)

from segment_anything import sam_model_registry
from sam_lora import LoRA_Sam


class SimpleNIRDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)


class SAMNIRModel(nn.Module):
    def __init__(self, sam_ckpt_path, lora_rank=4, freeze_encoder=True):
        super().__init__()

        sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt_path)
        self.sam_lora = LoRA_Sam(sam, lora_rank)
        self.encoder = self.sam_lora.sam.image_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = SimpleNIRDecoder(in_channels=256, out_channels=1)

    def forward(self, x):
        feats = self.encoder(x)                  # [B,256,64,64]
        nir = self.decoder(feats)               # [B,1,64,64]
        nir = F.interpolate(nir, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return nir