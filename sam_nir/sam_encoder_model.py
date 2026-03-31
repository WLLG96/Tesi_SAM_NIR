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
    def __init__(
        self,
        sam_ckpt_path,
        lora_rank=4,
        freeze_encoder=True,
        sam_input_size=1024,
    ):
        super().__init__()

        self.sam_input_size = sam_input_size

        sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt_path)
        self.sam_lora = LoRA_Sam(sam, lora_rank)
        self.encoder = self.sam_lora.sam.image_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = SimpleNIRDecoder(in_channels=256, out_channels=1)

    def forward(self, x):
        original_size = x.shape[-2:]  # es. (128, 128)

        # SAM vuole input 1024x1024
        x_resized = F.interpolate(
            x,
            size=(self.sam_input_size, self.sam_input_size),
            mode="bilinear",
            align_corners=False,
        )

        feats = self.encoder(x_resized)   # [B,256,64,64]
        nir = self.decoder(feats)         # [B,1,64,64]

        # prima riportiamo a 1024x1024
        nir = F.interpolate(
            nir,
            size=(self.sam_input_size, self.sam_input_size),
            mode="bilinear",
            align_corners=False,
        )

        # poi torniamo alla size originale del dataset
        nir = F.interpolate(
            nir,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        return nir