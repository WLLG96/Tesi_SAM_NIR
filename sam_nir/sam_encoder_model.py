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
    """
    Decoder vecchio compatibile con i checkpoint esistenti:
    chiavi attese nel checkpoint:
    - decoder.decoder.0.*
    - decoder.decoder.2.*
    - decoder.decoder.4.*
    """

    def __init__(self, in_ch=256, out_ch=1):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, sam_feats, output_size):
        x = F.interpolate(
            sam_feats,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
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

        self.decoder = SimpleNIRDecoder(
            in_ch=256,
            out_ch=1,
        )

    def forward(self, x):
        """
        x: [B,3,H,W] = [R,G,G]
        output: [B,1,H,W] in [0,1]
        """
        original_size = x.shape[-2:]

        x_resized = F.interpolate(
            x,
            size=(self.sam_input_size, self.sam_input_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_feats = self.encoder(x_resized)

        nir = self.decoder(
            sam_feats=sam_feats,
            output_size=original_size,
        )

        return nir