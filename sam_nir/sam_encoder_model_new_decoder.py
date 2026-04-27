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


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class DetailEncoder(nn.Module):
    """
    Branch shallow che prende l'input RGB-like [R,G,G]
    e produce feature locali ad alta risoluzione.
    """
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, base_ch),
            ResidualBlock(base_ch),
            ConvBNReLU(base_ch, base_ch),
            ResidualBlock(base_ch),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleNIRDecoder(nn.Module):
    """
    Decoder più robusto:
    - raffina feature SAM 64x64
    - fusiona dettagli dall'input originale
    - evita il giro 64->1024->128
    - produce output direttamente alla size finale
    """
    def __init__(self, in_ch=256, detail_ch=32, out_ch=1):
        super().__init__()

        self.sam_stem = nn.Sequential(
            ConvBNReLU(in_ch, 256),
            ResidualBlock(256),
            ConvBNReLU(256, 128),
            ResidualBlock(128),
            ConvBNReLU(128, 64),
            ResidualBlock(64),
        )

        self.detail_encoder = DetailEncoder(in_ch=3, base_ch=detail_ch)

        self.fuse = nn.Sequential(
            ConvBNReLU(64 + detail_ch, 64),
            ResidualBlock(64),
            ConvBNReLU(64, 32),
            ResidualBlock(32),
        )

        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, sam_feats, x_input, output_size):
        # feature SAM: [B,256,64,64]
        x_sam = self.sam_stem(sam_feats)  # [B,64,64,64]

        # portiamo SAM features direttamente alla size finale
        x_sam = F.interpolate(
            x_sam,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )  # [B,64,H,W]

        # dettagli locali dall'input originale
        x_detail = self.detail_encoder(x_input)  # [B,detail_ch,H,W]

        # fusione
        x = torch.cat([x_sam, x_detail], dim=1)
        x = self.fuse(x)
        out = self.head(x)

        return out


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

        self.decoder = MultiScaleNIRDecoder(
            in_ch=256,
            detail_ch=32,
            out_ch=1,
        )

    def forward(self, x):
        """
        x: [B,3,H,W] = [R,G,G]
        output: [B,1,H,W] in [0,1]
        """
        original_size = x.shape[-2:]

        # SAM encoder vuole 1024x1024
        x_resized = F.interpolate(
            x,
            size=(self.sam_input_size, self.sam_input_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_feats = self.encoder(x_resized)  # [B,256,64,64]

        nir = self.decoder(
            sam_feats=sam_feats,
            x_input=x,
            output_size=original_size,
        )

        return nir