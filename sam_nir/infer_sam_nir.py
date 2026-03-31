import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sam_nir.sam_encoder_model import SAMNIRModel
from sam_nir.dataset_sam_nir import SAMNIRDataset


def build_val_dataset():
    return SAMNIRDataset(
        root_dir="/Users/lindawandjilando/datasets/rgb2nir_dataset_real/splits/val",
        img_size=128,
        rmean=0.0,
        rstd=1.0,
        gmean=0.0,
        gstd=1.0,
        nirmean=0.0,
        nirstd=1.0,
        rmin=0.0,
        rmax=65535.0,
        gmin=0.0,
        gmax=65535.0,
        nirmin=0.0,
        nirmax=65535.0,
        train=1,
        verbose=True,
    )


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    sam_ckpt = os.path.join(ROOT_DIR, "checkpoints", "sam_vit_b_01ec64.pth")
    model_ckpt = os.path.join(ROOT_DIR, "sam_nir", "checkpoints", "sam_nir_epoch_002.pth")
    out_dir = os.path.join(ROOT_DIR, "sam_nir", "predictions")
    os.makedirs(out_dir, exist_ok=True)

    dataset = build_val_dataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = SAMNIRModel(
        sam_ckpt_path=sam_ckpt,
        lora_rank=4,
        freeze_encoder=True,
    ).to(device)

    ckpt = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["image_sam"].to(device).float()
            y = batch["image_nir"].to(device).float()
            r = batch["image_r"].to(device).float()
            g = batch["image_g"].to(device).float()
            name = batch["image_name"][0]

            pred = model(x)

            # salvataggio confronto: R | G | pred | gt
            grid = torch.cat([r, g, pred, y], dim=3)
            save_path = os.path.join(out_dir, f"{i:03d}_{name}_sam_nir.png")
            save_image(grid, save_path)

            print("saved:", save_path)

            if i >= 9:
                break


if __name__ == "__main__":
    main()