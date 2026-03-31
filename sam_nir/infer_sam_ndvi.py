import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
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


def calculate_ndvi(nir, red, eps=1e-8):
    return (nir - red) / (nir + red + eps)


def save_ndvi_colormap(ndvi_tensor, save_path, title="NDVI"):
    ndvi_np = ndvi_tensor.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(ndvi_np, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    sam_ckpt = os.path.join(ROOT_DIR, "checkpoints", "sam_vit_b_01ec64.pth")
    model_ckpt = os.path.join(ROOT_DIR, "sam_nir", "checkpoints", "sam_nir_epoch_002.pth")

    out_dir = os.path.join(ROOT_DIR, "sam_nir", "ndvi_predictions")
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

            ndvi_pred = calculate_ndvi(pred, r)
            ndvi_true = calculate_ndvi(y, r)

            # Per visualizzazione immagini grayscale su save_image:
            ndvi_pred_vis = ((ndvi_pred + 1.0) / 2.0).clamp(0, 1)
            ndvi_true_vis = ((ndvi_true + 1.0) / 2.0).clamp(0, 1)

            # salva confronto: R | G | NIR pred | NIR true | NDVI pred | NDVI true
            grid = torch.cat([r, g, pred, y, ndvi_pred_vis, ndvi_true_vis], dim=3)
            grid_path = os.path.join(out_dir, f"{i:03d}_{name}_sam_ndvi_grid.png")
            save_image(grid, grid_path)

            # salva colormap NDVI
            pred_color_path = os.path.join(out_dir, f"{i:03d}_{name}_ndvi_pred.png")
            true_color_path = os.path.join(out_dir, f"{i:03d}_{name}_ndvi_true.png")

            save_ndvi_colormap(ndvi_pred, pred_color_path, title=f"Pred NDVI - {name}")
            save_ndvi_colormap(ndvi_true, true_color_path, title=f"True NDVI - {name}")

            # statistiche semplici
            stats_path = os.path.join(out_dir, f"{i:03d}_{name}_ndvi_stats.txt")
            with open(stats_path, "w") as f:
                f.write(f"image_name: {name}\n")
                f.write(f"pred_ndvi_min: {ndvi_pred.min().item():.6f}\n")
                f.write(f"pred_ndvi_max: {ndvi_pred.max().item():.6f}\n")
                f.write(f"pred_ndvi_mean: {ndvi_pred.mean().item():.6f}\n")
                f.write(f"true_ndvi_min: {ndvi_true.min().item():.6f}\n")
                f.write(f"true_ndvi_max: {ndvi_true.max().item():.6f}\n")
                f.write(f"true_ndvi_mean: {ndvi_true.mean().item():.6f}\n")

            print("saved:", grid_path)
            print("saved:", pred_color_path)
            print("saved:", true_color_path)
            print("saved:", stats_path)

            if i >= 9:
                break


if __name__ == "__main__":
    main()