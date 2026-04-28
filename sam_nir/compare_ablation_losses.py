import os
import sys
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sam_nir.sam_encoder_model import SAMNIRModel
from sam_nir.dataset_sam_nir import SAMNIRDataset


VAL_ROOT = "/Users/lindawandjilando/datasets/rgb2nir_dataset_real/splits/val"
SAM_PRETRAINED_CKPT = os.path.join(ROOT_DIR, "checkpoints", "sam_vit_b_01ec64.pth")

VARIANTS = [
    {
        "name": "MSE_L1_Edge",
        "loss": "MSE + L1 + Edge",
        "ckpt": os.path.join(ROOT_DIR, "sam_nir", "checkpoints_r8_mse_l1_edge_ablation", "sam_nir_epoch_002.pth"),
        "lora_rank": 8,
    },
    {
        "name": "MSE_L1_Edge_NDVI",
        "loss": "MSE + L1 + Edge + NDVI",
        "ckpt": os.path.join(ROOT_DIR, "sam_nir", "checkpoints_r8_mse_l1_edge_ndvi_ablation", "sam_nir_epoch_002.pth"),
        "lora_rank": 8,
    },
    {
        "name": "MSE_L1_Edge_NDVI_Grad",
        "loss": "MSE + L1 + Edge + NDVI + Grad",
        "ckpt": os.path.join(ROOT_DIR, "sam_nir", "checkpoints_r8_mse_l1_edge_ndvi_grad", "sam_nir_epoch_002.pth"),
        "lora_rank": 8,
    },
]


BASELINE = {
    "psnr_ndvi": 23.335960906055156,
    "ssim_ndvi": 0.6674589870466829,
    "roi_ssim_ndvi": 0.6881595913907548,
}

SAM_MSE_L1 = {
    "psnr_ndvi": 24.958065792626797,
    "ssim_ndvi": 0.6624185214823676,
    "roi_ssim_ndvi": 0.670980846564244,
}


def build_val_dataset():
    return SAMNIRDataset(
        root_dir=VAL_ROOT,
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


def psnr_np(pred, target, data_range):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def ssim_np(pred, target, data_range):
    h, w = pred.shape
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return None
    return ssim(pred, target, data_range=data_range, win_size=win_size)


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def masked_ssim_bbox(pred, target, mask, data_range):
    box = bbox_from_mask(mask)
    if box is None:
        return None

    y0, y1, x0, x1 = box
    pred_crop = pred[y0:y1, x0:x1].copy()
    target_crop = target[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1]

    if pred_crop.shape[0] < 7 or pred_crop.shape[1] < 7:
        return None

    pred_crop[~mask_crop] = 0.0
    target_crop[~mask_crop] = 0.0

    return ssim_np(pred_crop, target_crop, data_range=data_range)


def mean_valid(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def evaluate_variant(variant, dataset, device):
    print("Evaluating:", variant["name"])
    print("Checkpoint:", variant["ckpt"])

    model = SAMNIRModel(
        sam_ckpt_path=SAM_PRETRAINED_CKPT,
        lora_rank=variant["lora_rank"],
        freeze_encoder=True,
    ).to(device)

    ckpt = torch.load(variant["ckpt"], map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    psnr_ndvi_list = []
    ssim_ndvi_list = []
    roi_ssim_ndvi_list = []

    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["image_sam"].to(device).float()
            y = batch["image_nir"].to(device).float().clamp(0, 1)
            r = batch["image_r"].to(device).float().clamp(0, 1)

            pred = model(x).clamp(0, 1)

            pred_np = pred.squeeze().cpu().numpy()
            y_np = y.squeeze().cpu().numpy()
            r_np = r.squeeze().cpu().numpy()

            ndvi_pred = calculate_ndvi(pred_np, r_np)
            ndvi_true = calculate_ndvi(y_np, r_np)

            psnr_ndvi_list.append(psnr_np(ndvi_pred, ndvi_true, data_range=2.0))

            ssim_ndvi_val = ssim_np(ndvi_pred, ndvi_true, data_range=2.0)
            if ssim_ndvi_val is not None:
                ssim_ndvi_list.append(ssim_ndvi_val)

            roi_mask = ndvi_true > 0.3
            roi_ssim = masked_ssim_bbox(ndvi_pred, ndvi_true, roi_mask, data_range=2.0)
            if roi_ssim is not None:
                roi_ssim_ndvi_list.append(roi_ssim)

    return {
        "loss": variant["loss"],
        "psnr_ndvi": mean_valid(psnr_ndvi_list),
        "ssim_ndvi": mean_valid(ssim_ndvi_list),
        "roi_ssim_ndvi": mean_valid(roi_ssim_ndvi_list),
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    dataset = build_val_dataset()

    results = {
        "Baseline Swin2MoSE": {
            "loss": "-",
            **BASELINE,
        },
        "SAM r8": {
            "loss": "MSE + L1",
            **SAM_MSE_L1,
        },
    }

    for variant in VARIANTS:
        results[variant["name"]] = evaluate_variant(variant, dataset, device)

    out_path = os.path.join(ROOT_DIR, "sam_nir", "comparison_ablation_losses.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved:", out_path)

    print("\nMarkdown table:\n")
    print("| Modello | Loss | PSNR NDVI | SSIM NDVI | ROI SSIM NDVI |")
    print("|---|---|---:|---:|---:|")
    for model_name, r in results.items():
        print(
            f"| {model_name} | {r['loss']} | "
            f"{r['psnr_ndvi']:.3f} | {r['ssim_ndvi']:.3f} | {r['roi_ssim_ndvi']:.3f} |"
        )


if __name__ == "__main__":
    main()
