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


SAM_VARIANT = "r8_mse_l1_edge_ndvi_grad_epoch_002"

SAM_CKPT = os.path.join(
    ROOT_DIR,
    "sam_nir",
    "checkpoints_r8_mse_l1_edge_ndvi_grad",
    "sam_nir_epoch_002.pth",
)

SAM_LORA_RANK = 8

VAL_ROOT = "/Users/lindawandjilando/datasets/rgb2nir_dataset_real/splits/val"

SAM_PRETRAINED_CKPT = os.path.join(
    ROOT_DIR,
    "checkpoints",
    "sam_vit_b_01ec64.pth",
)

OUT_GLOBAL = os.path.join(ROOT_DIR, "sam_nir", "comparison_results_ndvi_grad.json")
OUT_ROI = os.path.join(ROOT_DIR, "sam_nir", "comparison_results_roi_ndvi_grad.json")


BASELINE_GLOBAL = {
    "psnr": 17.875751572686273,
    "ssim": 0.44233360423071827,
    "psnr_ndvi": 23.335960906055156,
    "ssim_ndvi": 0.6674589870466829,
}

BASELINE_ROI = {
    "psnr": 22.132965218692966,
    "ssim": 0.4267936839556718,
    "psnr_ndvi": 24.880587800354977,
    "ssim_ndvi": 0.6881595913907548,
    "valid_ssim_images": 72,
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


def masked_psnr(pred, target, mask, data_range):
    if mask.sum() == 0:
        return None
    return psnr_np(pred[mask], target[mask], data_range=data_range)


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


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    dataset = build_val_dataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = SAMNIRModel(
        sam_ckpt_path=SAM_PRETRAINED_CKPT,
        lora_rank=SAM_LORA_RANK,
        freeze_encoder=True,
    ).to(device)

    ckpt = torch.load(SAM_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    psnr_list = []
    ssim_list = []
    psnr_ndvi_list = []
    ssim_ndvi_list = []

    psnr_roi_list = []
    ssim_roi_list = []
    psnr_ndvi_roi_list = []
    ssim_ndvi_roi_list = []

    valid_ssim_images = 0
    roi_pixel_counts = []
    n_images_with_roi = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["image_sam"].to(device).float()
            y = batch["image_nir"].to(device).float()
            r = batch["image_r"].to(device).float()

            pred = model(x).clamp(0, 1)
            y = y.clamp(0, 1)
            r = r.clamp(0, 1)

            pred_np = pred.squeeze().cpu().numpy()
            y_np = y.squeeze().cpu().numpy()
            r_np = r.squeeze().cpu().numpy()

            ndvi_pred = calculate_ndvi(pred_np, r_np)
            ndvi_true = calculate_ndvi(y_np, r_np)

            psnr_list.append(psnr_np(pred_np, y_np, data_range=1.0))

            ssim_val = ssim_np(pred_np, y_np, data_range=1.0)
            if ssim_val is not None:
                ssim_list.append(ssim_val)

            psnr_ndvi_list.append(psnr_np(ndvi_pred, ndvi_true, data_range=2.0))

            ssim_ndvi_val = ssim_np(ndvi_pred, ndvi_true, data_range=2.0)
            if ssim_ndvi_val is not None:
                ssim_ndvi_list.append(ssim_ndvi_val)

            roi_mask = ndvi_true > 0.3
            roi_pixels = int(roi_mask.sum())
            roi_pixel_counts.append(roi_pixels)

            if roi_pixels > 0:
                n_images_with_roi += 1

                psnr_roi = masked_psnr(pred_np, y_np, roi_mask, data_range=1.0)
                if psnr_roi is not None:
                    psnr_roi_list.append(psnr_roi)

                ssim_roi = masked_ssim_bbox(pred_np, y_np, roi_mask, data_range=1.0)
                if ssim_roi is not None:
                    ssim_roi_list.append(ssim_roi)
                    valid_ssim_images += 1

                psnr_ndvi_roi = masked_psnr(ndvi_pred, ndvi_true, roi_mask, data_range=2.0)
                if psnr_ndvi_roi is not None:
                    psnr_ndvi_roi_list.append(psnr_ndvi_roi)

                ssim_ndvi_roi = masked_ssim_bbox(
                    ndvi_pred,
                    ndvi_true,
                    roi_mask,
                    data_range=2.0,
                )
                if ssim_ndvi_roi is not None:
                    ssim_ndvi_roi_list.append(ssim_ndvi_roi)

    results_global = {
        "sam_variant": SAM_VARIANT,
        "baseline": BASELINE_GLOBAL,
        "sam": {
            "psnr": mean_valid(psnr_list),
            "ssim": mean_valid(ssim_list),
            "psnr_ndvi": mean_valid(psnr_ndvi_list),
            "ssim_ndvi": mean_valid(ssim_ndvi_list),
        },
        "n": len(psnr_list),
    }

    results_roi = {
        "sam_variant": SAM_VARIANT,
        "roi_definition": "ndvi_true > 0.3",
        "baseline_roi": BASELINE_ROI,
        "sam_roi": {
            "psnr": mean_valid(psnr_roi_list),
            "ssim": mean_valid(ssim_roi_list),
            "psnr_ndvi": mean_valid(psnr_ndvi_roi_list),
            "ssim_ndvi": mean_valid(ssim_ndvi_roi_list),
            "valid_ssim_images": valid_ssim_images,
        },
        "n_images": len(psnr_list),
        "n_images_with_roi": n_images_with_roi,
        "mean_roi_pixels": float(np.mean(roi_pixel_counts)),
    }

    with open(OUT_GLOBAL, "w") as f:
        json.dump(results_global, f, indent=2)

    with open(OUT_ROI, "w") as f:
        json.dump(results_roi, f, indent=2)

    print("Saved:", OUT_GLOBAL)
    print("Saved:", OUT_ROI)


if __name__ == "__main__":
    main()