import os
import json
import math
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as sk_ssim

from utils import load_swin2_mose, load_config
from data.dataset_cropped import NIRDataset_cropped as NIRDataset
from sam_nir.sam_encoder_model import SAMNIRModel


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config_linda.yaml")
BASELINE_CKPT_PATH = os.path.join(ROOT_DIR, "checkpoint_model", "ckpt_epoch_002.pth")
SAM_CKPT_PATH = os.path.join(ROOT_DIR, "sam_nir", "checkpoints", "sam_nir_epoch_002.pth")
SAM_ENCODER_CKPT = os.path.join(ROOT_DIR, "checkpoints", "sam_vit_b_01ec64.pth")

ROI_NDVI_THRESHOLD = 0.30
MIN_ROI_PIXELS_FOR_SSIM = 25
MIN_BBOX_SIZE_FOR_SSIM = 7


def calculate_ndvi(nir, red, eps=1e-8):
    return (nir - red) / (nir + red + eps)


def psnr_from_mse(err, data_range=1.0, eps=1e-12):
    if err < eps:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / err)


def _align_pred_to_target(pred, y):
    """
    Allinea output del modello al target:
    - se tuple/list -> prende pred[0]
    - se canali > 1 e target è 1 canale -> prende il primo canale
    - se size diversa -> interpolazione
    """
    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    if pred.dim() != 4:
        raise ValueError(
            f"pred deve essere 4D [B,C,H,W], trovato shape={tuple(pred.shape)}"
        )

    if pred.size(1) != y.size(1):
        if y.size(1) == 1 and pred.size(1) > 1:
            pred = pred[:, :1, :, :]
        else:
            raise ValueError(
                f"Mismatch canali: pred={pred.size(1)}, target={y.size(1)}"
            )

    if pred.shape[-2:] != y.shape[-2:]:
        pred = F.interpolate(pred, size=y.shape[-2:], mode="bilinear", align_corners=False)

    return pred


def masked_mse(pred, target, mask):
    pred_vals = pred[mask]
    target_vals = target[mask]
    if pred_vals.numel() == 0:
        return None
    return torch.mean((pred_vals - target_vals) ** 2).item()


def get_bbox_from_mask(mask_2d):
    ys, xs = np.where(mask_2d)
    if len(ys) == 0 or len(xs) == 0:
        return None
    r0, r1 = ys.min(), ys.max() + 1
    c0, c1 = xs.min(), xs.max() + 1
    return r0, r1, c0, c1


def masked_bbox_ssim(pred_2d, target_2d, mask_2d, data_range):
    if mask_2d.sum() < MIN_ROI_PIXELS_FOR_SSIM:
        return None

    bbox = get_bbox_from_mask(mask_2d)
    if bbox is None:
        return None

    r0, r1, c0, c1 = bbox
    h = r1 - r0
    w = c1 - c0
    if h < MIN_BBOX_SIZE_FOR_SSIM or w < MIN_BBOX_SIZE_FOR_SSIM:
        return None

    pred_crop = pred_2d[r0:r1, c0:c1]
    target_crop = target_2d[r0:r1, c0:c1]

    return float(sk_ssim(pred_crop, target_crop, data_range=data_range))


def load_baseline_model(cfg, device):
    print("Loading baseline model...")
    model = load_swin2_mose(cfg)
    ckpt = torch.load(BASELINE_CKPT_PATH, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_sam_model(device):
    print("Loading SAM model...")
    model = SAMNIRModel(
        sam_ckpt_path=SAM_ENCODER_CKPT,
        lora_rank=4,
        freeze_encoder=True,
    ).to(device)

    ckpt = torch.load(SAM_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(CONFIG_PATH)

    print("Loading dataset...")
    dataset = NIRDataset(
        root_dir=cfg["dataset"]["val_data_root"],
        img_size=cfg["dataset"]["img_size"],
        rmean=cfg["norm"]["mean_red"],
        rstd=cfg["norm"]["std_red"],
        gmean=cfg["norm"]["mean_green"],
        gstd=cfg["norm"]["std_green"],
        nirmean=cfg["norm"]["mean_nir"],
        nirstd=cfg["norm"]["std_nir"],
        rmin=cfg["norm"]["min_r"],
        rmax=cfg["norm"]["max_r"],
        gmin=cfg["norm"]["min_g"],
        gmax=cfg["norm"]["max_g"],
        nirmin=cfg["norm"]["min_n"],
        nirmax=cfg["norm"]["max_n"],
        train=1,
        verbose=True,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    baseline_model = load_baseline_model(cfg, device)
    sam_model = load_sam_model(device)

    results = {
        "roi_definition": f"ndvi_true > {ROI_NDVI_THRESHOLD}",
        "baseline_roi": {
            "psnr": 0.0,
            "ssim": 0.0,
            "psnr_ndvi": 0.0,
            "ssim_ndvi": 0.0,
            "valid_ssim_images": 0,
        },
        "sam_roi": {
            "psnr": 0.0,
            "ssim": 0.0,
            "psnr_ndvi": 0.0,
            "ssim_ndvi": 0.0,
            "valid_ssim_images": 0,
        },
        "n_images": 0,
        "n_images_with_roi": 0,
        "mean_roi_pixels": 0.0,
    }

    total_roi_pixels = 0

    for sample in tqdm(loader, desc="Comparing ROI"):
        r = sample["image_r"].to(device).float()
        g = sample["image_g"].to(device).float()
        y = sample["image_nir"].to(device).float()

        # Baseline
        x_base = torch.cat([r, g], dim=1)
        pred_base = baseline_model(x_base)
        pred_base = _align_pred_to_target(pred_base, y)

        # SAM
        x_sam = torch.cat([r, g, g], dim=1)
        pred_sam = sam_model(x_sam)
        pred_sam = _align_pred_to_target(pred_sam, y)

        pred_base = pred_base.clamp(0, 1)
        pred_sam = pred_sam.clamp(0, 1)
        y = y.clamp(0, 1)
        r = r.clamp(0, 1)

        ndvi_true = calculate_ndvi(y, r)
        ndvi_base = calculate_ndvi(pred_base, r)
        ndvi_sam = calculate_ndvi(pred_sam, r)

        roi_mask = (ndvi_true > ROI_NDVI_THRESHOLD)
        roi_pixels = int(roi_mask.sum().item())

        results["n_images"] += 1

        if roi_pixels == 0:
            continue

        results["n_images_with_roi"] += 1
        total_roi_pixels += roi_pixels

        # PSNR su ROI - NIR
        mse_base_roi = masked_mse(pred_base, y, roi_mask)
        mse_sam_roi = masked_mse(pred_sam, y, roi_mask)

        if mse_base_roi is not None:
            results["baseline_roi"]["psnr"] += psnr_from_mse(mse_base_roi, data_range=1.0)
        if mse_sam_roi is not None:
            results["sam_roi"]["psnr"] += psnr_from_mse(mse_sam_roi, data_range=1.0)

        # PSNR su ROI - NDVI
        mse_base_ndvi_roi = masked_mse(ndvi_base, ndvi_true, roi_mask)
        mse_sam_ndvi_roi = masked_mse(ndvi_sam, ndvi_true, roi_mask)

        if mse_base_ndvi_roi is not None:
            results["baseline_roi"]["psnr_ndvi"] += psnr_from_mse(mse_base_ndvi_roi, data_range=2.0)
        if mse_sam_ndvi_roi is not None:
            results["sam_roi"]["psnr_ndvi"] += psnr_from_mse(mse_sam_ndvi_roi, data_range=2.0)

        # SSIM su ROI via bounding box
        roi_mask_np = roi_mask[0, 0].cpu().numpy().astype(bool)

        y_np = y[0, 0].cpu().numpy()
        pred_base_np = pred_base[0, 0].cpu().numpy()
        pred_sam_np = pred_sam[0, 0].cpu().numpy()

        ndvi_true_np = ndvi_true[0, 0].cpu().numpy()
        ndvi_base_np = ndvi_base[0, 0].cpu().numpy()
        ndvi_sam_np = ndvi_sam[0, 0].cpu().numpy()

        ssim_base_roi = masked_bbox_ssim(pred_base_np, y_np, roi_mask_np, data_range=1.0)
        ssim_sam_roi = masked_bbox_ssim(pred_sam_np, y_np, roi_mask_np, data_range=1.0)

        ssim_base_ndvi_roi = masked_bbox_ssim(ndvi_base_np, ndvi_true_np, roi_mask_np, data_range=2.0)
        ssim_sam_ndvi_roi = masked_bbox_ssim(ndvi_sam_np, ndvi_true_np, roi_mask_np, data_range=2.0)

        if ssim_base_roi is not None:
            results["baseline_roi"]["ssim"] += ssim_base_roi
        if ssim_base_ndvi_roi is not None:
            results["baseline_roi"]["ssim_ndvi"] += ssim_base_ndvi_roi
            results["baseline_roi"]["valid_ssim_images"] += 1

        if ssim_sam_roi is not None:
            results["sam_roi"]["ssim"] += ssim_sam_roi
        if ssim_sam_ndvi_roi is not None:
            results["sam_roi"]["ssim_ndvi"] += ssim_sam_ndvi_roi
            results["sam_roi"]["valid_ssim_images"] += 1

    n_roi = results["n_images_with_roi"]

    if n_roi > 0:
        results["baseline_roi"]["psnr"] /= n_roi
        results["baseline_roi"]["psnr_ndvi"] /= n_roi
        results["sam_roi"]["psnr"] /= n_roi
        results["sam_roi"]["psnr_ndvi"] /= n_roi
        results["mean_roi_pixels"] = total_roi_pixels / n_roi

    if results["baseline_roi"]["valid_ssim_images"] > 0:
        results["baseline_roi"]["ssim"] /= results["baseline_roi"]["valid_ssim_images"]
        results["baseline_roi"]["ssim_ndvi"] /= results["baseline_roi"]["valid_ssim_images"]

    if results["sam_roi"]["valid_ssim_images"] > 0:
        results["sam_roi"]["ssim"] /= results["sam_roi"]["valid_ssim_images"]
        results["sam_roi"]["ssim_ndvi"] /= results["sam_roi"]["valid_ssim_images"]

    print("\n===== ROI RESULTS =====")
    print(json.dumps(results, indent=2))

    save_path = os.path.join(ROOT_DIR, "sam_nir", "comparison_results_roi.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved ROI results to: {save_path}")


if __name__ == "__main__":
    main()