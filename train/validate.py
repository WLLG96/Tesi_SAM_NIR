# swin2nir/train/validate.py
import os
import csv
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torchmetrics.image import PeakSignalNoiseRatio
from skimage.metrics import structural_similarity as sk_ssim
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


def _minmax(x, vmin, vmax, eps=1e-8):
    return (x - vmin) / (vmax - vmin + eps)


def _unpack(sample, device):
    r = sample["image_r"].to(device, non_blocking=True).float()
    g = sample["image_g"].to(device, non_blocking=True).float()
    nir = sample["image_nir"].to(device, non_blocking=True).float()

    # garantiamo 4D
    if r.dim() == 3:
        r = r.unsqueeze(0)
    if g.dim() == 3:
        g = g.unsqueeze(0)
    if nir.dim() == 3:
        nir = nir.unsqueeze(0)

    x = torch.cat([r, g], dim=1)  # [B,2,H,W]
    y = nir                       # [B,1,H,W]
    name = sample.get("image_name", None)
    return x, y, r, g, name


def _align_pred(pred, y):
    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    # canali
    if pred.dim() == 4 and pred.size(1) > 1 and y.size(1) == 1:
        pred = pred[:, :1, :, :]

    # size
    if pred.shape[-2:] != y.shape[-2:]:
        pred = F.interpolate(pred, size=y.shape[-2:], mode="bilinear", align_corners=False)

    return pred


def calculate_ndvi(nir, red, eps=1e-8):
    return (nir - red) / (nir + red + eps)


@torch.no_grad()
def validate(generator, dataloader, criterion, device, config, epoch):
    """
    Validation/Test:
      - generator(x) dove x = cat([R,G]) -> [B,2,H,W]
      - salva metriche su file (txt/json/csv) in paths.results_dir
      - salva immagini qualitative in:
          - paths.validation_results_dir (se val)
          - paths.test_results_dir (se test)
      - supporta:
          config["val"]["max_batches"] = None per FULL, oppure int per LIGHT
          config["val"]["save_every"]  = salva immagini ogni N batch (0 disabilita)
          config["val"]["log_every"]   = aggiorna tqdm ogni N batch (0 disabilita)
    """

    generator.eval()

    norm = config.get("norm", {})
    paths = config.get("paths", {})
    val_cfg = config.get("val", {})

    rmin = float(norm.get("min_r", 0.0))
    rmax = float(norm.get("max_r", 1.0))
    gmin = float(norm.get("min_g", 0.0))
    gmax = float(norm.get("max_g", 1.0))
    nmin = float(norm.get("min_n", 0.0))
    nmax = float(norm.get("max_n", 1.0))

    # capiamo se stiamo facendo TEST o VAL usando root_dir del dataset
    root_dir = getattr(dataloader.dataset, "root_dir", "")
    mode = "test" if "test" in str(root_dir).lower() else "val"

    # output dirs
    results_dir = paths.get("results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)

    if mode == "test":
        out_dir = paths.get("test_results_dir", "./test_results")
    else:
        out_dir = paths.get("validation_results_dir", "./validation_results")
    os.makedirs(out_dir, exist_ok=True)

    log_dir = paths.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # val controls
    max_batches = val_cfg.get("max_batches", None)  # None => full
    save_every = int(val_cfg.get("save_every", 0))  # 0 => no images
    log_every = int(val_cfg.get("log_every", 10))   # 0 => no tqdm postfix

    # metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_psnr_ndvi = 0.0
    total_ssim_ndvi = 0.0
    n_imgs = 0

    pbar = tqdm(dataloader, desc=f"Validation (epoch {epoch})", leave=False)

    for b, sample in enumerate(pbar, start=1):
        if max_batches is not None and b > int(max_batches):
            break

        x, y, red, green, name = _unpack(sample, device)

        pred = generator(x)
        pred = _align_pred(pred, y)

        loss = criterion(pred, y)
        total_loss += float(loss.item())

        # min-max -> [0,1]
        red01 = _minmax(red, rmin, rmax).clamp(0, 1)
        green01 = _minmax(green, gmin, gmax).clamp(0, 1)
        pred01 = _minmax(pred, nmin, nmax).clamp(0, 1)
        y01 = _minmax(y, nmin, nmax).clamp(0, 1)

        # NDVI
        ndvi_fake = calculate_ndvi(pred01, red01)
        ndvi_true = calculate_ndvi(y01, red01)

        # metriche per immagine
        B = pred01.size(0)
        for j in range(B):
            total_psnr += float(psnr_metric(pred01[j], y01[j]).item())

            a = pred01[j, 0].detach().cpu().numpy()
            t = y01[j, 0].detach().cpu().numpy()
            total_ssim += float(sk_ssim(a, t, data_range=1.0))

            total_psnr_ndvi += float(psnr_metric(ndvi_fake[j], ndvi_true[j]).item())
            a2 = ndvi_fake[j, 0].detach().cpu().numpy()
            t2 = ndvi_true[j, 0].detach().cpu().numpy()
            total_ssim_ndvi += float(sk_ssim(a2, t2, data_range=2.0))  # NDVI ~[-1,1]

            n_imgs += 1

        # salva immagini qualitative ogni N batch
        if save_every > 0 and (b % save_every == 0 or b == 1):
            ndvi_fake_vis = ((ndvi_fake + 1.0) / 2.0).clamp(0, 1)
            ndvi_true_vis = ((ndvi_true + 1.0) / 2.0).clamp(0, 1)

            # concat width: R | G | pred | gt | ndvi_fake | ndvi_true
            grid = torch.cat([red01, green01, pred01, y01, ndvi_fake_vis, ndvi_true_vis], dim=3)
            fname = f"{mode}_epoch_{epoch:03d}_batch_{b:04d}.png"
            save_image(grid, os.path.join(out_dir, fname))

        if log_every and (b % log_every == 0 or b == 1):
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    # medie
    denom_batches = max(1, min(len(dataloader), int(max_batches)) if max_batches is not None else len(dataloader))
    avg_loss = total_loss / denom_batches

    denom_imgs = max(1, n_imgs)
    avg_psnr = total_psnr / denom_imgs
    avg_ssim = total_ssim / denom_imgs
    avg_psnr_ndvi = total_psnr_ndvi / denom_imgs
    avg_ssim_ndvi = total_ssim_ndvi / denom_imgs

    # tensorboard
    writer.add_scalar(f"{mode}/Loss", avg_loss, epoch)
    writer.add_scalar(f"{mode}/PSNR_NIR", avg_psnr, epoch)
    writer.add_scalar(f"{mode}/SSIM_NIR", avg_ssim, epoch)
    writer.add_scalar(f"{mode}/PSNR_NDVI", avg_psnr_ndvi, epoch)
    writer.add_scalar(f"{mode}/SSIM_NDVI", avg_ssim_ndvi, epoch)
    writer.close()

    # log finale (LIGHT vs FULL)
    if max_batches is None:
        print(
            f"[VAL-FULL] epoch={epoch} batches={denom_batches}/{len(dataloader)} "
            f"loss={avg_loss:.6f} PSNR={avg_psnr:.2f} SSIM={avg_ssim:.3f} "
            f"PSNR_NDVI={avg_psnr_ndvi:.2f} SSIM_NDVI={avg_ssim_ndvi:.3f}"
        )
    else:
        print(
            f"[VAL-LIGHT] epoch={epoch} batches={denom_batches} "
            f"loss={avg_loss:.6f} PSNR={avg_psnr:.2f} SSIM={avg_ssim:.3f} "
            f"PSNR_NDVI={avg_psnr_ndvi:.2f} SSIM_NDVI={avg_ssim_ndvi:.3f}"
        )

    # -------------------------
    # SAVE METRICS FOR THESIS
    # -------------------------
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "epoch": int(epoch),
        "num_batches": int(denom_batches),
        "num_images": int(denom_imgs),
        "loss_mse": float(avg_loss),
        "psnr_nir": float(avg_psnr),
        "ssim_nir": float(avg_ssim),
        "psnr_ndvi": float(avg_psnr_ndvi),
        "ssim_ndvi": float(avg_ssim_ndvi),
    }

    txt_path = os.path.join(results_dir, f"{mode}_metrics_epoch_{epoch:03d}.txt")
    with open(txt_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    json_path = os.path.join(results_dir, f"{mode}_metrics_epoch_{epoch:03d}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(results_dir, f"{mode}_metrics.csv")
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not csv_exists:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"[SAVE] metrics -> {txt_path}")
    print(f"[SAVE] metrics -> {csv_path}")

    # cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    return metrics
