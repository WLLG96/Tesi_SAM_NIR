import os
import json
import math
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as sk_ssim
from torchmetrics.image import PeakSignalNoiseRatio

from utils import load_swin2_mose, load_config
from data.dataset_cropped import NIRDataset_cropped as NIRDataset
from sam_nir.sam_encoder_model import SAMNIRModel


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config_linda.yaml")
BASELINE_CKPT_PATH = os.path.join(ROOT_DIR, "checkpoint_model", "ckpt_epoch_002.pth")
SAM_CKPT_PATH = os.path.join(
    ROOT_DIR, "sam_nir", "checkpoints_r8_mse_l1_edge", "sam_nir_epoch_002.pth"
)
SAM_ENCODER_CKPT = os.path.join(ROOT_DIR, "checkpoints", "sam_vit_b_01ec64.pth")


def calculate_ndvi(nir, red, eps=1e-8):
    return (nir - red) / (nir + red + eps)


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
        lora_rank=8,
        freeze_encoder=True,
    ).to(device)

    ckpt = torch.load(SAM_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def psnr_from_mse(err, data_range=1.0, eps=1e-12):
    if err < eps:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / err)


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

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    baseline_model = load_baseline_model(cfg, device)
    sam_model = load_sam_model(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    results = {
        "sam_variant": "r8_mse_l1_edge_epoch_002",
        "baseline": {"psnr": 0.0, "ssim": 0.0, "psnr_ndvi": 0.0, "ssim_ndvi": 0.0},
        "sam": {"psnr": 0.0, "ssim": 0.0, "psnr_ndvi": 0.0, "ssim_ndvi": 0.0},
        "n": 0
    }

    for sample in tqdm(loader, desc="Comparing"):
        r = sample["image_r"].to(device).float()
        g = sample["image_g"].to(device).float()
        y = sample["image_nir"].to(device).float()

        x_base = torch.cat([r, g], dim=1)
        pred_base = baseline_model(x_base)
        if isinstance(pred_base, (tuple, list)):
            pred_base = pred_base[0]

        x_sam = torch.cat([r, g, g], dim=1)
        pred_sam = sam_model(x_sam)

        pred_base = pred_base.clamp(0, 1)
        pred_sam = pred_sam.clamp(0, 1)
        y = y.clamp(0, 1)
        r = r.clamp(0, 1)

        ndvi_true = calculate_ndvi(y, r)
        ndvi_base = calculate_ndvi(pred_base, r)
        ndvi_sam = calculate_ndvi(pred_sam, r)

        results["baseline"]["psnr"] += float(psnr_metric(pred_base[0], y[0]))
        results["baseline"]["ssim"] += float(sk_ssim(
            pred_base[0, 0].cpu().numpy(),
            y[0, 0].cpu().numpy(),
            data_range=1.0
        ))
        results["baseline"]["psnr_ndvi"] += float(psnr_from_mse(
            torch.mean((ndvi_base - ndvi_true) ** 2).item(),
            data_range=2.0
        ))
        results["baseline"]["ssim_ndvi"] += float(sk_ssim(
            ndvi_base[0, 0].cpu().numpy(),
            ndvi_true[0, 0].cpu().numpy(),
            data_range=2.0
        ))

        results["sam"]["psnr"] += float(psnr_metric(pred_sam[0], y[0]))
        results["sam"]["ssim"] += float(sk_ssim(
            pred_sam[0, 0].cpu().numpy(),
            y[0, 0].cpu().numpy(),
            data_range=1.0
        ))
        results["sam"]["psnr_ndvi"] += float(psnr_from_mse(
            torch.mean((ndvi_sam - ndvi_true) ** 2).item(),
            data_range=2.0
        ))
        results["sam"]["ssim_ndvi"] += float(sk_ssim(
            ndvi_sam[0, 0].cpu().numpy(),
            ndvi_true[0, 0].cpu().numpy(),
            data_range=2.0
        ))

        results["n"] += 1

    n = results["n"]
    for model_name in ["baseline", "sam"]:
        for metric_name in results[model_name]:
            results[model_name][metric_name] /= n

    print("\n===== FINAL RESULTS =====")
    print(json.dumps(results, indent=2))

    save_path = os.path.join(ROOT_DIR, "sam_nir", "comparison_results_r8_mse_l1_edge.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {save_path}")


if __name__ == "__main__":
    main()