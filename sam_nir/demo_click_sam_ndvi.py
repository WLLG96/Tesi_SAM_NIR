import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import cm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

SAM_LORA_DIR = os.path.join(ROOT_DIR, "Sam_LoRA")
if SAM_LORA_DIR not in sys.path:
    sys.path.append(SAM_LORA_DIR)

from sam_nir.sam_encoder_model import SAMNIRModel
from segment_anything import sam_model_registry, SamPredictor


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_ndvi(nir, red, eps=1e-8):
    return (nir - red) / (nir + red + eps)


def load_rgb_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    return img_np


def ask_click(image_rgb):
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.title("Clicca un punto dentro l'area di interesse, poi chiudi la finestra")
    pts = plt.ginput(1, timeout=-1)
    plt.close()

    if len(pts) == 0:
        raise RuntimeError("Nessun click ricevuto.")
    x, y = pts[0]
    return int(round(x)), int(round(y))


def load_sam_segmentor(device, sam_ckpt):
    sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def load_nir_model(device, sam_pretrained_ckpt, sam_nir_ckpt):
    model = SAMNIRModel(
        sam_ckpt_path=sam_pretrained_ckpt,
        lora_rank=4,
        freeze_encoder=True,
    ).to(device)

    ckpt = torch.load(sam_nir_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def make_samnir_input(image_rgb):
    """
    Input coerente con il training SAM->NIR:
    [R, G, G]
    """
    img = image_rgb.astype(np.float32) / 255.0
    r = img[:, :, 0]
    g = img[:, :, 1]
    x = np.stack([r, g, g], axis=0)  # [3,H,W]
    x = torch.from_numpy(x).float().unsqueeze(0)
    return x


def get_best_mask_from_click(predictor, image_rgb, click_xy):
    predictor.set_image(image_rgb)

    point_coords = np.array([[click_xy[0], click_xy[1]]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]
    best_score = float(scores[best_idx])
    return best_mask, best_score


def overlay_mask_on_rgb(image_rgb, mask, alpha=0.45):
    overlay = image_rgb.astype(np.float32).copy()
    color = np.array([255, 0, 0], dtype=np.float32)

    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def overlay_ndvi_on_rgb(image_rgb, ndvi, mask, alpha=0.60):
    """
    Colora solo la regione segmentata con la colormap NDVI.
    """
    rgb = image_rgb.astype(np.float32) / 255.0
    ndvi_norm = np.clip((ndvi + 1.0) / 2.0, 0.0, 1.0)
    ndvi_color = cm.get_cmap("RdYlGn")(ndvi_norm)[..., :3]  # RGB in [0,1]

    out = rgb.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * ndvi_color[mask]
    out = np.clip(out, 0, 1)
    return out


def compute_masked_stats(ndvi, mask):
    values = ndvi[mask]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
        }

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path immagine RGB")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(ROOT_DIR, "sam_nir", "demo_outputs"),
        help="Cartella output",
    )
    args = parser.parse_args()

    device = pick_device()

    sam_pretrained_ckpt = os.path.join(ROOT_DIR, "checkpoints", "sam_vit_b_01ec64.pth")
    sam_nir_ckpt = os.path.join(ROOT_DIR, "sam_nir", "checkpoints", "sam_nir_epoch_002.pth")

    os.makedirs(args.outdir, exist_ok=True)

    print("Device:", device)
    print("Loading image...")
    image_rgb = load_rgb_image(args.image)

    print("Click selection...")
    click_xy = ask_click(image_rgb)
    print("Selected point:", click_xy)

    print("Loading SAM segmentor...")
    predictor = load_sam_segmentor(device, sam_pretrained_ckpt)

    print("Segmenting ROI...")
    mask, mask_score = get_best_mask_from_click(predictor, image_rgb, click_xy)
    print("Best SAM score:", mask_score)

    print("Loading SAM->NIR model...")
    nir_model = load_nir_model(device, sam_pretrained_ckpt, sam_nir_ckpt)

    print("Predicting NIR...")
    x = make_samnir_input(image_rgb).to(device)
    pred_nir = nir_model(x).squeeze(0).squeeze(0).cpu().numpy()  # [H,W]

    red = x[:, 0:1, :, :].squeeze(0).squeeze(0).cpu().numpy()    # [H,W] in [0,1]
    ndvi = calculate_ndvi(pred_nir, red)

    stats = compute_masked_stats(ndvi, mask)

    # Visuals
    rgb_with_mask = overlay_mask_on_rgb(image_rgb, mask)
    ndvi_overlay = overlay_ndvi_on_rgb(image_rgb, ndvi, mask)

    base_name = os.path.splitext(os.path.basename(args.image))[0]

    # Save raw visuals
    Image.fromarray(rgb_with_mask).save(os.path.join(args.outdir, f"{base_name}_sam_mask_overlay.png"))

    plt.figure(figsize=(6, 6))
    plt.imshow(pred_nir, cmap="gray")
    plt.title("Predicted NIR")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{base_name}_nir_pred.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Predicted NDVI")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{base_name}_ndvi_pred.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(ndvi_overlay)
    plt.title("NDVI only on selected region")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{base_name}_ndvi_overlay.png"), dpi=150)
    plt.close()

    # Save full report figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].scatter([click_xy[0]], [click_xy[1]], c="red", s=60)
    axes[0, 0].set_title("RGB + click")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_with_mask)
    axes[0, 1].set_title(f"SAM mask (score={mask_score:.3f})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_nir, cmap="gray")
    axes[0, 2].set_title("Predicted NIR")
    axes[0, 2].axis("off")

    im1 = axes[1, 0].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1, 0].set_title("Predicted NDVI")
    axes[1, 0].axis("off")
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(ndvi_overlay)
    axes[1, 1].set_title("NDVI on ROI only")
    axes[1, 1].axis("off")

    stats_text = (
        f"Pixels in ROI: {stats['count']}\n"
        f"Mean NDVI: {stats['mean']:.4f}\n"
        f"Min NDVI: {stats['min']:.4f}\n"
        f"Max NDVI: {stats['max']:.4f}\n"
        f"Std NDVI: {stats['std']:.4f}"
        if stats["count"] > 0 else "ROI vuota"
    )
    axes[1, 2].text(0.05, 0.5, stats_text, fontsize=12, va="center")
    axes[1, 2].set_title("ROI statistics")
    axes[1, 2].axis("off")

    plt.tight_layout()
    report_path = os.path.join(args.outdir, f"{base_name}_demo_report.png")
    plt.savefig(report_path, dpi=150)
    plt.close()

    # Save stats json
    stats_json = {
        "image": args.image,
        "click_xy": [int(click_xy[0]), int(click_xy[1])],
        "sam_score": mask_score,
        "stats": stats,
    }
    with open(os.path.join(args.outdir, f"{base_name}_demo_stats.json"), "w") as f:
        json.dump(stats_json, f, indent=2)

    print("\nSaved outputs in:", args.outdir)
    print("Main report:", report_path)
    print("Stats:", json.dumps(stats_json, indent=2))


if __name__ == "__main__":
    main()