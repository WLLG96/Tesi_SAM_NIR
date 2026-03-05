import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Swin2MoSE


# -------------------------
# Helpers: broadcast stats
# -------------------------
def _as_1d_tensor(x, device, dtype=torch.float32):
    """
    Converte mean/std in tensore 1D [C].
    Accetta:
      - float/int
      - list/tuple di numeri
      - torch.Tensor
    """
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        return t.flatten()
    if isinstance(x, (float, int)):
        return torch.tensor([float(x)], device=device, dtype=dtype)
    if isinstance(x, (list, tuple)):
        return torch.tensor([float(v) for v in x], device=device, dtype=dtype).flatten()
    raise TypeError(f"mean/std type non supportato: {type(x)}")


def to_shape(stats_1d, tensor_4d):
    """
    stats_1d: Tensor [C] oppure [1]
    tensor_4d: Tensor [B,C,H,W]
    ritorna stats broadcastabili: [B,C,1,1]
    """
    if tensor_4d.dim() != 4:
        raise ValueError(f"tensor deve essere [B,C,H,W], trovato shape={tuple(tensor_4d.shape)}")

    B, C = tensor_4d.shape[0], tensor_4d.shape[1]

    # Se stats ha 1 valore ma C>1, lo ripetiamo
    if stats_1d.numel() == 1 and C > 1:
        stats_1d = stats_1d.repeat(C)

    if stats_1d.numel() != C:
        raise ValueError(f"stats ha {stats_1d.numel()} canali ma tensor ne ha {C}")

    return stats_1d.view(1, C, 1, 1).repeat(B, 1, 1, 1)


def normalize(tensor, mean, std, eps=1e-8):
    """
    Normalize: (x - mean) / std
    tensor: [B,C,H,W]
    mean/std: float o lista C
    """
    mean_t = _as_1d_tensor(mean, tensor.device)
    std_t = _as_1d_tensor(std, tensor.device)
    mean_b = to_shape(mean_t, tensor)
    std_b = to_shape(std_t, tensor)
    return (tensor - mean_b) / (std_b + eps)


def denormalize(tensor, mean, std):
    """
    Denormalize: x * std + mean
    """
    mean_t = _as_1d_tensor(mean, tensor.device)
    std_t = _as_1d_tensor(std, tensor.device)
    mean_b = to_shape(mean_t, tensor)
    std_b = to_shape(std_t, tensor)
    return tensor * std_b + mean_b


# -------------------------
# Config / Model loader
# -------------------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError("La configurazione non è stata caricata correttamente")
    if not isinstance(cfg, dict):
        raise ValueError(f"Config inattesa: {type(cfg)} (mi aspettavo dict)")
    return cfg


def _resolve_norm_layer(n):
    """
    Accetta:
      - None
      - "layernorm", "LayerNorm", "layer_norm"
      - nn.LayerNorm
      - altre stringhe future -> errore chiaro
    """
    if n is None:
        return nn.LayerNorm

    if n is nn.LayerNorm:
        return nn.LayerNorm

    if isinstance(n, str):
        key = n.strip().lower().replace("_", "")
        if key in ["layernorm", "ln"]:
            return nn.LayerNorm
        raise ValueError(f"norm_layer string non supportata: {n}")

    # se arriva una classe callable tipo nn.BatchNorm2d etc.
    if isinstance(n, type) and issubclass(n, nn.Module):
        return n

    raise ValueError(f"norm_layer non supportata: {n} ({type(n)})")


def load_swin2_mose(cfg):
    model_cfg = cfg["super_res"]["model"].copy()

    # conversione robusta norm_layer
    model_cfg["norm_layer"] = _resolve_norm_layer(model_cfg.get("norm_layer"))

    sr_model = Swin2MoSE(**model_cfg)
    return sr_model


# -------------------------
# Inference helper (opzionale, legacy)
# -------------------------
def run_swin2_mose(model, lr, hr, device="cuda", lr_stats=None, hr_stats=None):
    """
    Helper legacy per casi Sentinel-like.
    Nel tuo progetto RGB->NIR non è indispensabile.
    Per evitare dipendenze da model.cfg, stats vanno passate esplicitamente.
    """
    model = model.to(device).eval()

    if lr_stats is None or hr_stats is None:
        raise ValueError("Per run_swin2_mose devi passare lr_stats e hr_stats (mean/std) esplicitamente.")

    # lr, hr numpy -> torch
    lr_orig = torch.from_numpy(lr)[None].float().to(device)
    hr_orig = torch.from_numpy(hr)[None].float().to(device)

    lr_n = normalize(lr_orig, mean=lr_stats["mean"], std=lr_stats["std"])
    hr_n = normalize(hr_orig, mean=hr_stats["mean"], std=hr_stats["std"])

    with torch.no_grad():
        sr = model(lr_n)
        if not torch.is_tensor(sr):
            sr = sr[0]

    sr = denormalize(sr, mean=hr_stats["mean"], std=hr_stats["std"])

    # torna a uint16
    sr = sr.round().clamp(min=0).cpu().numpy().astype("uint16").squeeze()
    lr_u = lr_orig.round().clamp(min=0).cpu().numpy().astype("uint16").squeeze()
    hr_u = hr_orig.round().clamp(min=0).cpu().numpy().astype("uint16").squeeze()

    # se serve riallineamento size
    if sr.ndim >= 2 and hr_u.ndim >= 2 and sr.shape[-2:] != hr_u.shape[-2:]:
        sr = F.interpolate(
            torch.from_numpy(sr)[None].float(),
            size=hr_u.shape[-2:],
            mode="nearest"
        ).squeeze().numpy().astype("uint16")

    return {"lr": lr_u, "sr": sr, "hr": hr_u}