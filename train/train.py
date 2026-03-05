# train/train.py
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _unpack_sample(sample, device):
    required = ["image_r", "image_g", "image_nir"]
    if not all(k in sample for k in required):
        raise ValueError(f"Chiavi sample dict non trovate. keys={list(sample.keys())}")

    r = sample["image_r"].to(device, non_blocking=True).float()
    g = sample["image_g"].to(device, non_blocking=True).float()
    nir = sample["image_nir"].to(device, non_blocking=True).float()

    if r.dim() == 3:
        r = r.unsqueeze(0)
    if g.dim() == 3:
        g = g.unsqueeze(0)
    if nir.dim() == 3:
        nir = nir.unsqueeze(0)

    x = torch.cat([r, g], dim=1)  # [B,2,H,W]
    y = nir                       # [B,1,H,W]
    return x, y


def _align_pred_to_target(pred, y):
    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    if pred.dim() != 4:
        raise ValueError(
            f"pred deve essere 4D [B,C,H,W], trovato dim={pred.dim()} shape={tuple(pred.shape)}"
        )

    # canali
    if pred.size(1) != y.size(1):
        if pred.size(1) > 1 and y.size(1) == 1:
            pred = pred[:, :1, :, :]
        elif y.size(1) == 1:
            pred = pred[:, :1, :, :]

    # spazio
    if pred.shape[-2:] != y.shape[-2:]:
        pred = F.interpolate(pred, size=y.shape[-2:], mode="bilinear", align_corners=False)

    return pred


def _save_checkpoint(save_dir, epoch, generator, optimizer_g, avg_loss):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": generator.state_dict(),
        "optim_state_dict": optimizer_g.state_dict(),
        "avg_loss": float(avg_loss),
    }
    path = os.path.join(save_dir, f"ckpt_epoch_{epoch:03d}.pth")
    torch.save(ckpt, path)
    return path


def _maybe_empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _try_resume(resume_path, generator, optimizer_g):
    """
    Carica checkpoint (se esiste) e ritorna start_epoch corretto.
    Supporta:
      - dict con model_state_dict/optim_state_dict/epoch
      - state_dict puro (solo pesi)
    """
    if resume_path is None:
        return 1  # parte da epoch 1

    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint non trovato: {resume_path}")

    print(f"Resuming checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location="cpu")

    # 1) pesi
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        generator.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        # ckpt è direttamente uno state_dict
        generator.load_state_dict(ckpt, strict=True)

    # 2) optimizer (se disponibile)
    if isinstance(ckpt, dict) and "optim_state_dict" in ckpt and optimizer_g is not None:
        try:
            optimizer_g.load_state_dict(ckpt["optim_state_dict"])
        except Exception as e:
            print(f"[WARN] Impossibile caricare optimizer state (continuo lo stesso): {e}")

    # 3) epoch
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        last_epoch = int(ckpt["epoch"])
        return last_epoch + 1

    return 1


def train(
    generator,
    train_loader,
    criterion,
    optimizer_g,
    device,
    config,
    start_epoch=1,
    resume_path=None,
):
    """
    Training loop.

    Parametri:
      - start_epoch: int (default 1)
      - resume_path: path ckpt .pth (se passato, sovrascrive start_epoch caricando epoch dal ckpt)

    Config attesa:
      config["dataset"]["num_epochs"]
      config["paths"]["save_dir"]

    Opzionali:
      config["train"]["grad_clip"] (default 1.0)
      config["train"]["log_every"] (default 50)
      config["train"]["use_amp"]   (default False, SOLO CUDA)
      config["paths"]["save_every"] (default 1)
    """
    num_epochs = int(config["dataset"]["num_epochs"])
    save_dir = config["paths"]["save_dir"]

    grad_clip = float(config.get("train", {}).get("grad_clip", 1.0))
    log_every = int(config.get("train", {}).get("log_every", 50))
    save_every = int(config.get("paths", {}).get("save_every", 1))

    # se resume_path è dato, carica e calcola start_epoch corretto
    if resume_path is not None:
        start_epoch = _try_resume(resume_path, generator, optimizer_g)

    # AMP SOLO CUDA
    use_amp_cfg = bool(config.get("train", {}).get("use_amp", False))
    use_amp = use_amp_cfg and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("Training")
    print(f"Start epoch: {start_epoch} / {num_epochs}")

    for epoch in range(int(start_epoch), num_epochs + 1):
        generator.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", leave=True)

        for step, sample in enumerate(pbar, start=1):
            x, y = _unpack_sample(sample, device)

            optimizer_g.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = generator(x)
                    pred = _align_pred_to_target(out, y)
                    loss = criterion(pred, y)

                scaler.scale(loss).backward()

                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer_g)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)

                scaler.step(optimizer_g)
                scaler.update()
            else:
                out = generator(x)
                pred = _align_pred_to_target(out, y)
                loss = criterion(pred, y)

                loss.backward()

                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)

                optimizer_g.step()

            running_loss += float(loss.item())

            if step % log_every == 0 or step == 1:
                pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch}/{num_epochs}] - Avg Loss: {avg_loss:.6f}")

        # salva ckpt
        if save_every > 0 and (epoch % save_every == 0 or epoch == num_epochs):
            path = _save_checkpoint(save_dir, epoch, generator, optimizer_g, avg_loss)
            print(f"Saved checkpoint: {path}")

        _maybe_empty_cache(device)

    return generator
