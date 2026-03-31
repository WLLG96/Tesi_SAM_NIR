import os
import sys
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sam_nir.sam_encoder_model import SAMNIRModel
from sam_nir.dataset_sam_nir import SAMNIRDataset


def load_config():
    """
    Config minima hardcoded per partire subito.
    Poi, se vuoi, la spostiamo in un file yaml.
    """
    return {
        "dataset": {
            "train_data_root": "/Users/lindawandjilando/datasets/rgb2nir_dataset_real/splits/train",
            "val_data_root": "/Users/lindawandjilando/datasets/rgb2nir_dataset_real/splits/val",
            "img_size": 128,
            "batch_size": 1,
            "num_epochs": 3,
        },
        "norm": {
            "mean_red": 0.0,
            "std_red": 1.0,
            "mean_green": 0.0,
            "std_green": 1.0,
            "mean_nir": 0.0,
            "std_nir": 1.0,
            "min_r": 0.0,
            "max_r": 65535.0,
            "min_g": 0.0,
            "max_g": 65535.0,
            "min_n": 0.0,
            "max_n": 65535.0,
        },
        "train": {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "save_every": 1,
        },
        "sam": {
            "checkpoint": os.path.join(ROOT_DIR, "checkpoints", "sam_vit_b_01ec64.pth"),
            "lora_rank": 4,
            "freeze_encoder": True,
        },
        "paths": {
            "save_dir": os.path.join(ROOT_DIR, "sam_nir", "checkpoints"),
            "log_dir": os.path.join(ROOT_DIR, "sam_nir", "logs"),
        },
    }


def make_dataloader(dataset, batch_size=1, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def build_train_dataset(cfg):
    return SAMNIRDataset(
        root_dir=cfg["dataset"]["train_data_root"],
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
        train=0,
        verbose=True,
    )


def build_val_dataset(cfg):
    return SAMNIRDataset(
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


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(save_dir, epoch, model, optimizer, avg_loss):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"sam_nir_epoch_{epoch:03d}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss,
        },
        ckpt_path,
    )
    return ckpt_path


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch["image_sam"].to(device).float()
        y = batch["image_nir"].to(device).float()

        pred = model(x)
        loss = criterion(pred, y)

        total_loss += float(loss.item())
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    return avg_loss


def train():
    cfg = load_config()
    device = pick_device()

    print("Device:", device)
    print("SAM checkpoint exists:", os.path.exists(cfg["sam"]["checkpoint"]))

    os.makedirs(cfg["paths"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)

    train_dataset = build_train_dataset(cfg)
    val_dataset = build_val_dataset(cfg)

    train_loader = make_dataloader(
        train_dataset,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=True,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=1,
        shuffle=False,
    )

    model = SAMNIRModel(
        sam_ckpt_path=cfg["sam"]["checkpoint"],
        lora_rank=cfg["sam"]["lora_rank"],
        freeze_encoder=cfg["sam"]["freeze_encoder"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    history = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "epochs": [],
    }

    num_epochs = cfg["dataset"]["num_epochs"]

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", leave=True)

        for batch in pbar:
            x = batch["image_sam"].to(device).float()
            y = batch["image_nir"].to(device).float()

            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = running_loss / max(1, n_batches)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if epoch % cfg["train"]["save_every"] == 0:
            ckpt_path = save_checkpoint(
                cfg["paths"]["save_dir"],
                epoch,
                model,
                optimizer,
                train_loss,
            )
            print("Saved checkpoint:", ckpt_path)

    history_path = os.path.join(cfg["paths"]["log_dir"], "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("Training finished.")
    print("History saved to:", history_path)


if __name__ == "__main__":
    train()