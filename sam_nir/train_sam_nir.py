import os
import sys
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sam_nir.sam_encoder_model import SAMNIRModel
from sam_nir.dataset_sam_nir import SAMNIRDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]],
            dtype=torch.float32,
        ).unsqueeze(0)

        sobel_y = torch.tensor(
            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]],
            dtype=torch.float32,
        ).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)

        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)

        pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)
        target_grad = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-6)

        return torch.mean(torch.abs(pred_grad - target_grad))


def calculate_ndvi(nir, red, eps=1e-8):
    return (nir - red) / (nir + red + eps)


def gradient_loss(pred, target):
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

    loss_dx = F.l1_loss(pred_dx, target_dx)
    loss_dy = F.l1_loss(pred_dy, target_dy)

    return loss_dx + loss_dy


def load_config():
    experiment_name = "r8_mse_l1_edge_ndvi_grad"

    return {
        "seed": 42,
        "dataset": {
            "train_data_root": "/Users/lindawandjilando/datasets/rgb2nir_dataset_real/splits/train",
            "val_data_root": "/Users/lindawandjilando/datasets/rgb2nir_dataset_real/splits/val",
            "img_size": 128,
            "batch_size": 1,
            "num_epochs": 5,
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
            "mse_weight": 0.35,
            "l1_weight": 0.35,
            "edge_weight": 0.15,
            "ndvi_weight": 0.30,
            "grad_weight": 0.05,
        },
        "sam": {
            "checkpoint": os.path.join(ROOT_DIR, "checkpoints", "sam_vit_b_01ec64.pth"),
            "lora_rank": 8,
            "freeze_encoder": True,
            "sam_input_size": 1024,
        },
        "paths": {
            "experiment_name": experiment_name,
            "save_dir": os.path.join(ROOT_DIR, "sam_nir", f"checkpoints_{experiment_name}"),
            "log_dir": os.path.join(ROOT_DIR, "sam_nir", f"logs_{experiment_name}"),
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


def save_checkpoint(save_dir, filename, epoch, model, optimizer, train_loss, val_loss, cfg):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": cfg,
        },
        ckpt_path,
    )
    return ckpt_path


def compute_total_loss(
    pred,
    y,
    red,
    mse,
    l1,
    edge,
    mse_weight,
    l1_weight,
    edge_weight,
    ndvi_weight,
    grad_weight,
):
    loss_mse = mse(pred, y)
    loss_l1 = l1(pred, y)
    loss_edge = edge(pred, y)

    ndvi_pred = calculate_ndvi(pred, red)
    ndvi_true = calculate_ndvi(y, red)
    loss_ndvi = l1(ndvi_pred, ndvi_true)

    loss_grad = gradient_loss(pred, y)

    total = (
        mse_weight * loss_mse
        + l1_weight * loss_l1
        + edge_weight * loss_edge
        + ndvi_weight * loss_ndvi
        + grad_weight * loss_grad
    )

    return total, loss_mse, loss_l1, loss_edge, loss_ndvi, loss_grad


@torch.no_grad()
def validate(
    model,
    dataloader,
    mse,
    l1,
    edge,
    mse_weight,
    l1_weight,
    edge_weight,
    ndvi_weight,
    grad_weight,
    device,
):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_edge = 0.0
    total_ndvi = 0.0
    total_grad = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch["image_sam"].to(device).float()
        y = batch["image_nir"].to(device).float()
        red = batch["image_r"].to(device).float()

        pred = model(x)

        loss, loss_mse, loss_l1, loss_edge, loss_ndvi, loss_grad = compute_total_loss(
            pred,
            y,
            red,
            mse,
            l1,
            edge,
            mse_weight,
            l1_weight,
            edge_weight,
            ndvi_weight,
            grad_weight,
        )

        total_loss += float(loss.item())
        total_mse += float(loss_mse.item())
        total_l1 += float(loss_l1.item())
        total_edge += float(loss_edge.item())
        total_ndvi += float(loss_ndvi.item())
        total_grad += float(loss_grad.item())
        n_batches += 1

    n_batches = max(1, n_batches)

    return {
        "val_loss": total_loss / n_batches,
        "val_mse": total_mse / n_batches,
        "val_l1": total_l1 / n_batches,
        "val_edge": total_edge / n_batches,
        "val_ndvi": total_ndvi / n_batches,
        "val_grad": total_grad / n_batches,
    }


def train():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = pick_device()

    print("===================================================")
    print("Device:", device)
    print("Experiment:", cfg["paths"]["experiment_name"])
    print("SAM checkpoint exists:", os.path.exists(cfg["sam"]["checkpoint"]))
    print("Save dir:", cfg["paths"]["save_dir"])
    print("Log dir:", cfg["paths"]["log_dir"])
    print("===================================================")

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
        sam_input_size=cfg["sam"]["sam_input_size"],
    ).to(device)

    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    edge = EdgeLoss().to(device)

    mse_weight = cfg["train"]["mse_weight"]
    l1_weight = cfg["train"]["l1_weight"]
    edge_weight = cfg["train"]["edge_weight"]
    ndvi_weight = cfg["train"]["ndvi_weight"]
    grad_weight = cfg["train"]["grad_weight"]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    history = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "experiment_name": cfg["paths"]["experiment_name"],
        "lora_rank": cfg["sam"]["lora_rank"],
        "freeze_encoder": cfg["sam"]["freeze_encoder"],
        "sam_input_size": cfg["sam"]["sam_input_size"],
        "loss_type": "mse_l1_edge_ndvi_grad_combined",
        "mse_weight": mse_weight,
        "l1_weight": l1_weight,
        "edge_weight": edge_weight,
        "ndvi_weight": ndvi_weight,
        "grad_weight": grad_weight,
        "num_epochs": cfg["dataset"]["num_epochs"],
        "learning_rate": cfg["train"]["lr"],
        "weight_decay": cfg["train"]["weight_decay"],
        "epochs": [],
    }

    num_epochs = cfg["dataset"]["num_epochs"]
    best_val_loss = float("inf")
    history_path = os.path.join(cfg["paths"]["log_dir"], "train_history.json")

    for epoch in range(1, num_epochs + 1):
        model.train()

        running_loss = 0.0
        running_mse = 0.0
        running_l1 = 0.0
        running_edge = 0.0
        running_ndvi = 0.0
        running_grad = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", leave=True)

        for batch in pbar:
            x = batch["image_sam"].to(device).float()
            y = batch["image_nir"].to(device).float()
            red = batch["image_r"].to(device).float()

            optimizer.zero_grad(set_to_none=True)

            pred = model(x)

            loss, loss_mse, loss_l1, loss_edge, loss_ndvi, loss_grad = compute_total_loss(
                pred,
                y,
                red,
                mse,
                l1,
                edge,
                mse_weight,
                l1_weight,
                edge_weight,
                ndvi_weight,
                grad_weight,
            )

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_mse += float(loss_mse.item())
            running_l1 += float(loss_l1.item())
            running_edge += float(loss_edge.item())
            running_ndvi += float(loss_ndvi.item())
            running_grad += float(loss_grad.item())
            n_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.6f}",
                mse=f"{loss_mse.item():.6f}",
                l1=f"{loss_l1.item():.6f}",
                edge=f"{loss_edge.item():.6f}",
                ndvi=f"{loss_ndvi.item():.6f}",
                grad=f"{loss_grad.item():.6f}",
            )

        n_batches = max(1, n_batches)

        train_loss = running_loss / n_batches
        train_mse = running_mse / n_batches
        train_l1 = running_l1 / n_batches
        train_edge = running_edge / n_batches
        train_ndvi = running_ndvi / n_batches
        train_grad = running_grad / n_batches

        val_metrics = validate(
            model,
            val_loader,
            mse,
            l1,
            edge,
            mse_weight,
            l1_weight,
            edge_weight,
            ndvi_weight,
            grad_weight,
            device,
        )

        val_loss = val_metrics["val_loss"]

        print(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.6f} "
            f"(mse={train_mse:.6f}, l1={train_l1:.6f}, edge={train_edge:.6f}, "
            f"ndvi={train_ndvi:.6f}, grad={train_grad:.6f}) | "
            f"val_loss={val_loss:.6f} "
            f"(mse={val_metrics['val_mse']:.6f}, l1={val_metrics['val_l1']:.6f}, "
            f"edge={val_metrics['val_edge']:.6f}, ndvi={val_metrics['val_ndvi']:.6f}, "
            f"grad={val_metrics['val_grad']:.6f})"
        )

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_l1": train_l1,
            "train_edge": train_edge,
            "train_ndvi": train_ndvi,
            "train_grad": train_grad,
            "val_loss": val_loss,
            "val_mse": val_metrics["val_mse"],
            "val_l1": val_metrics["val_l1"],
            "val_edge": val_metrics["val_edge"],
            "val_ndvi": val_metrics["val_ndvi"],
            "val_grad": val_metrics["val_grad"],
        }
        history["epochs"].append(epoch_info)

        if epoch % cfg["train"]["save_every"] == 0:
            ckpt_path = save_checkpoint(
                save_dir=cfg["paths"]["save_dir"],
                filename=f"sam_nir_epoch_{epoch:03d}.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                val_loss=val_loss,
                cfg=cfg,
            )
            print("Saved checkpoint:", ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = save_checkpoint(
                save_dir=cfg["paths"]["save_dir"],
                filename="sam_nir_best.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                val_loss=val_loss,
                cfg=cfg,
            )
            print(f"New best checkpoint saved: {best_ckpt_path} (val_loss={val_loss:.6f})")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print("===================================================")
    print("Training finished.")
    print("Best val_loss:", best_val_loss)
    print("History saved to:", history_path)
    print("===================================================")


if __name__ == "__main__":
    train()