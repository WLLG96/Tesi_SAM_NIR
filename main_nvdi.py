import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import load_swin2_mose, load_config
from data.dataset_cropped import NIRDataset_cropped as NIRDataset

import train.train as train_mod
import train.validate as validate_mod


def _make_loader(dataset, batch_size, shuffle):
    # su macOS/MPS: num_workers=0 evita freeze/leaked semaphores
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def _build_dataset(cfg, split_root, train_flag):
    return NIRDataset(
        root_dir=split_root,
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
        train=train_flag,  # 0=train crop, 1=val/test no-crop
    )


def _pick_device(function_name: str):
    # Train: MPS > CUDA > CPU
    # Validate/Test: CPU (più stabile, niente MPS OOM)
    if function_name == "train":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def _load_ckpt_into_model(generator, ckpt_path: str, map_location="cpu"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # supporta sia checkpoint dict che state_dict puro
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    generator.load_state_dict(state, strict=True)

    epoch = int(ckpt["epoch"]) if isinstance(ckpt, dict) and "epoch" in ckpt else 0
    return ckpt, epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", type=str, choices=["train", "validate", "test"], default="train")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image", type=str, choices=["resize", "cropped"], default="resize")

    # validate/test
    parser.add_argument("--ckpt", type=str, default=None, help="Path checkpoint .pth per validate/test")

    # train resume
    parser.add_argument("--resume", type=str, default=None, help="Riprendi training da checkpoint .pth")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs del config")

    args = parser.parse_args()
    config = load_config(args.config)

    # override epochs da CLI se richiesto
    if args.epochs is not None:
        config["dataset"]["num_epochs"] = int(args.epochs)

    device = _pick_device(args.function)
    print(f"Using device: {device}")

    # --- MODEL ---
    generator = load_swin2_mose(cfg=config)
    criterion = nn.MSELoss()

    # -------------------------
    # TRAIN
    # -------------------------
    if args.function == "train":
        # dataset train + (opzionale) val per avere loader pronto (ma train.py non valida)
        train_dataset = _build_dataset(config, config["dataset"]["train_data_root"], train_flag=0)
        val_dataset = _build_dataset(config, config["dataset"]["val_data_root"], train_flag=1)

        train_loader = _make_loader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
        _ = _make_loader(val_dataset, batch_size=1, shuffle=False)  # non usato qui, ma utile se lo vuoi dopo

        generator.to(device)

        optimizer_g = torch.optim.Adam(
            generator.parameters(),
            lr=float(config["opt"]["lr"]),
            betas=tuple(config["opt"]["betas"]),
        )

        # resume (continua epoche)
        if args.resume is not None:
            print(f"Resuming from: {args.resume}")
            ckpt, start_epoch = _load_ckpt_into_model(generator, args.resume, map_location="cpu")
            # se nel checkpoint c'è l'optimizer, lo ripristiniamo
            if isinstance(ckpt, dict) and "optim_state_dict" in ckpt:
                try:
                    optimizer_g.load_state_dict(ckpt["optim_state_dict"])
                except Exception as e:
                    print(f"[WARN] Non riesco a caricare optimizer state: {e}")

            # passiamo a train.py uno start_epoch se lo supporta, altrimenti il train.py parte da 1
            # (il tuo train.py recente stampa Start epoch: ...)
            try:
                train_mod.train(generator, train_loader, criterion, optimizer_g, device, config, start_epoch=start_epoch + 1)
            except TypeError:
                # fallback: train.py senza start_epoch
                train_mod.train(generator, train_loader, criterion, optimizer_g, device, config)

        else:
            train_mod.train(generator, train_loader, criterion, optimizer_g, device, config)

        print("\nTRAIN FINITO.")
        print("Per validare un checkpoint:")
        print("  PYTHONPATH=. python main_nvdi.py --function validate --config configs/config_linda.yaml --ckpt ./checkpoint_model/ckpt_epoch_XXX.pth --image resize")
        print("Per fare test:")
        print("  PYTHONPATH=. python main_nvdi.py --function test --config configs/config_linda.yaml --ckpt ./checkpoint_model/ckpt_epoch_XXX.pth --image resize")
        return

    # -------------------------
    # VALIDATE o TEST
    # -------------------------
    # carica SOLO il dataset necessario
    if args.function == "validate":
        split_root = config["dataset"]["val_data_root"]
    else:
        split_root = config["dataset"]["test_data_root"]

    dataset = _build_dataset(config, split_root, train_flag=1)
    loader = _make_loader(dataset, batch_size=1, shuffle=False)

    # ckpt obbligatorio per validate/test
    ckpt_path = args.ckpt or config.get("paths", {}).get("resume_ckpt", None)
    if ckpt_path is None:
        raise ValueError("Per validate/test devi passare --ckpt oppure mettere paths.resume_ckpt nel config.")

    # model on CPU (device già CPU qui)
    generator.to(device)
    _, epoch = _load_ckpt_into_model(generator, ckpt_path, map_location="cpu")
    generator.to(device)

    # usa validate.py anche per test (stesse metriche)
    validate_mod.validate(generator, loader, criterion, device, config, epoch)
    print(f"\n{args.function.upper()} FINITO.")


if __name__ == "__main__":
    main()



