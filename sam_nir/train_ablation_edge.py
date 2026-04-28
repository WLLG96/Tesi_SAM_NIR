import os
import sam_nir.train_sam_nir as base

ROOT_DIR = base.ROOT_DIR

# salva funzione originale
original_load_config = base.load_config


def load_config_edge():
    cfg = original_load_config()

    experiment_name = "r8_mse_l1_edge_ablation"

    cfg["paths"]["experiment_name"] = experiment_name
    cfg["paths"]["save_dir"] = os.path.join(
        ROOT_DIR, "sam_nir", f"checkpoints_{experiment_name}"
    )
    cfg["paths"]["log_dir"] = os.path.join(
        ROOT_DIR, "sam_nir", f"logs_{experiment_name}"
    )

    cfg["dataset"]["num_epochs"] = 2

    cfg["train"]["mse_weight"] = 0.35
    cfg["train"]["l1_weight"] = 0.35
    cfg["train"]["edge_weight"] = 0.15
    cfg["train"]["ndvi_weight"] = 0.0
    cfg["train"]["grad_weight"] = 0.0

    return cfg


base.load_config = load_config_edge

if __name__ == "__main__":
    base.train()
