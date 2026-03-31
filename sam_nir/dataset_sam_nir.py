import os
import sys
import torch
from torch.utils.data import Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data.dataset_cropped import NIRDataset_cropped


class SAMNIRDataset(Dataset):
    """
    Adapter dataset per usare il dataset RGB->NIR esistente con SAM.

    Input SAM:
        x_sam = [R, G, G]   -> shape [3, H, W]

    Target:
        y_nir = NIR         -> shape [1, H, W]
    """

    def __init__(
        self,
        root_dir,
        img_size=128,
        rmean=None,
        rstd=None,
        gmean=None,
        gstd=None,
        nirmean=None,
        nirstd=None,
        rmin=None,
        rmax=None,
        gmin=None,
        gmax=None,
        nirmin=None,
        nirmax=None,
        train=0,
        verbose=True,
    ):
        self.base_dataset = NIRDataset_cropped(
            root_dir=root_dir,
            img_size=img_size,
            rmean=rmean,
            rstd=rstd,
            gmean=gmean,
            gstd=gstd,
            nirmean=nirmean,
            nirstd=nirstd,
            rmin=rmin,
            rmax=rmax,
            gmin=gmin,
            gmax=gmax,
            nirmin=nirmin,
            nirmax=nirmax,
            train=train,
            verbose=verbose,
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        r = sample["image_r"]      # [1,H,W]
        g = sample["image_g"]      # [1,H,W]
        nir = sample["image_nir"]  # [1,H,W]

        # Costruzione input a 3 canali per SAM
        x_sam = torch.cat([r, g, g], dim=0)   # [3,H,W]

        return {
            "image_sam": x_sam,
            "image_nir": nir,
            "image_r": r,
            "image_g": g,
            "image_name": sample["image_name"],
        }