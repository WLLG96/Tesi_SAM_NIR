import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class NIRDataset_cropped(Dataset):
    def __init__(
        self,
        root_dir,
        img_size=None,
        rmean=None, rstd=None,
        gmean=None, gstd=None,
        nirmean=None, nirstd=None,
        rmin=None, rmax=None,
        gmin=None, gmax=None,
        nirmin=None, nirmax=None,
        train=None,
        verbose=True,
    ):
        self.root_dir = root_dir
        self.img_size = int(img_size) if img_size is not None else 128

        # mean/std (opzionali)
        self.rmean = rmean
        self.rstd = rstd
        self.gmean = gmean
        self.gstd = gstd
        self.nirmean = nirmean
        self.nirstd = nirstd

        # min/max
        self.rmin = rmin
        self.rmax = rmax
        self.gmin = gmin
        self.gmax = gmax
        self.nirmin = nirmin
        self.nirmax = nirmax

        # train flag: 0=train, 1=val/test (come nel tuo main)
        self.train = train

        images_dir = os.path.join(root_dir, "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Cartella non trovata: {images_dir}")

        # Supporta sia *_MS_R.TIF che *_R.TIF
        r_files_ms = [f for f in os.listdir(images_dir) if f.endswith("_MS_R.TIF")]
        r_files_plain = [f for f in os.listdir(images_dir) if f.endswith("_R.TIF")]

        if len(r_files_ms) > 0:
            self.suffix_r = "_MS_R.TIF"
            self.suffix_g = "_MS_G.TIF"
            self.suffix_n = "_MS_NIR.TIF"
            r_files = r_files_ms
        elif len(r_files_plain) > 0:
            self.suffix_r = "_R.TIF"
            self.suffix_g = "_G.TIF"
            self.suffix_n = "_NIR.TIF"
            r_files = r_files_plain
        else:
            raise FileNotFoundError(
                f"Nessun file R trovato in {images_dir}. Attesi *_MS_R.TIF o *_R.TIF"
            )

        prefixes = sorted([f.replace(self.suffix_r, "") for f in r_files])

        # Filtra solo triplette valide
        valid = []
        for p in prefixes:
            pr = os.path.join(images_dir, f"{p}{self.suffix_r}")
            pg = os.path.join(images_dir, f"{p}{self.suffix_g}")
            pn = os.path.join(images_dir, f"{p}{self.suffix_n}")
            if os.path.exists(pr) and os.path.exists(pg) and os.path.exists(pn):
                valid.append(p)

        if len(valid) == 0:
            raise RuntimeError("Nessun campione valido con R+G+NIR trovato.")

        self.image_prefixes = valid

        # ---- transforms ----
        # TRAIN: RandomCrop
        self.train_crop = transforms.RandomCrop((self.img_size, self.img_size))
        # VAL/TEST: CenterCrop (così è deterministico e veloce)
        self.eval_crop = transforms.CenterCrop((self.img_size, self.img_size))

        if verbose:
            print(f"[Dataset] images_dir={images_dir}")
            print(f"[Dataset] R files trovati: {len(r_files)}")
            print(f"[Dataset] Triplette valide (R+G+NIR): {len(self.image_prefixes)}")

    def __len__(self):
        return len(self.image_prefixes)

    @staticmethod
    def _to_float32_np(pil_img):
        return np.array(pil_img).astype(np.float32)

    @staticmethod
    def _minmax(x, vmin, vmax):
        if vmin is None or vmax is None:
            return x
        vmin = float(vmin)
        vmax = float(vmax)
        denom = vmax - vmin
        if denom <= 0:
            return x
        x = (x - vmin) / (denom + 1e-8)
        return np.clip(x, 0.0, 1.0)

    @staticmethod
    def _meanstd(x, mean, std):
        if mean is None or std is None:
            return x
        std = float(std)
        if std == 0:
            return x
        return (x - float(mean)) / (std + 1e-8)

    def __getitem__(self, idx):
        prefix = self.image_prefixes[idx]
        images_dir = os.path.join(self.root_dir, "images")

        img_name_r = os.path.join(images_dir, f"{prefix}{self.suffix_r}")
        img_name_g = os.path.join(images_dir, f"{prefix}{self.suffix_g}")
        img_name_n = os.path.join(images_dir, f"{prefix}{self.suffix_n}")

        # load -> np float32
        r = self._to_float32_np(Image.open(img_name_r))
        g = self._to_float32_np(Image.open(img_name_g))
        n = self._to_float32_np(Image.open(img_name_n))

        # stack per crop coerente: (3, H, W)
        rgn = torch.from_numpy(np.stack([r, g, n], axis=0))

        # Crop coerente SEMPRE:
        # - train==0 -> random
        # - else -> center (val/test)
        if self.train == 0:
            rgn = self.train_crop(rgn)
        else:
            rgn = self.eval_crop(rgn)

        # torna a numpy (H,W)
        r = rgn[0].numpy()
        g = rgn[1].numpy()
        n = rgn[2].numpy()

        # 1) MIN-MAX -> [0,1]
        r = self._minmax(r, self.rmin, self.rmax)
        g = self._minmax(g, self.gmin, self.gmax)
        n = self._minmax(n, self.nirmin, self.nirmax)

        # 2) mean/std (opzionale)
        r = self._meanstd(r, self.rmean, self.rstd)
        g = self._meanstd(g, self.gmean, self.gstd)
        n = self._meanstd(n, self.nirmean, self.nirstd)

        # to torch float32 + channel dim (1,H,W)
        image_array_r = torch.from_numpy(r).float().unsqueeze(0)
        image_array_g = torch.from_numpy(g).float().unsqueeze(0)
        image_array_nir = torch.from_numpy(n).float().unsqueeze(0)

        return {
            "image_r": image_array_r,
            "image_g": image_array_g,
            "image_nir": image_array_nir,
            "image_name": prefix,
        }