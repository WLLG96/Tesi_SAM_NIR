import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
class NIRDataset_combined(Dataset):             
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # Trova i file che contengono "_R.TIF"
        self.image_prefixes = sorted([f.replace('_R.TIF', '') for f in os.listdir(os.path.join(root_dir, 'images')) if f.endswith('_R.TIF')])

    def __len__(self):
        return len(self.image_prefixes)

    def __getitem__(self, idx):
        # Prendi il prefisso comune
        prefix = self.image_prefixes[idx]

        # Costruisci i nomi dei file corrispondenti per R, G e NIR
        img_name_r = os.path.join(self.root_dir, 'images', f"{prefix}_R.TIF")
        img_name_g = os.path.join(self.root_dir, 'images', f"{prefix}_G.TIF")
        img_name_nir = os.path.join(self.root_dir, 'images', f"{prefix}_NIR.TIF")

        # Carica le immagini
        image_r = Image.open(img_name_r)  
        image_g = Image.open(img_name_g)  
        image_nir = Image.open(img_name_nir) 

        # Converte le immagini in array numpy
        image_r = np.array(image_r).astype('float')
        image_g = np.array(image_g).astype('float')
        image_nir = np.array(image_nir).astype('float')

        # Converti le immagini in Tensor
        image_r = torch.tensor(image_r, dtype=torch.float32).unsqueeze(0)
        image_g = torch.tensor(image_g, dtype=torch.float32).unsqueeze(0)
        image_nir = torch.tensor(image_nir, dtype=torch.float32).unsqueeze(0)

        # Restituisce un dizionario con le immagini
        sample = {
            'image_r': image_r, 
            'image_g': image_g, 
            'image_nir': image_nir,
            'image_name': prefix
        }
        return sample
