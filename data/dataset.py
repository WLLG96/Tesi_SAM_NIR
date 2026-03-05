import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import cv2
from torchvision import transforms

class ResizeTo:
    def __init__(self, height, width):
        self.size = (height, width)

    def __call__(self, image):
        # Resize dell'immagine
        resized_image_array = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        return resized_image_array


class NIRDataset(Dataset):             
    def __init__(self, root_dir,img_size=None, rmean=None, rstd=None,gmean=None, gstd=None,nirmean=None, nirstd=None,rmin=None,rmax=None,gmin=None,gmax=None,nirmin=None,nirmax=None,train=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.rmean = rmean
        self.rstd = rstd
        self.gmean = gmean
        self.gstd = gstd
        self.nirmean = nirmean
        self.nirstd = nirstd
        self.rmin = rmin
        self.rmax = rmax
        self.gmin = gmin
        self.gmax = gmax
        self.nirmin = nirmin
        self.nirmax = nirmax
        self.train = train

        # Trova i file che contengono "_R.TIF"
        self.image_prefixes = sorted([f.replace('_R.TIF', '') for f in os.listdir(os.path.join(root_dir, 'images')) if f.endswith('_R.TIF')])

        # Trasformazioni da applicare alle immagini durante l'addestramento
        self.transforms = transforms.Compose([
            ResizeTo(self.img_size, self.img_size),
        ]) 

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


        image_r = np.array(image_r).astype('float')
        image_g = np.array(image_g).astype('float')
        image_nir = np.array(image_nir).astype('float')


        #print("Image shape:", image_r.shape)
        #print("Image type:", image_r.dtype)
        #print(f"Not Normalized image 1 - Min: {image_r.min()}, Max: {image_r.max()}")
        #print(f"Not Normalized image 1 - Min: {image_g.min()}, Max: {image_g.max()}")
        #print(f"Not Normalized image 1 - Min: {image_nir.min()}, Max: {image_nir.max()}")
        #plt.imshow(image_r)
        #plt.show()


        # Funzione per normalizzare le immagini
        def normalize_image_mean_std(image_array, mean, std):
            normalized = (image_array - mean) / std
            return normalized
        

         # Applica la normalizzazione usando i valori di mean e std calcolati sul dataset
        image_array_r = normalize_image_mean_std(image_r, self.rmean, self.rstd)
        image_array_g = normalize_image_mean_std(image_g, self.gmean, self.gstd)
        image_array_nir = normalize_image_mean_std(image_nir, self.nirmean, self.nirstd)

        # Applica le trasformazioni (solo se il set è di addestramento)
        if self.train and self.transforms:
            image_array_r = self.transforms(image_array_r)
            image_array_g = self.transforms(image_array_g)
            image_array_nir = self.transforms(image_array_nir)
        
        

        #converti in tensore pytorch32 e aggiungo una dim
        image_array_r1 = torch.tensor(image_array_r, dtype=torch.float32).unsqueeze(0)
        image_array_g1 = torch.tensor(image_array_g, dtype=torch.float32).unsqueeze(0)
        image_array_nir1 = torch.tensor(image_array_nir, dtype=torch.float32).unsqueeze(0)

        #print("Image shape:", image_array_r1.shape)
        #print(f" Normalized image 2 - Min: {image_array_r1.min()}, Max: {image_array_r1.max()}")
        #print(f" Normalized image 2 - Min: {image_array_g1.min()}, Max: {image_array_g1.max()}")
        #print(f" Normalized image 2 - Min: {image_array_nir1.min()}, Max: {image_array_nir1.max()}")
        #plt.imshow(image_array_r1.squeeze(0))
        #plt.show()
        

        # Restituisce un dizionario con le immagini
        sample = {
            'image_r': image_array_r1, 
            'image_g': image_array_g1, 
            'image_nir': image_array_nir1,
            'image_name': prefix
        }
        return sample
