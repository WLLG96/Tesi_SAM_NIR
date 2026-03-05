<<<<<<< HEAD
# Predizione della banda NIR da immagini RGB e calcolo NDVI

## Descrizione del progetto

Questo progetto ha l'obiettivo di **stimare la banda Near Infrared (NIR)** a partire da immagini RGB utilizzando una rete neurale basata su **Swin Transformer (Swin2MoSE)**.

Una volta stimata la banda NIR, viene calcolato l'indice di vegetazione **NDVI (Normalized Difference Vegetation Index)**, utilizzato per analizzare lo stato della vegetazione.

Il sistema include inoltre **Segment Anything (SAM)** per permettere all’utente di selezionare una regione dell'immagine (ad esempio un campo agricolo) e visualizzare l’NDVI solo nell’area selezionata.

---

# Pipeline del sistema

Il sistema segue la seguente pipeline:

1. Acquisizione di immagini RGB da drone
2. Predizione della banda **NIR** tramite rete neurale
3. Calcolo dell'indice **NDVI**

Formula utilizzata:

```
NDVI = (NIR - R) / (NIR + R)
```

4. Segmentazione della regione di interesse tramite **Segment Anything**
5. Calcolo delle statistiche NDVI nella regione selezionata

---

# Struttura del progetto

```
swin2nir/

configs/
    config_linda.yaml

data/
    dataset_cropped.py

train/
    train.py
    validate.py

model.py
utils.py
main_nvdi.py

checkpoint_model/
results/
validation_results/
test_results/
logs/
```

---

# Dataset

Il dataset è composto da immagini multispettrali TIF contenenti tre bande:

* banda **R**
* banda **G**
* banda **NIR**

Ogni campione è composto da tre file:

```
prefix_R.TIF
prefix_G.TIF
prefix_NIR.TIF
```

Il dataset loader:

* carica le tre bande
* applica crop coerente
* normalizza i valori delle immagini
* restituisce tensori PyTorch.

Input del modello:

```
[R, G]
```

Output del modello:

```
NIR
```

---

# File di configurazione

Il file di configurazione principale è:

```
configs/config_linda.yaml
```

Contiene:

* percorsi dataset
* parametri del modello
* parametri di training
* directory per risultati e checkpoint

---

# Training del modello

Per avviare il training utilizzare il comando:

```bash
python main_nvdi.py --function train --config configs/config_linda.yaml
```

Durante il training:

* il dataset viene caricato tramite DataLoader
* il modello Swin2MoSE viene inizializzato
* viene utilizzata la **loss MSE**
* i checkpoint vengono salvati automaticamente

I checkpoint vengono salvati nella cartella:

```
checkpoint_model/
```

Esempio:

```
ckpt_epoch_010.pth
```

---

# Riprendere il training

È possibile riprendere il training da un checkpoint:

```bash
python main_nvdi.py \
--function train \
--config configs/config_linda.yaml \
--resume checkpoint_model/ckpt_epoch_005.pth
```

---

# Validazione del modello

Per eseguire la validazione:

```bash
python main_nvdi.py --function validate --config configs/config_linda.yaml
```

Durante la validazione vengono calcolate le seguenti metriche:

* PSNR
* SSIM
* PSNR su NDVI
* SSIM su NDVI

I risultati vengono salvati nella cartella:

```
results/
```

Esempio:

```
val_metrics_epoch_010.txt
val_metrics_epoch_010.json
val_metrics.csv
```

---

# Test del modello

Per testare un modello addestrato:

```bash
python main_nvdi.py \
--function test \
--config configs/config_linda.yaml \
--ckpt checkpoint_model/ckpt_epoch_010.pth
```

---

# Visualizzazione dei risultati

Durante la validazione vengono generate immagini di confronto:

```
R | G | NIR predetto | NIR reale | NDVI predetto | NDVI reale
```

Salvate nella cartella:

```
validation_results/
```

---

# Architettura del modello

Il modello utilizzato è basato su **Swin Transformer**.

Caratteristiche principali:

* self-attention su finestre locali
* architettura gerarchica
* residual connections
* patch embedding

Input del modello:

```
R + G
```

Output del modello:

```
NIR predetto
```

---

# Metriche utilizzate

### PSNR

Misura la qualità della ricostruzione dell'immagine.

### SSIM

Misura la similarità strutturale tra immagini.

### NDVI

Confronto tra NDVI reale e NDVI calcolato usando la NIR predetta.

---

# Dipendenze principali

Il progetto utilizza:

```
python
pytorch
torchvision
numpy
opencv
tqdm
scikit-image
torchmetrics
gradio
```

Installazione:

```bash
pip install -r requirements.txt
```

---

# Autore

Progetto di Tesi Magistrale
Predizione della banda NIR da immagini RGB e calcolo NDVI per monitoraggio della vegetazione.
=======
# Segmentazione e Predizione della Banda NIR con SAM e LoRA

---

## Obiettivo

Immagine RGB → Encoder di SAM → Rete neurale → Banda NIR (output)

La banda NIR è utile per il calcolo di indici di vegetazione come l’**NDVI (Normalized Difference Vegetation Index)**,
che permette di stimare lo stato di salute delle colture.

---

##  Struttura del progetto

tesi/
 ├── code/
 │   └── Sam_LoRA/
 │       ├── train_nir_head.py
 │       ├── infer_and_ndvi.py
 │       ├── sam_click_ndvi.py
 │       ├── nir_dataset.py
 │       ├── sam_encoder_features.py
 │       ├── segment_anything/
 │       ├── runs_nir_head/
 │       └── runs_demo_click/
 └── data/
     ├── rgb2nir/
     └── field_with_road.png

---

## Esecuzione

### Training
cd ~/tesi/code/Sam_LoRA
python train_nir_head.py

### Inferenza su dataset
python infer_and_ndvi.py

### Demo interattiva su immagine RGB
python sam_click_ndvi.py ~/tesi/data/field_with_road.png

---

## Esempio di output

Cartella: `runs_demo_click/`

- *_rgb.png → immagine originale
- *_sam_mask.png → maschera SAM
- *_overlay.png → maschera + immagine
- *_nir_pred.png → banda NIR predetta
- *_ndvi_jet.png → mappa NDVI colorata
- *_ndvi_stats.txt → valori numerici NDVI

---
>>>>>>> 7790f84c47b097fc538956052817f84914359d85
