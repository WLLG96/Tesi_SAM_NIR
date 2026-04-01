# Tesi Magistrale — Generazione banda NIR e NDVI da immagini RGB con baseline Swin2MoSE e integrazione SAM

---

## Descrizione del progetto

Questo progetto affronta il problema della **stima della banda Near Infrared (NIR)** a partire da immagini RGB/multibanda e del calcolo dell’**NDVI (Normalized Difference Vegetation Index)**.

Il lavoro è strutturato in due parti principali:

1. **Baseline RGB → NIR → NDVI**
   Modello basato su **Swin2MoSE**

2. **Integrazione SAM (Segment Anything Model)**
   Uso dell’**image encoder di SAM** per generare la banda NIR + demo interattiva con click utente

---

## Obiettivo finale

Costruire una pipeline che permetta:

1. Input immagine RGB
2. Click su una regione di interesse
3. Segmentazione con SAM
4. Predizione della banda NIR
5. Calcolo NDVI
6. Visualizzazione NDVI sulla regione
7. Calcolo statistiche locali

---

# Struttura del progetto

```
TESI_SAM_NIR/

configs/
data/
train/
test_/

Sam_LoRA/
sam_nir/

model.py
main_nvdi.py
utils.py
```

---

# Dataset

Il dataset è composto da triplette:

```
*_R.TIF
*_G.TIF
*_NIR.TIF
```

Il loader:

* carica le bande
* applica crop coerente
* normalizza
* restituisce tensori PyTorch

---

# Formula NDVI

```
NDVI = (NIR - R) / (NIR + R)
```

---

# PARTE 1 — BASELINE (Swin2MoSE)

## Input / Output

```
Input:  R + G
Output: NIR
```

---

## Training baseline

```bash
python main_nvdi.py --function train --config configs/config_linda.yaml --epochs 3
```

---

## Validazione baseline

```bash
PYTHONPATH=. python main_nvdi.py \
--function validate \
--config configs/config_linda.yaml \
--ckpt ./checkpoint_model/ckpt_epoch_001.pth --image resize

PYTHONPATH=. python main_nvdi.py \
--function validate \
--config configs/config_linda.yaml \
--ckpt ./checkpoint_model/ckpt_epoch_002.pth --image resize

PYTHONPATH=. python main_nvdi.py \
--function validate \
--config configs/config_linda.yaml \
--ckpt ./checkpoint_model/ckpt_epoch_003.pth --image resize
```

---

## Test baseline

```bash
PYTHONPATH=. python main_nvdi.py \
--function test \
--config configs/config_linda.yaml \
--ckpt ./checkpoint_model/ckpt_epoch_002.pth --image resize
```

---

## Metriche baseline

* PSNR
* SSIM
* PSNR NDVI
* SSIM NDVI

---

# PARTE 2 — INTEGRAZIONE SAM

## Idea

Usare l’encoder di SAM come backbone:

```
[R, G, G] → SAM encoder → decoder → NIR
```

---

## Setup SAM

```bash
mkdir -p checkpoints
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
-o checkpoints/sam_vit_b_01ec64.pth
```

---

## Test encoder SAM

```bash
python sam_nir/smoke_test_sam_nir.py
```

Output atteso:

```
output shape: torch.Size([1, 1, 1024, 1024])
```

---

## Training SAM → NIR

```bash
python sam_nir/train_sam_nir.py
```

Output:

```
sam_nir/checkpoints/
```

---

## Inferenza NIR (SAM)

```bash
python sam_nir/infer_sam_nir.py
```

Output:

```
sam_nir/predictions/
```

---

## Inferenza NDVI (SAM)

```bash
python sam_nir/infer_sam_ndvi.py
```

Output:

```
sam_nir/ndvi_predictions/
```

---

# PARTE 3 — CONFRONTO BASELINE vs SAM

```bash
PYTHONPATH=. python sam_nir/compare_sam_vs_baseline.py
```

---

## Risultati ottenuti

### Baseline

* PSNR: 17.88
* SSIM: 0.44
* PSNR NDVI: 23.34
* SSIM NDVI: 0.67

### SAM

* PSNR: 23.68
* SSIM: 0.48
* PSNR NDVI: 24.83
* SSIM NDVI: 0.64

---

## Interpretazione

* SAM migliora la qualità della NIR
* migliora PSNR NDVI
* output più smooth → leggero calo SSIM NDVI

---

# PARTE 4 — DEMO INTERATTIVA

## Esecuzione

```bash
PYTHONPATH=. python sam_nir/demo_click_sam_ndvi.py \
--image /Users/.../immagine.png
```

---

## Pipeline

1. input immagine RGB
2. click utente
3. segmentazione SAM
4. predizione NIR
5. calcolo NDVI
6. overlay NDVI
7. statistiche locali

---

## Output demo

```
sam_nir/demo_outputs/
```

Contiene:

* maschera SAM
* NIR predetta
* NDVI
* overlay
* report
* statistiche JSON

---

# Dipendenze

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy matplotlib opencv-python pillow tqdm pyyaml timm safetensors
```

---

# Limitazioni

* input SAM adattato `[R, G, G]`
* SAM non fine-tuned su campi coltivati
* output SAM più smooth


