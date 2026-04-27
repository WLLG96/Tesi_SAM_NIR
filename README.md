# Tesi Magistrale — Generazione NIR e NDVI da immagini RGB con Swin2MoSE e SAM + LoRA

## Obiettivo

Il progetto studia la generazione della banda **NIR** a partire da immagini RGB e il calcolo dell'**NDVI** per applicazioni di agricoltura di precisione.

Sono confrontati:

1. **Baseline Swin2MoSE**
2. **SAM + LoRA (RGB → NIR)**
3. **SAM + LoRA con loss migliorata (Edge + NDVI + Gradient)**

---

## Problema iniziale

Il modello SAM iniziale produceva:

* mappe **troppo smooth**
* perdita di dettagli locali
* distribuzione NDVI poco realistica
* **SSIM NDVI peggiore della baseline**

👉 causa principale: loss solo pixel-wise (MSE + L1)

---

## Soluzione proposta

Introduzione di una loss combinata:

* MSE loss
* L1 loss
* Edge loss (preserva struttura spaziale)
* NDVI loss (allinea al target agronomico)
* Gradient loss (preserva variazioni locali)

👉 risultato: riduzione dello smoothing e NDVI più realistico

---

## Risultati finali

### Benchmark globale

| Modello         |  PSNR NIR | SSIM NIR | PSNR NDVI | SSIM NDVI |
| --------------- | --------: | -------: | --------: | --------: |
| Baseline        |     17.88 |    0.442 |     23.34 |     0.667 |
| SAM (prima)     |     23.65 |    0.490 |     24.96 |     0.662 |
| SAM + NDVI/Grad | **24.10** |    0.467 | **25.47** | **0.691** |

---

### Benchmark ROI vegetata

ROI definita come:

```
NDVI_true > 0.3
```

| Modello         |  PSNR NIR |  SSIM NIR | PSNR NDVI | SSIM NDVI |
| --------------- | --------: | --------: | --------: | --------: |
| Baseline        |     22.13 |     0.427 |     24.88 |     0.688 |
| SAM + NDVI/Grad | **22.88** | **0.919** | **25.80** | **0.916** |

---

## Conclusione

Il problema non era l'encoder SAM, ma la funzione di loss.

La nuova loss:

* riduce il bias verso soluzioni smooth
* migliora la distribuzione NDVI
* aumenta la qualità nelle regioni vegetate

👉 problema della SSIM NDVI **risolto**

---

## Struttura progetto

```
sam_nir/
 ├── train_sam_nir.py
 ├── infer_sam_nir.py
 ├── compare_sam_ndvi_grad_vs_baseline.py
 ├── sam_encoder_model.py
 ├── dataset_sam_nir.py
 ├── comparison_results_ndvi_grad.json
 ├── comparison_results_roi_ndvi_grad.json
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision numpy matplotlib opencv-python pillow tqdm scikit-image timm safetensors
```

---

## Training

```bash
python sam_nir/train_sam_nir.py
```

Checkpoint finale:

```
sam_nir/checkpoints_r8_mse_l1_edge_ndvi_grad/sam_nir_epoch_002.pth
```

---

## Inferenza

```bash
python sam_nir/infer_sam_nir.py
```

---

## Confronto metriche

```bash
python sam_nir/compare_sam_ndvi_grad_vs_baseline.py
```

---

## Output principali

```
sam_nir/ndvi_epoch2_r8_mse_l1_edge_ndvi_grad/
sam_nir/comparison_results_ndvi_grad.json
sam_nir/comparison_results_roi_ndvi_grad.json
```

---

## Stato finale

✔ modello funzionante
✔ metriche migliorate
✔ problema smoothing risolto
✔ pronto per presentazione tesi


