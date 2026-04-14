# Documentazione codice — Generazione NIR e NDVI

## 1. Obiettivo del progetto

Il progetto ha come obiettivo la generazione della banda NIR a partire da immagini RGB e il calcolo dell’NDVI, con confronto tra:
- modello baseline (Swin2MoSE)
- modello basato su SAM

---

## 2. Dataset

Il dataset è composto da triplette:
- R (Red)
- G (Green)
- NIR (target)

Pipeline:
- crop coerente tra bande
- normalizzazione min-max → [0,1]
- opzionale mean/std

File principale:
- `data/dataset_cropped.py`

---

## 3. Pipeline baseline

### Input
- [R, G]

### Output
- NIR

### Modello
- Swin2MoSE

### File principali
- `model.py`
- `train/train.py`
- `train/validate.py`

---

## 4. Pipeline SAM

### Input
- [R, G, G]

### Architettura
- encoder: SAM pretrained (ViT-B)
- LoRA applicato
- decoder: convoluzionale

### File principali
- `sam_nir/sam_encoder_model.py`
- `sam_nir/train_sam_nir.py`

---

## 5. Calcolo NDVI

Formula:

NDVI = (NIR - R) / (NIR + R)

Usato per:
- valutazione
- benchmark
- demo

---

## 6. Benchmark

### 6.1 Benchmark globale
Metriche su tutta l’immagine:
- PSNR NIR
- SSIM NIR
- PSNR NDVI
- SSIM NDVI

Script:
- `sam_nir/compare_sam_vs_baseline.py`

---

### 6.2 Benchmark ROI
Metriche su regioni selezionate:

ROI definita come:
- NDVI reale > 0.3

Script:
- `sam_nir/compare_sam_vs_baseline_roi.py`

---

## 7. Risultati principali

### Benchmark globale
SAM migliora:
- PSNR NIR
- PSNR NDVI

Baseline migliore su:
- SSIM NDVI

### Benchmark ROI
Comportamento coerente:
- SAM migliore su NIR
- baseline leggermente migliore su SSIM NDVI

---

## 8. Demo interattiva

Pipeline:
1. input RGB
2. click utente
3. segmentazione SAM
4. predizione NIR
5. calcolo NDVI
6. overlay su ROI
7. statistiche

File:
- `sam_nir/demo_click_sam_ndvi.py`

---

## 9. Struttura progetto

- data/ → dataset
- train/ → training baseline
- sam_nir/ → modelli SAM + benchmark
- Sam_LoRA/ → implementazione LoRA
- configs/ → configurazioni

---

## 10. Stato attuale

✔ baseline funzionante  
✔ modello SAM funzionante  
✔ benchmark globale  
✔ benchmark ROI  
✔ demo interattiva  

---

## 11. Possibili sviluppi

- fine-tuning SAM su segmentazione agricola
- ablation study (LoRA rank, decoder)
- miglioramento SSIM NDVI