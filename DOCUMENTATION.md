# Documentazione tecnica — Generazione NIR e NDVI

## 1. Obiettivo

Generare la banda **NIR** da immagini RGB e calcolare l'**NDVI**, confrontando:

* baseline Swin2MoSE
* SAM + LoRA
* SAM + LoRA con loss avanzata

---

## 2. Dataset

Triplette:

```
*_R.TIF
*_G.TIF
*_NIR.TIF
```

Loader:

```
sam_nir/dataset_sam_nir.py
```

Output:

```
image_sam = [R, G, G]
image_r
image_g
image_nir
image_name
```

---

## 3. NDVI

Formula:

```
NDVI = (NIR - R) / (NIR + R)
```

---

## 4. Pipeline SAM

Input:

```
[R, G, G]
```

Pipeline:

```
SAM encoder + LoRA → decoder → NIR predetta
```

---

## 5. Problema iniziale

Effetti osservati:

* smoothing eccessivo
* perdita dettagli
* NDVI poco variabile
* SSIM NDVI peggiorata

Causa:

```
loss pixel-wise → soluzione media
```

---

## 6. Loss finale

```
Loss totale =
    MSE
  + L1
  + Edge
  + NDVI
  + Gradient
```

Effetti:

* preserva bordi
* migliora NDVI
* riduce smoothing

---

## 7. Risultati globali

```json
{
  "sam": {
    "psnr": 24.10,
    "ssim": 0.466,
    "psnr_ndvi": 25.47,
    "ssim_ndvi": 0.690
  }
}
```

---

## 8. Risultati ROI

ROI:

```
NDVI_true > 0.3
```

```json
{
  "sam_roi": {
    "psnr": 22.88,
    "ssim": 0.919,
    "psnr_ndvi": 25.80,
    "ssim_ndvi": 0.916
  }
}
```

---

## 9. Conclusione tecnica

Il limite principale era la loss.

L'aggiunta di:

* NDVI loss
* Gradient loss

ha portato a:

* NDVI più realistico
* maggiore variabilità
* miglioramento SSIM NDVI

---

## 10. Stato finale

✔ modello stabile
✔ metriche migliorate
✔ problema smoothing risolto
✔ pronto per valutazione accademica
