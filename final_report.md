# Breast Ultrasound CAD System — Final Report

> Template. Fill in each `[TODO]` after running real training (`python train.py all`)
> and evaluation (`python evaluate.py ...`) — no results exist yet since training has
> not been executed.

## 1. Overview

A cascaded CAD pipeline for the BUSI breast ultrasound dataset: YOLO26n (tumor detection)
-> U-Net (segmentation) -> ResNet50 (benign/malignant classification), with a Streamlit
clinician-in-the-loop demo for physician-reviewed inference.

## 2. Dataset

- Source: BUSI (Breast Ultrasound Images), `dataset/BUSI_Jpeg/{benign,malignant}` + masks.
- Classes: benign, malignant (no `normal` class in this copy).
- Total images / lesion instances: [TODO — see `results/preprocessing/report.json`].
- Split: stratified 70/15/15 (train/val/test), fixed seed, content-duplicate-aware
  (see `preprocessing/validation.py`'s `duplicate_image_cross_class` check).
- Data quality notes: [TODO — summarize `results/preprocessing/report.json` validation section].

## 3. Methodology

### 3.1 Preprocessing & Augmentation
Mask validation, stratified split, mask -> YOLO bounding-box conversion (connected-component
analysis, padding, clipping), stage-aware Albumentations pipelines (see
`preprocessing/augmentation.py` for the full safe/cautious/avoid transform rationale).

### 3.2 Stage 2 — YOLO26n Detection
Architecture: YOLO26n (Ultralytics). Training config: [TODO — epochs/batch/lr from
`config.yaml`'s `yolo:` section actually used].

### 3.3 Stage 3 — U-Net Segmentation
Architecture(s) compared: [TODO — plain U-Net/ResNet34, Attention U-Net/ResNet34,
U-Net/EfficientNet-B0, U-Net++/ResNet34 — whichever were run]. Loss: Dice + BCE.

### 3.4 Stage 4 — ResNet50 Classification
ImageNet-pretrained ResNet50, freeze-then-fine-tune schedule on the masked tumor ROI.

## 4. Results

### 4.1 Stage 2 — Detection
| Metric | Value |
|---|---|
| mAP50 | [TODO] |
| mAP50-95 | [TODO] |
| Precision | [TODO] |
| Recall | [TODO] |
| F1 | [TODO] |
| Mean IoU | [TODO] |

### 4.2 Stage 3 — Segmentation
| Architecture | Dice | IoU | Precision | Recall | Specificity |
|---|---|---|---|---|---|
| U-Net (ResNet34) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| Attention U-Net (ResNet34) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| U-Net (EfficientNet-B0) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

### 4.3 Stage 4 — Classification
| Metric | Value |
|---|---|
| Accuracy | [TODO] |
| Precision | [TODO] |
| Recall | [TODO] |
| F1 | [TODO] |
| ROC-AUC | [TODO] |
| PR-AUC | [TODO] |
| Cohen's kappa | [TODO] |
| MCC | [TODO] |

### 4.4 End-to-end cascade
Real YOLO -> U-Net -> ResNet50 chained (not ground-truth-fed), from
`evaluation/cascade.py`: [TODO — classification metrics + mean detection IoU + mean
segmentation Dice, quantifying error propagation vs. the isolated per-stage results above].

## 5. Discussion

[TODO — error analysis, failure modes, comparison across U-Net variants, limitations of
the BUSI dataset size/annotation quality (e.g. the cross-class duplicate image found
during validation).]

## 6. Conclusion

[TODO]

## Appendix: Reproducibility

- Seed: `config.yaml`'s `seed` (default 42), applied via `utils.seed.set_seed`.
- Config used for final results: [TODO — note any `configs/*.yaml` overrides applied].
- Hardware: [TODO — CPU/CUDA/MPS, approximate training time per stage].
