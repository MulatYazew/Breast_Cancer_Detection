# Breast ultrasound CAD — logged results and limits

The October 2026 training run produced checkpoints. The test numbers below are copied from `logs/*_evaluate.log` (timestamps 2026-10-07). They were not recomputed on 2026-11-08. A machine-readable copy is `reports/logged_baseline_metrics.json`.

The split is image-level. It is not patient-level. The 2026-11-08 audit found 63 near-duplicate pairs that cross this split. Read the metrics with that limit.

## Dataset

- 647 images: 437 benign, 210 malignant. 665 masks.
- Split sizes: train 452, val 98, test 97.
- One exact cross-label duplicate: benign (433) and malignant (145), kept in one split.
- No BI-RADS descriptor or assessment labels.

## Logged test metrics (2026-10-07)

### Detection, YOLO26n

| Metric | Value |
| --- | --- |
| Precision | 0.884 |
| Recall | 0.770 |
| F1 | 0.823 |
| mAP@0.50 | 0.734 |
| mAP@0.50:0.95 | 0.545 |

Best validation epoch in the training CSV was epoch 86 (mAP@0.50 0.907). That is not the test number.

### Segmentation, U-Net ResNet34

| Dice | IoU | Precision | Recall | Specificity |
| --- | --- | --- | --- | --- |
| 0.936 | 0.882 | 0.927 | 0.949 | 0.918 |

HD95 and ASSD were not computed. Other U-Net variants in `configs/` were not the checkpoint that was evaluated.

### Classification, ResNet50 masked ROI

| Accuracy | Precision | Malignant recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| 0.93 | 0.962 | 0.806 | 0.877 | 0.984 | 0.968 |

### Cascade, predicted box and mask

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Missed detections | Mean detection IoU | Mean Dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.830 | 0.688 | 0.815 | 0.746 | 0.892 | 0.752 | 9 | 0.670 | 0.807 |

## What these numbers are not

They are not a BI-RADS result, not a patient-level result, and not an external validation. Descriptor and assessment models were not trained. See `ENGINEERING_REPORT.md`.
