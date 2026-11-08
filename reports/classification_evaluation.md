# Classification evaluation

Research classification of folder labels. Not a clinical diagnosis and not a BI-RADS assessment.

Device: `mps`.

The selected model is the one with the highest validation ROC-AUC. Test numbers are reported after that choice.

Selected backbone: **densenet121**.
Validation threshold frozen for test: 0.65.

## Validation

| Model | ROC-AUC | Malignant sensitivity | Benign specificity | Epochs | Peak RSS GiB |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b0 | 0.948 | 0.828 | 0.941 | 5 | 1.05 |
| densenet121 | 0.973 | 0.862 | 0.985 | 6 | 1.28 |
| mobilenet_v3_small | 0.914 | 1.000 | 0.735 | 6 | 1.56 |
| resnet18 | 0.944 | 0.931 | 0.882 | 6 | 1.65 |

## Selected model on the untouched test split

- Malignant sensitivity: 0.781
- Benign specificity: 0.983
- Precision (malignant): 0.962
- F1 (malignant): 0.862
- Macro-F1: 0.899
- ROC-AUC: 0.9359110169491526
- PR-AUC: 0.9194753802424545
- Brier score: 0.086
- Confusion matrix (rows true benign/malignant): [[58, 1], [7, 25]]

Patient identifiers are not in this dataset. Groups are exact duplicates and near-duplicates. Conflict groups held out: 5.

These figures describe folder labels on this split. They are not a BI-RADS assessment and not a clinical claim.
