# Dataset audit

Generated: 2026-11-08T11:57:32.337851+00:00

Research use only. This report describes files on disk. It does not add labels.

## What the files contain

- Images: **647** (benign 437, malignant 210).
- Formats: {'.png': 647}. Pixel modes: {'RGB': 647}. Channels: [3].
- Size range: width 190–1048, height 324–719 (545 unique sizes).
- Unreadable images: 0.
- Lesion masks: 665. Images without a mask: none.
- Masks per image: {'1': 630, '2': 16, '3': 1}.
- No pixel spacing, patient identifier, examination identifier, or `normal` class is stored.

## Labels that are not in this copy

No CSV/JSON/TXT annotation table of BI-RADS descriptors or assessment categories was found. The PDFs are terminology references, not image labels. Folder names supply benign/malignant only.

Reference PDFs found: dataset/BIRADS Reference Card 1.pdf, dataset/usg-16034 (1).pdf, dataset/BUSI_Jpeg/BIRADS Reference Card (1).pdf.

Classification labels are the folder names `benign` and `malignant`. No histopathology table is in the repository. Those folder labels must not be converted into BI-RADS categories or descriptors.

## Duplicates and split leakage

- Exact duplicate groups: [['benign (433)', 'malignant (145)']].
- Near-duplicate pairs (dHash Hamming 1–5): **128**.
- Of those, cross-split: **63**; cross-label: **5**.

Existing files in dataset/splits are stratified by folder label and group byte-identical images. They are not patient-level. Near-duplicates already cross the frozen split, so published checkpoint metrics are not a clean patient-level estimate. The split was not rebuilt, because rebuilding it would make the saved YOLO, U-Net, and ResNet50 checkpoints incomparable.

## YOLO label files

- `train`: 452 images, 461 boxes, class ids {'0': 461}, invalid boxes 0, images missing labels 0.
- `val`: 98 images, 104 boxes, class ids {'0': 104}, invalid boxes 0, images missing labels 0.
- `test`: 97 images, 100 boxes, class ids {'0': 100}, invalid boxes 0, images missing labels 0.

## Consequence for later stages

- Detection, segmentation, and benign/malignant classification can be trained from these files.
- A multi-task BI-RADS descriptor model cannot be trained. There is no expert descriptor annotation.
- A BI-RADS assessment model cannot be trained. Categories 0–6 are not image labels here.
- Patient-level metrics cannot be computed until a patient identifier exists.
