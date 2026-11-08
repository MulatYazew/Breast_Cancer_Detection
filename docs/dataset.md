# Dataset

The images used by training are the BUSI JPEG tree named in `config.yaml`:

- `dataset/BUSI_Jpeg/benign` and `benign_mask`
- `dataset/BUSI_Jpeg/malignant` and `malignant_mask`

Audit on 2026-11-08 (`reports/dataset_audit.md`):

- 647 PNG images, all RGB, 8-bit. 437 benign, 210 malignant. No `normal` folder in this copy.
- 665 masks. 630 images have one lesion mask, 16 have two, 1 has three.
- Width 190–1048 px, height 324–719 px, 545 distinct sizes. No pixel spacing.
- 0 unreadable images.
- Exact duplicate: `benign (433)` and `malignant (145)`.
- 128 near-duplicate pairs (dHash Hamming 1–5). 63 cross the saved split. 5 cross the folder label.
- No patient id, examination id, BI-RADS descriptor table, or BI-RADS assessment table.

The classification target in the saved ResNet50 is the folder name. A histopathology table is not in this repository.

Splits in `dataset/splits/` are 452 / 98 / 97 images (train / val / test), stratified by folder and grouped when the file bytes match. They are not patient-level. YOLO labels derived from the masks use one class, `tumor`, and the 2026-11-08 check found no box outside the normalized image.

Reference documents, which are not labels:

- `dataset/BIRADS Reference Card 1.pdf` — ACR BI-RADS Atlas, fifth edition, quick reference, dated 07.15.
- `dataset/usg-16034 (1).pdf` — Lee J. Practical and illustrated summary of updated BI-RADS for ultrasonography. Ultrasonography. 2017;36(1):71–81.

Regenerate the audit with:

```bash
python -m dataset.inspect
```
