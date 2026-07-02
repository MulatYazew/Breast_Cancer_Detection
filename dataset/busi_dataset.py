"""PyTorch Dataset classes for Stage 3 (U-Net) and Stage 4 (ResNet50).

Both datasets are built from a lesion-instance manifest (see
``dataset.lesion_manifest``) and share the same crop/normalize utilities from
``preprocessing`` so ROI extraction behaves identically across training,
evaluation, and the Streamlit app.

Stage 2 (YOLO) does not use a torch ``Dataset`` here — Ultralytics consumes a
directory-based dataset built by ``yolo.dataset_builder``.
"""

from __future__ import annotations

from typing import Callable

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.mask_to_yolo import component_mask
from preprocessing.normalization import imagenet_normalize, load_image_rgb, to_unit_range
from preprocessing.roi_utils import apply_mask, crop_roi_pair

CLASS_NAMES = ["benign", "malignant"]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _to_chw_tensor(image_float01: np.ndarray, normalize: bool) -> torch.Tensor:
    if normalize:
        image_float01 = imagenet_normalize(image_float01)
    return torch.from_numpy(image_float01.transpose(2, 0, 1).copy()).float()


class SegmentationROIDataset(Dataset):
    """ROI crop -> binary lesion mask, for Stage 3 (U-Net) training/evaluation.

    Ground truth ROI boxes come from mask connected components (padded, see
    ``preprocessing.mask_to_yolo``), so ``pad_ratio=0`` is used at crop time
    here — the padding is already baked into the stored bbox.
    """

    def __init__(
        self,
        lesion_rows: list[dict],
        roi_size: tuple[int, int] = (256, 256),
        transform: A.Compose | Callable | None = None,
        normalize: bool = True,
    ) -> None:
        self.rows = lesion_rows
        self.roi_size = roi_size
        self.transform = transform
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = load_image_rgb(row["image_path"])
        mask = component_mask(row["source_mask_path"], row["component_label"])

        image_crop, mask_crop, _ = crop_roi_pair(
            image, mask, tuple(row["bbox"]), pad_ratio=0.0, target_size=self.roi_size
        )

        if self.transform is not None:
            augmented = self.transform(image=image_crop, mask=mask_crop)
            image_crop, mask_crop = augmented["image"], augmented["mask"]

        image_tensor = _to_chw_tensor(to_unit_range(image_crop), self.normalize)
        mask_tensor = torch.from_numpy((mask_crop > 0).astype(np.float32)).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": row["image_id"],
            "label": row["label"],
        }


class ClassificationROIDataset(Dataset):
    """Masked tumor ROI -> benign/malignant label, for Stage 4 (ResNet50).

    Trained independently of U-Net using ground-truth lesion masks (teacher
    forcing) — the chained inference pipeline (``inference/``) is what feeds
    U-Net's *predicted* mask forward at serving time.
    """

    def __init__(
        self,
        lesion_rows: list[dict],
        roi_size: tuple[int, int] = (256, 256),
        transform: A.Compose | Callable | None = None,
        normalize: bool = True,
        use_mask: bool = True,
    ) -> None:
        self.rows = lesion_rows
        self.roi_size = roi_size
        self.transform = transform
        self.normalize = normalize
        self.use_mask = use_mask

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = load_image_rgb(row["image_path"])
        mask = component_mask(row["source_mask_path"], row["component_label"])

        image_crop, mask_crop, _ = crop_roi_pair(
            image, mask, tuple(row["bbox"]), pad_ratio=0.0, target_size=self.roi_size
        )

        masked_crop = apply_mask(image_crop, mask_crop) if self.use_mask else image_crop

        if self.transform is not None:
            augmented = self.transform(image=masked_crop)
            masked_crop = augmented["image"]

        image_tensor = _to_chw_tensor(to_unit_range(masked_crop), self.normalize)
        label_idx = LABEL_TO_IDX[row["label"]]

        return {
            "image": image_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "image_id": row["image_id"],
        }
