"""Input definitions for the benign/malignant experiments.

The checkpoint ``checkpoints/resnet/best_resnet.pth`` was trained as
``roi_masked``: a ground-truth box crop with pixels outside the mask set to
zero, then resized to 256×256, ImageNet-normalized, 3 channels. The other
modes below are implemented so they can be trained later on the same split.
They have not been trained, and no architecture is selected from them.

Label source for every mode is the BUSI folder name. This repository does not
contain a per-image histopathology table.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset.busi_dataset import LABEL_TO_IDX
from preprocessing.mask_to_yolo import component_mask
from preprocessing.normalization import imagenet_normalize, load_image_rgb, to_unit_range
from preprocessing.roi_utils import apply_mask, crop_roi_pair

InputMode = Literal["full_image", "roi", "roi_masked", "roi_plus_mask"]

EXPERIMENTS = {
    "full_image": {
        "description": "Entire ultrasound frame resized to the ROI size. No box and no mask.",
        "channels": 3,
        "trained": False,
    },
    "roi": {
        "description": "Tumor ROI from the ground-truth box, with surrounding pad kept and the mask not applied.",
        "channels": 3,
        "trained": False,
    },
    "roi_masked": {
        "description": "Current ResNet50 baseline. ROI with non-lesion pixels zeroed.",
        "channels": 3,
        "trained": True,
        "checkpoint": "checkpoints/resnet/best_resnet.pth",
    },
    "roi_plus_mask": {
        "description": "ROI image plus the mask as a fourth channel. Image and mask are resized together.",
        "channels": 4,
        "trained": False,
    },
}


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    pil = Image.fromarray(image)
    return np.asarray(pil.resize((width, height), Image.Resampling.BILINEAR))


class ClassificationExperimentDataset(Dataset):
    """One lesion row -> a tensor for a named input mode. No augmentation here.

    Augmentation, class weights, and the optimizer belong in a training script
    that has not been executed for the untrained modes.
    """

    def __init__(
        self,
        lesion_rows: list[dict],
        mode: InputMode = "roi_masked",
        roi_size: tuple[int, int] = (256, 256),
    ) -> None:
        if mode not in EXPERIMENTS:
            raise ValueError(f"mode must be one of {tuple(EXPERIMENTS)}")
        self.rows = lesion_rows
        self.mode = mode
        self.roi_size = roi_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = load_image_rgb(row["image_path"])
        if self.mode == "full_image":
            resized = _resize(image, self.roi_size)
            tensor = _rgb_tensor(resized)
        else:
            mask = component_mask(row["source_mask_path"], row["component_label"])
            image_crop, mask_crop, _ = crop_roi_pair(
                image, mask, tuple(row["bbox"]), pad_ratio=0.0, target_size=self.roi_size
            )
            if self.mode == "roi":
                tensor = _rgb_tensor(image_crop)
            elif self.mode == "roi_masked":
                tensor = _rgb_tensor(apply_mask(image_crop, mask_crop))
            else:
                rgb = to_unit_range(image_crop)
                mask_float = (mask_crop > 0).astype(np.float32)[..., None]
                stacked = np.concatenate([imagenet_normalize(rgb), mask_float], axis=2)
                tensor = torch.from_numpy(stacked.transpose(2, 0, 1).copy()).float()
        return {
            "image": tensor,
            "label": torch.tensor(LABEL_TO_IDX[row["label"]], dtype=torch.long),
            "image_id": row["image_id"],
            "mode": self.mode,
        }


def _rgb_tensor(image_uint8: np.ndarray) -> torch.Tensor:
    normalized = imagenet_normalize(to_unit_range(image_uint8))
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy()).float()
