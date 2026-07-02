"""Stage 4 dataloader construction: wires the shared ``ClassificationROIDataset``
and augmentation builder to config, mirroring ``unet/dataset.py``.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from dataset.busi_dataset import ClassificationROIDataset
from dataset.lesion_manifest import load_lesion_manifest
from dataset.splits import apply_quick_run_subsample
from preprocessing.augmentation import build_resnet_augmentations


def build_resnet_dataloader(
    config: dict,
    split: str,
    shuffle: bool | None = None,
    num_workers: int = 4,
) -> DataLoader:
    """Build a train/val/test DataLoader for Stage 4 from the saved lesion manifests."""
    paths = config["paths"]
    prep_cfg = config["preprocessing"]
    resnet_cfg = config["resnet"]
    aug_cfg = config["augmentation"]

    rows = load_lesion_manifest(f"{paths['splits_dir']}/{split}_lesions.json")
    rows = apply_quick_run_subsample(rows, config)
    roi_size = tuple(prep_cfg["roi_size"])

    transform = (
        build_resnet_augmentations(roi_size, safe_prob=aug_cfg["safe_prob"]) if split == "train" else None
    )

    dataset = ClassificationROIDataset(rows, roi_size=roi_size, transform=transform, use_mask=True)

    return DataLoader(
        dataset,
        batch_size=resnet_cfg["batch_size"],
        shuffle=shuffle if shuffle is not None else (split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
