"""Stage 3 dataloader construction: wires the shared ``SegmentationROIDataset``
and augmentation builder to config, so ``unet/train.py`` and
``unet/evaluate.py`` don't each re-derive dataset/augmentation setup.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from dataset.busi_dataset import SegmentationROIDataset
from dataset.lesion_manifest import load_lesion_manifest
from dataset.splits import apply_quick_run_subsample
from preprocessing.augmentation import build_unet_augmentations


def build_unet_dataloader(
    config: dict,
    split: str,
    shuffle: bool | None = None,
    num_workers: int = 4,
) -> DataLoader:
    """Build a train/val/test DataLoader for Stage 3 from the saved lesion manifests."""
    paths = config["paths"]
    prep_cfg = config["preprocessing"]
    unet_cfg = config["unet"]
    aug_cfg = config["augmentation"]

    rows = load_lesion_manifest(f"{paths['splits_dir']}/{split}_lesions.json")
    rows = apply_quick_run_subsample(rows, config)
    roi_size = tuple(prep_cfg["roi_size"])

    transform = (
        build_unet_augmentations(
            roi_size,
            safe_prob=aug_cfg["safe_prob"],
            cautious_prob=aug_cfg["cautious_prob"],
            use_cautious_set=aug_cfg["unet"]["use_cautious_set"],
            use_elastic=aug_cfg["unet"]["use_elastic"],
            elastic_alpha=aug_cfg["unet"]["elastic_alpha"],
            elastic_sigma=aug_cfg["unet"]["elastic_sigma"],
        )
        if split == "train"
        else None
    )

    dataset = SegmentationROIDataset(rows, roi_size=roi_size, transform=transform)

    return DataLoader(
        dataset,
        batch_size=unet_cfg["batch_size"],
        shuffle=shuffle if shuffle is not None else (split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
