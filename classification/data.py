"""Full-image classification data. Augmentation is applied to training only.

Images are letterboxed to a square so aspect ratio is kept. The folder name is
the label. This copy has benign and malignant only.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from classification.grouping import group_records
from dataset.manifest import build_manifest
from preprocessing.normalization import imagenet_normalize, load_image_rgb, to_unit_range

CLASS_NAMES = ["benign", "malignant"]
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Scale the long side to ``size`` and pad the short side with zeros."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    top = (size - resized_h) // 2
    left = (size - resized_w) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def _augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
    if rng.random() < 0.3:
        factor = rng.uniform(0.9, 1.1)
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        angle = rng.uniform(-10, 10)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
    return image


class UltrasoundImageDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int, train: bool, seed: int) -> None:
        self.rows = rows
        self.image_size = image_size
        self.train = train
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = load_image_rgb(row["image_path"])
        if self.train:
            image = _augment(image, random.Random(self.seed + self.epoch * 100_000 + index))
        image = letterbox(image, self.image_size)
        tensor = torch.from_numpy(imagenet_normalize(to_unit_range(image)).transpose(2, 0, 1).copy()).float()
        return {
            "image": tensor,
            "label": torch.tensor(LABEL_TO_INDEX[row["label"]], dtype=torch.long),
            "image_id": row["image_id"],
            "preview": image,
        }


def make_leakage_aware_split(config: dict, repo_root: Path, hamming_max: int = 5, seed: int = 42) -> dict:
    """Split whole near-duplicate groups. Conflict groups are excluded."""
    paths = config["paths"]
    root = Path(repo_root)
    records = build_manifest(
        root / paths["raw_benign"],
        root / paths["raw_benign_mask"],
        root / paths["raw_malignant"],
        root / paths["raw_malignant_mask"],
    )
    rows = [
        {"image_id": record.image_id, "label": record.label, "image_path": str(record.image_path)}
        for record in records
    ]
    grouped = group_records(rows, hamming_max=hamming_max)
    by_id = {row["image_id"]: row for row in rows}
    groups = grouped["groups"]
    labels = [group["label"] for group in groups]
    train_groups, temp_groups, _, temp_labels = train_test_split(
        groups, labels, train_size=0.70, random_state=seed, stratify=labels
    )
    val_groups, test_groups = train_test_split(
        temp_groups, train_size=0.5, random_state=seed, stratify=temp_labels
    )
    splits = {
        "train": [by_id[member] for group in train_groups for member in group["members"]],
        "val": [by_id[member] for group in val_groups for member in group["members"]],
        "test": [by_id[member] for group in test_groups for member in group["members"]],
    }
    for name in splits:
        splits[name] = sorted(splits[name], key=lambda row: (row["label"], row["image_id"]))
    split_of = {row["image_id"]: name for name, split_rows in splits.items() for row in split_rows}
    cross_split = []
    for pair in grouped["pairs"]:
        left, right = split_of.get(pair["image_a"]), split_of.get(pair["image_b"])
        if left and right and left != right:
            cross_split.append(pair)
    summary = {
        "seed": seed,
        "hamming_max": hamming_max,
        "patient_identifiers_available": False,
        "grouping": "exact file hash plus dHash Hamming <= %d; not a patient id" % hamming_max,
        "n_groups": len(groups),
        "n_conflict_groups": len(grouped["conflicts"]),
        "conflict_groups": grouped["conflicts"],
        "held_out_ids": sorted(member for group in grouped["conflicts"] for member in group["members"]),
        "cross_split_near_duplicate_pairs": len(cross_split),
        "split_counts": {
            name: {label: sum(row["label"] == label for row in rows_for_split) for label in CLASS_NAMES}
            for name, rows_for_split in splits.items()
        },
        "split_sizes": {name: len(rows_for_split) for name, rows_for_split in splits.items()},
    }
    out_dir = root / "dataset" / "splits_classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        (out_dir / f"{name}.json").write_text(json.dumps(split_rows, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    summary["splits"] = splits
    return summary


def load_split_rows(repo_root: Path, name: str) -> list[dict]:
    path = Path(repo_root) / "dataset" / "splits_classification" / f"{name}.json"
    return json.loads(path.read_text())


def make_loader(rows: list[dict], image_size: int, batch_size: int, train: bool, seed: int) -> DataLoader:
    dataset = UltrasoundImageDataset(rows, image_size=image_size, train=train, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def class_weights(rows: list[dict]) -> torch.Tensor:
    counts = [sum(row["label"] == name for row in rows) for name in CLASS_NAMES]
    total = sum(counts)
    weights = [total / (len(CLASS_NAMES) * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32)
