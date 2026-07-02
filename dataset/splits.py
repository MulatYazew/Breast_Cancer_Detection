"""Stratified, reproducible train/val/test split over the manifest rows."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from dataset.manifest import ManifestRecord, file_hash, manifest_to_rows


def _group_by_content(records: list[ManifestRecord]) -> list[list[ManifestRecord]]:
    """Group records with byte-identical image content.

    BUSI contains at least one case (see preprocessing validation's
    ``duplicate_image_cross_class`` finding) where the same underlying frame
    appears under two different class folders with two different lesion
    masks. Splitting content-duplicates across train/val/test would leak the
    same background pixels between splits, so duplicate groups are always
    kept together and assigned to a single split as a unit.
    """
    groups: dict[str, list[ManifestRecord]] = {}
    for record in records:
        h = file_hash(record.image_path)
        groups.setdefault(h, []).append(record)
    return list(groups.values())


def stratified_split(
    records: list[ManifestRecord],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ManifestRecord]]:
    """Split records into train/val/test, stratified by class label.

    Content-duplicate groups (see ``_group_by_content``) are split as single
    units, stratified by the label of the group's first (sorted) record.

    Args:
        records: Manifest records to split (order-independent; sorted internally
            for determinism before the sklearn split is applied).
        train_frac, val_frac, test_frac: Must sum to ~1.0.
        seed: Fixed seed so the split is identical across runs.
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    records = sorted(records, key=lambda r: (r.label, r.image_id))
    groups = [sorted(g, key=lambda r: (r.label, r.image_id)) for g in _group_by_content(records)]
    groups.sort(key=lambda g: (g[0].label, g[0].image_id))
    group_labels = [g[0].label for g in groups]

    train_groups, temp_groups, _, temp_labels = train_test_split(
        groups,
        group_labels,
        train_size=train_frac,
        random_state=seed,
        stratify=group_labels,
    )

    remaining_frac = val_frac + test_frac
    val_groups, test_groups = train_test_split(
        temp_groups,
        train_size=val_frac / remaining_frac,
        random_state=seed,
        stratify=temp_labels,
    )

    return {
        "train": [r for g in train_groups for r in g],
        "val": [r for g in val_groups for r in g],
        "test": [r for g in test_groups for r in g],
    }


def save_splits(splits: dict[str, list[ManifestRecord]], out_dir: str | Path) -> None:
    """Write one JSON manifest file per split (``train.json``, ``val.json``, ``test.json``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        with open(out_dir / f"{split_name}.json", "w") as f:
            json.dump(manifest_to_rows(records), f, indent=2)


def load_split(split_dir: str | Path, split_name: str) -> list[dict]:
    """Load a previously saved split (e.g. ``load_split("dataset/splits", "train")``)."""
    path = Path(split_dir) / f"{split_name}.json"
    with open(path, "r") as f:
        return json.load(f)


def subsample_rows(rows: list[dict], samples_per_class: int, label_key: str = "label") -> list[dict]:
    """Keep at most ``samples_per_class`` rows per class, preserving manifest order."""
    kept_counts: dict[str, int] = {}
    kept_rows: list[dict] = []
    for row in rows:
        label = row[label_key]
        if kept_counts.get(label, 0) < samples_per_class:
            kept_rows.append(row)
            kept_counts[label] = kept_counts.get(label, 0) + 1
    return kept_rows


def apply_quick_run_subsample(rows: list[dict], config: dict, label_key: str = "label") -> list[dict]:
    """Cap ``rows`` to ``debug.quick_run_samples_per_class`` per class when ``debug.quick_run`` is set.

    Used by every train-time dataloader/dataset-builder so ``configs/quick_run.yaml``
    actually smoke-tests on a tiny subset rather than just running fewer epochs
    over the full dataset.
    """
    debug_cfg = config.get("debug", {})
    if not debug_cfg.get("quick_run", False):
        return rows
    return subsample_rows(rows, debug_cfg["quick_run_samples_per_class"], label_key=label_key)


def split_class_counts(splits: dict[str, list[ManifestRecord]]) -> dict[str, dict[str, int]]:
    """Return per-split class counts, e.g. for the preprocessing report."""
    counts: dict[str, dict[str, int]] = {}
    for split_name, records in splits.items():
        split_counts: dict[str, int] = {}
        for r in records:
            split_counts[r.label] = split_counts.get(r.label, 0) + 1
        counts[split_name] = split_counts
    return counts
