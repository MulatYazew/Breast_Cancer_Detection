"""Build the Stage 1 preprocessing report: class counts, resolution stats, mismatches."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from dataset.manifest import ManifestRecord
from dataset.splits import split_class_counts
from preprocessing.validation import ValidationReport


def resolution_stats(records: list[ManifestRecord]) -> dict:
    """Min/max/most-common image resolution across the dataset."""
    sizes: dict[tuple[int, int], int] = {}
    for record in records:
        with Image.open(record.image_path) as img:
            sizes[img.size] = sizes.get(img.size, 0) + 1

    most_common = max(sizes.items(), key=lambda kv: kv[1])
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    return {
        "num_unique_resolutions": len(sizes),
        "most_common_resolution": {"size": most_common[0], "count": most_common[1]},
        "min_resolution": [min(widths), min(heights)],
        "max_resolution": [max(widths), max(heights)],
    }


def build_preprocessing_report(
    records: list[ManifestRecord],
    validation_report: ValidationReport,
    splits: dict[str, list[ManifestRecord]] | None = None,
    save_path: str | Path | None = None,
) -> dict:
    """Assemble the full Stage 1 report and optionally save it as JSON."""
    class_counts: dict[str, int] = {}
    lesion_counts: dict[str, int] = {}
    for record in records:
        class_counts[record.label] = class_counts.get(record.label, 0) + 1
        lesion_counts[record.label] = lesion_counts.get(record.label, 0) + len(record.mask_paths)

    report = {
        "total_images": len(records),
        "class_counts": class_counts,
        "total_lesion_masks": sum(lesion_counts.values()),
        "lesion_masks_by_class": lesion_counts,
        "resolution_stats": resolution_stats(records),
        "validation": validation_report.to_dict(),
    }
    if splits is not None:
        report["split_class_counts"] = split_class_counts(splits)
        report["split_sizes"] = {name: len(recs) for name, recs in splits.items()}

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    return report
