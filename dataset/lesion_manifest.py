"""Per-lesion-instance manifest, built on top of the per-image split manifest.

BUSI labels (benign/malignant) are per-image, but some images contain more
than one lesion (multiple mask files, or rarely multiple blobs in one mask
file). U-Net and ResNet50 are trained per lesion instance, not per image, so
this module expands each split's image-level rows into one row per detected
lesion component using ``preprocessing.mask_to_yolo`` — the same
connected-component logic used to generate YOLO labels, reused here rather
than re-implemented.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataset.manifest import ManifestRecord
from preprocessing.mask_to_yolo import image_lesion_components


def build_lesion_manifest(
    records: list[ManifestRecord],
    min_area_px: int = 20,
    bbox_pad_ratio: float = 0.08,
) -> list[dict]:
    """Expand image-level records into one row per lesion instance.

    Each row's ``bbox`` is already padded by ``bbox_pad_ratio`` (see
    ``preprocessing.mask_to_yolo``), matching the box a well-trained detector
    should localize at inference time.
    """
    rows: list[dict] = []
    for record in records:
        components = image_lesion_components(
            record.mask_paths, min_area_px=min_area_px, pad_ratio=bbox_pad_ratio
        )
        for component in components:
            rows.append(
                {
                    "image_id": record.image_id,
                    "label": record.label,
                    "image_path": str(record.image_path),
                    "source_mask_path": str(component.source_mask_path),
                    "component_label": component.component_label,
                    "bbox": list(component.bbox),
                }
            )
    return rows


def save_lesion_manifest(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def load_lesion_manifest(path: str | Path) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)
