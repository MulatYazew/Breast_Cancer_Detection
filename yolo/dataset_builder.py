"""Build an Ultralytics-format detection dataset from the Stage 1 split manifests.

Only assembles the on-disk layout Ultralytics expects (``images/{split}``,
``labels/{split}``, ``data.yaml``); label generation itself is delegated to
``preprocessing.mask_to_yolo`` rather than re-implemented here. Images are
symlinked (not copied) to avoid duplicating the dataset on disk.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from preprocessing.mask_to_yolo import image_lesion_components, write_yolo_label
from utils.logging import get_logger

logger = get_logger("yolo_dataset_builder")


def build_yolo_dataset(
    splits: dict[str, list[dict]],
    out_dir: str | Path,
    min_area_px: int = 20,
    bbox_pad_ratio: float = 0.08,
    class_names: tuple[str, ...] = ("tumor",),
) -> Path:
    """Materialize an Ultralytics dataset directory and return its ``data.yaml`` path.

    Args:
        splits: Mapping of split name -> list of row dicts as produced by
            ``dataset.manifest.manifest_to_rows`` / saved by ``dataset.splits.save_splits``
            (each row has ``image_path`` and ``mask_paths``).
        out_dir: Destination directory (e.g. ``dataset/yolo_dataset``).
        min_area_px, bbox_pad_ratio: Forwarded to ``preprocessing.mask_to_yolo``.
        class_names: Single "tumor" class by default — benign/malignant
            distinction happens downstream at the ResNet50 stage, not here.

    Idempotent: safe to re-run (existing symlinks/labels are overwritten).
    """
    out_dir = Path(out_dir)

    for split_name, rows in splits.items():
        image_dir = out_dir / "images" / split_name
        label_dir = out_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for row in rows:
            image_path = Path(row["image_path"]).resolve()
            link_path = image_dir / image_path.name
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(image_path)

            with Image.open(image_path) as img:
                width, height = img.size

            components = image_lesion_components(
                row["mask_paths"], min_area_px=min_area_px, pad_ratio=bbox_pad_ratio
            )
            label_path = label_dir / f"{image_path.stem}.txt"
            write_yolo_label(label_path, components, (height, width), class_id=0)

        logger.info("Built YOLO %s split: %d images -> %s", split_name, len(rows), image_dir)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    data_yaml_path = out_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    logger.info("Wrote %s", data_yaml_path)
    return data_yaml_path
