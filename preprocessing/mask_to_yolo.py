"""Derive bounding boxes from segmentation masks and export YOLO label files.

Pipeline: binarize mask -> connected-component analysis (handles multi-lesion
masks, both multiple mask files per image and multiple blobs within one mask
file) -> filter tiny components -> bounding rect per component -> pad -> clip
to image bounds -> normalize to YOLO format -> write one ``.txt`` label file
per image.

Every other stage that needs "image + mask(s) -> lesion box(es)" (the YOLO
dataset builder, the U-Net/ResNet per-lesion training-set builder) calls into
this module rather than re-implementing connected-component analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from preprocessing.roi_utils import pad_and_clip_bbox


@dataclass
class LesionComponent:
    """A single connected-component lesion instance found in a mask file."""

    source_mask_path: Path
    component_label: int
    bbox: tuple[int, int, int, int]   # (x_min, y_min, x_max, y_max), pixel coords, padded+clipped
    raw_bbox: tuple[int, int, int, int]  # bbox before padding, for area/QA checks
    area_px: int
    mask: np.ndarray                  # bool array, full image shape, True where this lesion is


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Load a mask file and binarize it to a {0, 1} uint8 array (single channel)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask file: {mask_path}")
    return (mask > 127).astype(np.uint8)


def mask_components(
    binary_mask: np.ndarray,
    min_area_px: int = 20,
    pad_ratio: float = 0.08,
    source_mask_path: str | Path = "",
) -> list[LesionComponent]:
    """Run connected-component analysis on one binary mask and return lesion components.

    Components smaller than ``min_area_px`` are discarded (noise / annotation
    artifacts). Each returned bbox is padded by ``pad_ratio`` and clipped to
    the mask's own bounds.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )

    components: list[LesionComponent] = []
    for label_id in range(1, num_labels):  # label 0 is background
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        raw_bbox = (x, y, x + w, y + h)
        padded_bbox = pad_and_clip_bbox(raw_bbox, binary_mask.shape, pad_ratio)

        components.append(
            LesionComponent(
                source_mask_path=Path(source_mask_path),
                component_label=label_id,
                bbox=padded_bbox,
                raw_bbox=raw_bbox,
                area_px=area,
                mask=(labels == label_id),
            )
        )
    return components


def component_mask(source_mask_path: str | Path, component_label: int) -> np.ndarray:
    """Recompute one lesion component's boolean mask from its source mask file.

    ``cv2.connectedComponentsWithStats`` is deterministic for a fixed input,
    so re-deriving the component here (rather than caching full-resolution
    boolean arrays in a manifest) keeps manifests small and stateless. Used
    by ``dataset.busi_dataset`` (U-Net/ResNet training) and
    ``preprocessing.pipeline`` (sanity-check visualization).
    """
    binary = load_binary_mask(source_mask_path)
    _, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return (labels == component_label).astype(np.uint8)


def image_lesion_components(
    mask_paths: list[str | Path],
    min_area_px: int = 20,
    pad_ratio: float = 0.08,
) -> list[LesionComponent]:
    """Aggregate lesion components across every mask file belonging to one image."""
    components: list[LesionComponent] = []
    for mask_path in mask_paths:
        binary_mask = load_binary_mask(mask_path)
        components.extend(
            mask_components(
                binary_mask,
                min_area_px=min_area_px,
                pad_ratio=pad_ratio,
                source_mask_path=mask_path,
            )
        )
    return components


def bbox_to_yolo_line(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    class_id: int = 0,
) -> str:
    """Convert a pixel bbox to a normalized YOLO label line: ``class xc yc w h``."""
    x_min, y_min, x_max, y_max = bbox
    h_img, w_img = image_shape[:2]

    box_w = (x_max - x_min) / w_img
    box_h = (y_max - y_min) / h_img
    x_center = (x_min + x_max) / 2 / w_img
    y_center = (y_min + y_max) / 2 / h_img

    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def write_yolo_label(
    label_path: str | Path,
    components: list[LesionComponent],
    image_shape: tuple[int, int],
    class_id: int = 0,
) -> None:
    """Write one YOLO ``.txt`` label file, one line per lesion component."""
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [bbox_to_yolo_line(c.bbox, image_shape, class_id) for c in components]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def overlay_boxes(
    image: np.ndarray,
    components: list[LesionComponent],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw lesion bboxes on a copy of ``image`` (BGR or RGB, either is fine) for sanity checks."""
    vis = image.copy()
    for c in components:
        x_min, y_min, x_max, y_max = c.bbox
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color, thickness)
    return vis
