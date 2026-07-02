"""Thin wrapper around the Ultralytics YOLO26n detector.

Kept deliberately small: model architecture/weights loading only. Training
loop, evaluation, and ROI cropping all live elsewhere (``yolo.train``,
``yolo.evaluate``, ``preprocessing.roi_utils``).
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def build_yolo_model(weights: str = "yolo26n.pt") -> YOLO:
    """Instantiate YOLO26n, starting from pretrained COCO weights by default."""
    return YOLO(weights)


def load_trained_yolo(weights_path: str | Path) -> YOLO:
    """Load a previously-trained checkpoint (e.g. ``checkpoints/yolo/.../best.pt``)."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {weights_path}")
    return YOLO(str(weights_path))
