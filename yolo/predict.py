"""Single-image YOLO26n inference helper, used by ``inference/`` and the Streamlit app.

Returns plain (bbox, confidence) tuples rather than Ultralytics ``Results``
objects, so downstream code (``preprocessing.roi_utils.crop_roi``,
manual-override logic) does not need to know about Ultralytics internals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

Bbox = tuple[int, int, int, int]


def detect_tumor(
    model: YOLO,
    image: str | Path | np.ndarray,
    imgsz: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[tuple[Bbox, float]]:
    """Run detection and return every box above ``conf_threshold`` as (bbox, confidence).

    Boxes are sorted by confidence, descending, so ``result[0]`` is the top
    detection (the one the Streamlit app pre-selects for physician review).
    """
    results = model.predict(
        image, imgsz=imgsz, conf=conf_threshold, iou=iou_threshold, verbose=False
    )
    boxes = results[0].boxes
    if len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    detections = [
        (tuple(int(round(v)) for v in box), float(conf)) for box, conf in zip(xyxy, confs)
    ]
    detections.sort(key=lambda d: d[1], reverse=True)
    return detections


def detect_top_box(
    model: YOLO,
    image: str | Path | np.ndarray,
    imgsz: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> tuple[Bbox, float] | None:
    """Convenience wrapper returning only the highest-confidence detection, if any."""
    detections = detect_tumor(model, image, imgsz, conf_threshold, iou_threshold)
    return detections[0] if detections else None
