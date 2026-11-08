"""Boundary distances for binary segmentation masks.

Empty masks are reported explicitly. A Dice score of 1 on two empty masks is
not treated as a successful surface distance.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _surfaces(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError("surface distance expects a 2-D mask")
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    eroded = ndimage.binary_erosion(binary, iterations=1, border_value=0)
    return binary & ~eroded


def _directed_distances(from_surface: np.ndarray, to_mask: np.ndarray) -> np.ndarray:
    if not from_surface.any():
        return np.array([], dtype=np.float64)
    # Distance to the nearest foreground pixel of the other mask.
    distances = ndimage.distance_transform_edt(~(to_mask > 0))
    return distances[from_surface].astype(np.float64)


def surface_distances(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Symmetric surface distances in pixels. Spacing is unknown for BUSI."""
    pred = np.asarray(pred_mask) > 0
    gt = np.asarray(gt_mask) > 0
    if pred.ndim == 3:
        pred = pred.max(axis=0) if pred.shape[0] < 8 else pred.max(axis=2)
    if gt.ndim == 3:
        gt = gt.max(axis=0) if gt.shape[0] < 8 else gt.max(axis=2)

    pred_empty = not bool(pred.any())
    gt_empty = not bool(gt.any())
    if pred_empty or gt_empty:
        return {
            "hd95_px": None,
            "assd_px": None,
            "pred_empty": pred_empty,
            "gt_empty": gt_empty,
            "defined": False,
            "reason": "Surface distance is undefined when either mask is empty.",
        }

    pred_surface = _surfaces(pred)
    gt_surface = _surfaces(gt)
    forward = _directed_distances(pred_surface, gt)
    backward = _directed_distances(gt_surface, pred)
    pooled = np.concatenate([forward, backward])
    if pooled.size == 0:
        return {
            "hd95_px": None,
            "assd_px": None,
            "pred_empty": False,
            "gt_empty": False,
            "defined": False,
            "reason": "No boundary pixels were found.",
        }
    return {
        "hd95_px": float(np.percentile(pooled, 95)),
        "assd_px": float(pooled.mean()),
        "pred_empty": False,
        "gt_empty": False,
        "defined": True,
        "units": "pixels",
        "spacing_available": False,
    }
