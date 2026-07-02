"""Generic bounding-box -> ROI crop utilities.

Pure geometry: box + image (+ optional mask) + padding -> crop. Deliberately
detection-agnostic — the bbox may come from a YOLO prediction, a
ground-truth mask-derived box, or a physician's manually-drawn rectangle in
the Streamlit app. All three call sites use the same ``crop_roi`` /
``crop_roi_pair`` functions so ROI extraction behaves identically everywhere.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

Bbox = tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max), pixel coords


def pad_and_clip_bbox(bbox: Bbox, image_shape: tuple[int, int], pad_ratio: float) -> Bbox:
    """Expand ``bbox`` by ``pad_ratio`` of its own width/height, then clip to image bounds."""
    x_min, y_min, x_max, y_max = bbox
    h_img, w_img = image_shape[:2]
    box_w, box_h = x_max - x_min, y_max - y_min

    pad_x = box_w * pad_ratio
    pad_y = box_h * pad_ratio

    x_min = max(0, int(round(x_min - pad_x)))
    y_min = max(0, int(round(y_min - pad_y)))
    x_max = min(w_img, int(round(x_max + pad_x)))
    y_max = min(h_img, int(round(y_max + pad_y)))
    return x_min, y_min, x_max, y_max


def crop_roi(
    image: np.ndarray,
    bbox: Bbox,
    pad_ratio: float = 0.05,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, Bbox]:
    """Crop ``image`` to ``bbox`` (padded by ``pad_ratio``, clipped to bounds).

    Args:
        image: HxW(xC) array.
        bbox: (x_min, y_min, x_max, y_max) in pixel coordinates. May come from
            a YOLO prediction, a mask-derived ground-truth box (which may
            already include its own padding — pass ``pad_ratio=0`` in that
            case), or a manually-drawn box.
        pad_ratio: Extra padding applied around ``bbox`` before cropping.
        target_size: If given, (width, height) to resize the crop to.

    Returns:
        Tuple of (cropped array, the padded+clipped bbox actually used — callers
        that also need to crop an aligned mask should reuse this exact bbox).
    """
    padded_bbox = pad_and_clip_bbox(bbox, image.shape, pad_ratio)
    x_min, y_min, x_max, y_max = padded_bbox
    crop = image[y_min:y_max, x_min:x_max]

    if target_size is not None:
        if cv2 is None:
            raise ImportError("opencv-python is required to resize ROI crops")
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    return crop, padded_bbox


def crop_roi_pair(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Bbox,
    pad_ratio: float = 0.05,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, Bbox]:
    """Crop an image and its aligned mask to the same padded+clipped bbox.

    Used when building the U-Net training set (crop image + its ground-truth
    lesion mask together) and by the Streamlit app (crop image + predicted or
    manually-corrected mask together).
    """
    padded_bbox = pad_and_clip_bbox(bbox, image.shape, pad_ratio)
    x_min, y_min, x_max, y_max = padded_bbox

    image_crop = image[y_min:y_max, x_min:x_max]
    mask_crop = mask[y_min:y_max, x_min:x_max]

    if target_size is not None:
        if cv2 is None:
            raise ImportError("opencv-python is required to resize ROI crops")
        image_crop = cv2.resize(image_crop, target_size, interpolation=cv2.INTER_LINEAR)
        mask_crop = cv2.resize(
            mask_crop, target_size, interpolation=cv2.INTER_NEAREST
        )

    return image_crop, mask_crop, padded_bbox


def apply_mask(image: np.ndarray, mask: np.ndarray, background: int = 0) -> np.ndarray:
    """Zero out (or set to ``background``) every pixel outside the binary ``mask``.

    Produces the "masked tumor ROI" that Stage 4 (ResNet50) consumes: the ROI
    crop with everything but the segmented lesion suppressed.
    """
    binary = (mask > 0).astype(image.dtype)
    if image.ndim == 3 and binary.ndim == 2:
        binary = binary[..., None]
    return image * binary + background * (1 - binary)
