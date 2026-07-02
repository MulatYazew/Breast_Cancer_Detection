"""Consistent image loading, resizing, and channel/pixel normalization.

Shared by every stage so a BUSI grayscale ultrasound frame is turned into the
same 3-channel, correctly-scaled tensor representation everywhere (YOLO
dataset builder, U-Net/ResNet datasets, Streamlit app).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image file and return it as HxWx3 RGB uint8, converting grayscale if needed."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image file: {path}")
    return ensure_rgb(image)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or BGR(A) array to 3-channel RGB uint8."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image shape for RGB conversion: {image.shape}")


def to_unit_range(image: np.ndarray) -> np.ndarray:
    """Scale a uint8 image to float32 in [0, 1]."""
    return image.astype(np.float32) / 255.0


def imagenet_normalize(image_float01: np.ndarray) -> np.ndarray:
    """Apply ImageNet mean/std normalization to an HxWx3 float32 [0, 1] image."""
    return (image_float01 - IMAGENET_MEAN) / IMAGENET_STD
