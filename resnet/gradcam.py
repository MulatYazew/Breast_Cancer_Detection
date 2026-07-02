"""Grad-CAM explanations for Stage 4's ResNet50 classifier.

Used by ``resnet/evaluate.py`` (misclassified-example visualization) and by
the Streamlit app (per-case Grad-CAM overlay alongside the prediction).
"""

from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn


def build_gradcam(model: nn.Module) -> GradCAM:
    """Grad-CAM hooked to ResNet50's last conv block (``layer4``), the standard choice."""
    return GradCAM(model=model, target_layers=[model.layer4[-1]])


def gradcam_overlay(
    cam: GradCAM,
    image_tensor: torch.Tensor,
    image_float01: np.ndarray,
    target_class: int | None = None,
) -> np.ndarray:
    """Compute the Grad-CAM heatmap overlay for one image.

    Args:
        cam: A ``GradCAM`` instance from ``build_gradcam``.
        image_tensor: The model-input tensor, shape (1, C, H, W), normalized.
        image_float01: The same image as an HxWx3 float32 array in [0, 1]
            (un-normalized), used as the background for the overlay.
        target_class: Class index to explain (defaults to the model's own
            top prediction if None).

    Returns:
        HxWx3 uint8 RGB overlay image.
    """
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    return show_cam_on_image(image_float01, grayscale_cam, use_rgb=True)
