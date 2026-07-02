"""Stage-aware, synchronized image(+mask/+bbox) augmentation pipelines.

Ultrasound morphology (margin shape, echo texture) *is* the diagnostic
signal, so augmentation strength is deliberately different per stage:

- YOLO (detection) only needs to localize the lesion, so it gets the full
  safe set plus the tolerant "cautious" set (bbox-aware crop, light additive
  noise) and a tiny elastic warp.
- U-Net (segmentation) gets the same safe+cautious sets plus a tiny elastic
  warp (mask-aware), since boundary robustness helps segmentation and a small
  warp does not change the classification label.
- ResNet50 (classification) gets only the conservative safe set. Elastic
  deformation, heavy noise, and lesion-cropping risk are excluded entirely
  because they can silently flip apparent margin character (smooth vs.
  spiculated) while the benign/malignant label stays fixed.

Explicitly never used anywhere (see prompt rationale): vertical flip (breaks
the real near/far-field depth axis), mixup/cutmix (blends two possibly
different-class lesions), cutout over the lesion region, heavy hue/color
jitter (meaningless for grayscale US).
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np


def _safe_transforms(prob: float) -> list[A.BasicTransform]:
    """Safe / generally beneficial transforms, used at every stage."""
    return [
        A.HorizontalFlip(p=prob),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=prob),
        A.RandomGamma(gamma_limit=(80, 120), p=prob),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=prob * 0.5),
        A.Affine(rotate=(-15, 15), fill=0, fill_mask=0, p=prob),
        # Tight-limit shift+scale (probe positioning), rotation handled above.
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.90, 1.10), rotate=0, fill=0, fill_mask=0, p=prob),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=prob * 0.5),
        A.GaussianBlur(blur_limit=(3, 3), p=prob * 0.3),
    ]


def _cautious_bbox_transforms(prob: float) -> list[A.BasicTransform]:
    """Cautious-set transforms usable only where bbox-aware (detection stage)."""
    return [
        A.GaussNoise(std_range=(0.02, 0.05), p=prob * 0.4),
        # erosion_rate=0.0 guarantees every bbox stays fully inside the crop.
        A.BBoxSafeRandomCrop(erosion_rate=0.0, p=prob * 0.3),
    ]


def _cautious_mask_transforms(prob: float, roi_size: tuple[int, int]) -> list[A.BasicTransform]:
    """Cautious-set transforms usable only where mask-aware (segmentation stage)."""
    return [
        A.GaussNoise(std_range=(0.02, 0.05), p=prob * 0.4),
        # Conservative zoom (scale close to 1.0): mild enough that a lesion
        # already centered by ROI padding is very unlikely to be clipped.
        A.RandomResizedCrop(size=roi_size, scale=(0.9, 1.0), ratio=(0.95, 1.05), p=prob * 0.3),
    ]


def _elastic(alpha: float, sigma: float, prob: float) -> A.BasicTransform:
    """Tiny elastic warp. Detection/segmentation only — never classification."""
    return A.ElasticTransform(alpha=alpha, sigma=sigma, fill=0, fill_mask=0, p=prob)


def build_yolo_augmentations(
    imgsz: int,
    safe_prob: float = 0.4,
    cautious_prob: float = 0.3,
    use_cautious_set: bool = True,
    use_elastic: bool = True,
    elastic_alpha: float = 20,
    elastic_sigma: float = 5,
) -> A.Compose:
    """Bbox-aware augmentation pipeline for Stage 2 (YOLO detection)."""
    transforms = _safe_transforms(safe_prob)
    if use_cautious_set:
        transforms += _cautious_bbox_transforms(cautious_prob)
    if use_elastic:
        transforms.append(_elastic(elastic_alpha, elastic_sigma, cautious_prob * 0.3))
    transforms.append(A.Resize(imgsz, imgsz))
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def build_unet_augmentations(
    roi_size: tuple[int, int],
    safe_prob: float = 0.4,
    cautious_prob: float = 0.3,
    use_cautious_set: bool = True,
    use_elastic: bool = True,
    elastic_alpha: float = 15,
    elastic_sigma: float = 4,
) -> A.Compose:
    """Mask-synchronized augmentation pipeline for Stage 3 (U-Net segmentation)."""
    transforms = _safe_transforms(safe_prob)
    if use_cautious_set:
        transforms += _cautious_mask_transforms(cautious_prob, roi_size)
    if use_elastic:
        transforms.append(_elastic(elastic_alpha, elastic_sigma, cautious_prob * 0.3))
    transforms.append(A.Resize(*roi_size))
    return A.Compose(transforms)


def build_resnet_augmentations(roi_size: tuple[int, int], safe_prob: float = 0.4) -> A.Compose:
    """Conservative, safe-set-only pipeline for Stage 4 (ResNet50 classification).

    Deliberately excludes elastic deformation, heavy noise, and aggressive
    crop/zoom: fine morphological distortion here can corrupt the
    benign/malignant signal while the label stays fixed.
    """
    conservative_prob = safe_prob * 0.75  # lighter than detection/segmentation
    transforms = [
        A.HorizontalFlip(p=conservative_prob),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=conservative_prob),
        A.RandomGamma(gamma_limit=(90, 110), p=conservative_prob),
        A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=conservative_prob * 0.5),
        A.Affine(rotate=(-10, 10), fill=0, p=conservative_prob),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.92, 1.08), rotate=0, fill=0, p=conservative_prob),
        A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=True, p=conservative_prob * 0.4),
        A.Resize(*roi_size),
    ]
    return A.Compose(transforms)


def visualize_augmentations(
    image: np.ndarray,
    transform: A.Compose,
    n_samples: int = 5,
    save_path: str | Path | None = None,
    mask: np.ndarray | None = None,
) -> plt.Figure:
    """Show the original image plus ``n_samples`` augmented variants, side by side."""
    fig, axes = plt.subplots(2, n_samples + 1, figsize=(3 * (n_samples + 1), 6))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("original")
    axes[0, 0].axis("off")
    if mask is not None:
        axes[1, 0].imshow(mask, cmap="gray")
    axes[1, 0].axis("off")

    for i in range(n_samples):
        kwargs = {"image": image}
        if mask is not None:
            kwargs["mask"] = mask
        augmented = transform(**kwargs)
        axes[0, i + 1].imshow(augmented["image"])
        axes[0, i + 1].set_title(f"aug {i + 1}")
        axes[0, i + 1].axis("off")
        if mask is not None:
            axes[1, i + 1].imshow(augmented["mask"], cmap="gray")
        axes[1, i + 1].axis("off")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
