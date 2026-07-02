"""Generic, stage-agnostic plotting primitives shared across the pipeline.

Stage-specific composition (e.g. "the 4-panel case report for one inference
run") lives in each stage's own module and calls into these primitives rather
than duplicating matplotlib boilerplate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_figure(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_image_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    alpha: float = 0.4,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Three-panel figure: image / mask / overlay. Used for mask sanity checks."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("mask")
    axes[2].imshow(image)
    axes[2].imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=alpha)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_sample_grid(
    samples: list[tuple[np.ndarray, np.ndarray, str]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grid of (image, mask, label) sanity-check samples, one row each."""
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for row, (image, mask, label) in enumerate(samples):
        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(label, fontsize=9)
        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 2].imshow(image)
        axes[row, 2].imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.4)
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_title("image")
    axes[0, 1].set_title("mask")
    axes[0, 2].set_title("overlay")
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_bbox_overlay(
    images_with_boxes: list[tuple[np.ndarray, list[tuple[int, int, int, int]]]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grid of images with bounding boxes drawn, for mask->YOLO-box sanity checks."""
    n = len(images_with_boxes)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (image, boxes) in zip(axes, images_with_boxes):
        ax.imshow(image)
        for x_min, y_min, x_max, y_max in boxes:
            ax.add_patch(
                plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )
        ax.axis("off")
    for ax in axes[len(images_with_boxes):]:
        ax.axis("off")

    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "training curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one or more named loss/metric curves (e.g. {'train_loss': [...], 'val_loss': [...]})."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in history.items():
        ax.plot(range(1, len(values) + 1), values, label=name)
    ax.set_xlabel("epoch")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "confusion matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_curve(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generic 2D curve plot (ROC, PR, etc.)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig
