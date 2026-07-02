"""Per-case visualization: composes the shared ``visualization.plots`` primitives
into the 4-panel end-to-end pipeline figure (original / detected ROI / mask
overlay / classification result). Used by ``inference.py`` (batch CLI) and
available to the Streamlit app for the downloadable case report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inference.pipeline import CaseResult
from visualization.plots import save_figure


def plot_case_result(result: CaseResult, save_path: str | Path | None = None) -> plt.Figure:
    """Render the full pipeline output for one case: image, ROI, mask overlay, classification."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))

    axes[0].imshow(result.image)
    x_min, y_min, x_max, y_max = result.bbox
    axes[0].add_patch(
        plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor="lime", linewidth=2
        )
    )
    axes[0].set_title(f"detection ({result.bbox_source})")

    axes[1].imshow(result.roi_crop)
    axes[1].set_title("ROI crop")

    axes[2].imshow(result.roi_crop)
    axes[2].imshow(np.ma.masked_where(result.mask == 0, result.mask), cmap="autumn", alpha=0.45)
    axes[2].set_title(f"segmentation ({result.mask_source})")

    axes[3].imshow(result.masked_roi)
    axes[3].set_title(f"{result.predicted_class} ({result.confidence:.1%})")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    save_figure(fig, save_path)
    return fig
