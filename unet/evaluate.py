"""Stage 3 evaluation: Dice, IoU, precision, recall, specificity + visualizations.

Safe to run given an already-trained checkpoint (forward passes only, no
gradient updates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from evaluation.metrics import segmentation_metrics
from unet.dataset import build_unet_dataloader
from unet.model import build_unet
from utils.logging import get_logger
from utils.seed import get_device
from visualization.plots import save_figure

logger = get_logger("unet_evaluate")


def load_unet_checkpoint(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet_cfg = checkpoint["config"]
    model = build_unet(
        architecture=unet_cfg["architecture"],
        encoder_name=unet_cfg["encoder"],
        encoder_weights=None,  # weights are overwritten by the checkpoint anyway
        in_channels=unet_cfg["in_channels"],
        classes=unet_cfg["classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_unet(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
    save_dir: str | Path = "results/unet",
) -> dict:
    """Compute mean Dice/IoU/precision/recall/specificity over ``split``."""
    device = get_device(config.get("device"))
    model = load_unet_checkpoint(checkpoint_path, device)
    loader = build_unet_dataloader(config, split, shuffle=False)

    per_sample_metrics: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            for pred, gt in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                per_sample_metrics.append(segmentation_metrics(pred, gt))

    mean_metrics = {
        key: float(np.mean([m[key] for m in per_sample_metrics])) for key in per_sample_metrics[0]
    }
    logger.info("U-Net %s-split metrics: %s", split, mean_metrics)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    visualize_predictions(model, loader, device, n_samples=6, save_path=save_dir / "predictions_grid.png")

    return mean_metrics


def visualize_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    n_samples: int = 6,
    save_path: str | Path | None = None,
) -> None:
    """Prediction vs. ground truth vs. overlay vs. difference map, for a handful of samples."""
    import matplotlib.pyplot as plt

    model.eval()
    batch = next(iter(loader))
    images = batch["image"][:n_samples].to(device)
    masks = batch["mask"][:n_samples].numpy()

    with torch.no_grad():
        preds = (torch.sigmoid(model(images)) > 0.5).float().cpu().numpy()

    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min() + 1e-7)

    n = len(images_np)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for row in range(n):
        gt = masks[row, 0]
        pred = preds[row, 0]
        diff = np.abs(gt - pred)

        axes[row, 0].imshow(images_np[row])
        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 2].imshow(pred, cmap="gray")
        axes[row, 3].imshow(diff, cmap="hot")
        for col in range(4):
            axes[row, col].axis("off")

    axes[0, 0].set_title("image")
    axes[0, 1].set_title("ground truth")
    axes[0, 2].set_title("prediction")
    axes[0, 3].set_title("difference")
    fig.tight_layout()
    save_figure(fig, save_path)


if __name__ == "__main__":
    import argparse

    from utils.config import load_config

    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint (Stage 3)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--overrides", default=None)
    args = parser.parse_args()

    cfg = load_config(overrides=args.overrides)
    evaluate_unet(cfg, args.checkpoint, split=args.split)
