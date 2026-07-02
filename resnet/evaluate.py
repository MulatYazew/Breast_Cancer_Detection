"""Stage 4 evaluation: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, kappa, MCC.

Plus ROC curve, PR curve, confusion matrix, Grad-CAM, and misclassified
examples. Safe to run given an already-trained checkpoint (forward passes
+ Grad-CAM backward-for-explanation only — no optimizer updates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from dataset.busi_dataset import CLASS_NAMES
from evaluation.metrics import classification_metrics
from resnet.dataset import build_resnet_dataloader
from resnet.gradcam import build_gradcam, gradcam_overlay
from resnet.model import build_resnet50
from utils.logging import get_logger
from utils.seed import get_device
from visualization.plots import plot_confusion_matrix, plot_curve, save_figure

logger = get_logger("resnet_evaluate")


def load_resnet_checkpoint(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    resnet_cfg = checkpoint["config"]
    model = build_resnet50(pretrained=False, num_classes=resnet_cfg["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_resnet(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "test",
    save_dir: str | Path = "results/resnet",
) -> dict:
    """Compute the full Stage 4 metric suite and save diagnostic plots."""
    device = get_device(config.get("device"))
    model = load_resnet_checkpoint(checkpoint_path, device)
    loader = build_resnet_dataloader(config, split, shuffle=False)

    all_labels, all_preds, all_probs, all_images, all_ids = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_images.append(images.cpu().numpy())
            all_ids.extend(batch["image_id"])

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    images_np = np.concatenate(all_images)

    metrics = classification_metrics(y_true, y_pred, y_prob)
    logger.info("ResNet50 %s-split metrics: %s", split, metrics)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, save_path=save_dir / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plot_curve(fpr, tpr, "False Positive Rate", "True Positive Rate", "ROC curve", save_dir / "roc_curve.png")

    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
    plot_curve(pr_recall, pr_precision, "Recall", "Precision", "PR curve", save_dir / "pr_curve.png")

    visualize_misclassified(model, images_np, y_true, y_pred, all_ids, device, save_dir / "misclassified.png")

    return metrics


def visualize_misclassified(
    model: torch.nn.Module,
    images_np: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    image_ids: list[str],
    device: torch.device,
    save_path: str | Path,
    n_samples: int = 6,
) -> None:
    """Grad-CAM overlays for up to ``n_samples`` misclassified examples."""
    import matplotlib.pyplot as plt

    mis_idx = np.where(y_true != y_pred)[0][:n_samples]
    if len(mis_idx) == 0:
        logger.info("No misclassified examples to visualize.")
        return

    cam = build_gradcam(model)
    fig, axes = plt.subplots(1, len(mis_idx), figsize=(3.2 * len(mis_idx), 3.2))
    axes = np.atleast_1d(axes)

    for ax, idx in zip(axes, mis_idx):
        image_tensor = torch.from_numpy(images_np[idx : idx + 1]).to(device)
        image_float = images_np[idx].transpose(1, 2, 0)
        image_float = (image_float - image_float.min()) / (image_float.max() - image_float.min() + 1e-7)

        overlay = gradcam_overlay(cam, image_tensor, image_float.astype(np.float32), target_class=int(y_pred[idx]))
        ax.imshow(overlay)
        ax.set_title(
            f"{image_ids[idx]}\ntrue={CLASS_NAMES[y_true[idx]]} pred={CLASS_NAMES[y_pred[idx]]}", fontsize=8
        )
        ax.axis("off")

    fig.tight_layout()
    save_figure(fig, save_path)


if __name__ == "__main__":
    import argparse

    from utils.config import load_config

    parser = argparse.ArgumentParser(description="Evaluate a trained ResNet50 checkpoint (Stage 4)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--overrides", default=None)
    args = parser.parse_args()

    cfg = load_config(overrides=args.overrides)
    evaluate_resnet(cfg, args.checkpoint, split=args.split)
