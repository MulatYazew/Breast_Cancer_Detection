"""Metric functions shared across Stage 2 (YOLO), Stage 3 (U-Net), and Stage 4 (ResNet50).

Kept in one place so IoU/precision/recall/F1 math is defined once instead of
duplicated per stage-specific evaluate.py.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Intersection-over-Union between two binary masks (or boxes rasterized to masks)."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((intersection + eps) / (union + eps))


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Dice / F1 coefficient between two binary masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return float((2 * intersection + eps) / (pred.sum() + gt.sum() + eps))


def segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Dice, IoU, precision, recall, specificity for one predicted/GT mask pair."""
    pred = pred_mask.astype(bool).flatten()
    gt = gt_mask.astype(bool).flatten()

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    eps = 1e-7

    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "specificity": float((tn + eps) / (tn + fp + eps)),
    }


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """IoU between two (x_min, y_min, x_max, y_max) boxes."""
    xa_min, ya_min, xa_max, ya_max = box_a
    xb_min, yb_min, xb_max, yb_max = box_b

    inter_x_min = max(xa_min, xb_min)
    inter_y_min = max(ya_min, yb_min)
    inter_x_max = min(xa_max, xb_max)
    inter_y_max = min(ya_max, yb_max)

    inter_area = max(0.0, inter_x_max - inter_x_min) * max(0.0, inter_y_max - inter_y_min)
    area_a = max(0.0, xa_max - xa_min) * max(0.0, ya_max - ya_min)
    area_b = max(0.0, xb_max - xb_min) * max(0.0, yb_max - yb_min)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Cohen's kappa, MCC.

    Args:
        y_true: Ground-truth binary labels (0=benign, 1=malignant).
        y_pred: Predicted binary labels.
        y_prob: Predicted probability of the positive (malignant) class, needed
            for ROC-AUC / PR-AUC. If omitted, those two metrics are set to NaN.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics
