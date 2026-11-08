"""Classification metrics for benign versus malignant research labels.

Malignant is the positive class. The decision threshold is chosen on the
validation set and then frozen. These figures are not a clinical diagnosis.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Positive class is index 1 (malignant)."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0
    precision = precision_score(y_true, predicted, pos_label=1, zero_division=0)
    recall = recall_score(y_true, predicted, pos_label=1, zero_division=0)
    result = {
        "threshold": float(threshold),
        "sensitivity_malignant": float(sensitivity),
        "specificity_benign": float(specificity),
        "precision_malignant": float(precision),
        "recall_malignant": float(recall),
        "f1_malignant": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "support": {"benign": int((y_true == 0).sum()), "malignant": int((y_true == 1).sum())},
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        result["pr_auc"] = float(average_precision_score(y_true, probabilities))
        fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
        result["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        }
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
        result["roc_curve"] = None
    result["brier"] = float(brier_score_loss(y_true, probabilities))
    try:
        fraction_positive, mean_predicted = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")
        result["calibration"] = {
            "fraction_positive": fraction_positive.tolist(),
            "mean_predicted": mean_predicted.tolist(),
        }
    except ValueError:
        result["calibration"] = None
    return result


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Pick the validation threshold with the highest balanced accuracy."""
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = binary_metrics(y_true, probabilities, float(threshold))
        score = 0.5 * (metrics["sensitivity_malignant"] + metrics["specificity_benign"])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold
