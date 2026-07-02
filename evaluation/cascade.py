"""End-to-end cascade evaluation: real YOLO detection -> real U-Net segmentation
-> real ResNet50 classification, chained exactly as at serving time.

Distinct from each stage's own ``evaluate.py`` (which scores that stage in
isolation against ground truth): this measures how much error propagates
through the full cascade, and requires all three checkpoints referenced by
``config["inference"]`` to already exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from dataset.busi_dataset import CLASS_NAMES, LABEL_TO_IDX
from dataset.splits import load_split
from evaluation.metrics import bbox_iou, classification_metrics, dice_score
from inference.pipeline import build_pipeline
from preprocessing.mask_to_yolo import image_lesion_components
from utils.logging import get_logger

logger = get_logger("cascade_evaluate")


def _closest_gt_lesion(pred_bbox: tuple[int, int, int, int], mask_paths: list[str], prep_cfg: dict):
    """Match the detected box to whichever ground-truth lesion component overlaps it most."""
    components = image_lesion_components(
        mask_paths, min_area_px=prep_cfg["min_mask_area_px"], pad_ratio=prep_cfg["bbox_pad_ratio"]
    )
    if not components:
        return None
    return max(components, key=lambda c: bbox_iou(pred_bbox, c.bbox))


def evaluate_cascade(config: dict, split: str = "test", save_dir: str | Path = "results/reports") -> dict:
    """Run the full cascade over every image in ``split`` and report end-to-end metrics."""
    pipeline = build_pipeline(config)
    rows = load_split(config["paths"]["splits_dir"], split)
    prep_cfg = config["preprocessing"]

    y_true, y_pred, y_prob, box_ious, mask_dices = [], [], [], [], []
    num_detection_failures = 0

    for row in rows:
        try:
            result = pipeline.run(row["image_path"])
        except RuntimeError:
            num_detection_failures += 1
            continue

        y_true.append(LABEL_TO_IDX[row["label"]])
        y_pred.append(LABEL_TO_IDX[result.predicted_class])
        y_prob.append(result.class_probs[CLASS_NAMES[1]])

        gt_lesion = _closest_gt_lesion(result.bbox, row["mask_paths"], prep_cfg)
        if gt_lesion is not None:
            box_ious.append(bbox_iou(result.bbox, gt_lesion.bbox))
            # Crop the GT mask to the *predicted* ROI (result.bbox) rather than
            # the GT lesion's own bbox, then resize to result.mask's shape —
            # this scores the segmentation on the exact region the model saw,
            # so detection error correctly propagates into the dice score.
            x_min, y_min, x_max, y_max = result.bbox
            gt_mask_roi = gt_lesion.mask[y_min:y_max, x_min:x_max].astype(np.uint8)
            gt_mask_roi = cv2.resize(
                gt_mask_roi, result.mask.shape[::-1], interpolation=cv2.INTER_NEAREST
            )
            mask_dices.append(dice_score(result.mask, gt_mask_roi))

    metrics = classification_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    metrics["num_detection_failures"] = num_detection_failures
    metrics["mean_detection_iou"] = float(np.mean(box_ious)) if box_ious else float("nan")
    metrics["mean_segmentation_dice"] = float(np.mean(mask_dices)) if mask_dices else float("nan")

    logger.info("End-to-end cascade metrics on %s split: %s", split, metrics)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"cascade_{split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse

    from utils.config import load_config

    parser = argparse.ArgumentParser(description="End-to-end cascade evaluation (Stage 2->3->4)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--overrides", default=None)
    args = parser.parse_args()

    cfg = load_config(overrides=args.overrides)
    evaluate_cascade(cfg, split=args.split)
