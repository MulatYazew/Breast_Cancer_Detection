"""Stage 2 evaluation: mAP50, mAP50-95, precision, recall, F1, IoU + plots.

Safe to run given an already-trained checkpoint (no gradient updates). mAP /
precision / recall and the PR-curve / confusion-matrix / example-detection
plots come from Ultralytics' own ``model.val(..., plots=True)``; mean IoU is
computed explicitly here since Ultralytics does not expose it directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.metrics import bbox_iou
from utils.logging import get_logger
from yolo.model import load_trained_yolo

logger = get_logger("yolo_evaluate")


def evaluate_yolo(
    weights_path: str | Path,
    data_yaml: str | Path,
    imgsz: int = 640,
    split: str = "test",
    save_dir: str | Path = "results/yolo",
    conf_threshold: float = 0.25,
) -> dict:
    """Run Ultralytics validation and return the headline detection metrics.

    Side effect: PR curve, confusion matrix, training/loss curves (if
    available in the run dir) and example detections are saved under
    ``<save_dir>/eval/`` by Ultralytics itself.
    """
    model = load_trained_yolo(weights_path)

    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        split=split,
        project=str(save_dir),
        name="eval",
        plots=True,
        conf=conf_threshold,
        exist_ok=True,
    )

    box = metrics.box
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = float(2 * precision * recall / (precision + recall + 1e-7))

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }
    logger.info("YOLO %s-split metrics: %s", split, result)
    return result


def compute_mean_iou(
    weights_path: str | Path,
    lesion_rows: list[dict],
    imgsz: int = 640,
    conf_threshold: float = 0.25,
) -> dict[str, float]:
    """Mean IoU between the top predicted box and the ground-truth box, per lesion instance.

    Args:
        lesion_rows: Per-lesion manifest rows (see ``dataset.lesion_manifest``),
            each with an ``image_path`` and ground-truth ``bbox``.
    """
    model = load_trained_yolo(weights_path)
    ious: list[float] = []

    for row in lesion_rows:
        gt_box = tuple(row["bbox"])
        results = model.predict(row["image_path"], imgsz=imgsz, conf=conf_threshold, verbose=False)
        pred_boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) else np.empty((0, 4))

        if len(pred_boxes) == 0:
            ious.append(0.0)
            continue
        best_iou = max(bbox_iou(gt_box, tuple(pred_box)) for pred_box in pred_boxes)
        ious.append(best_iou)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    logger.info("Mean IoU over %d lesion instances: %.4f", len(ious), mean_iou)
    return {"mean_iou": mean_iou, "num_instances": len(ious)}


if __name__ == "__main__":
    import argparse

    from dataset.lesion_manifest import load_lesion_manifest

    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO26n checkpoint (Stage 2)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-yaml", default="dataset/yolo_dataset/data.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--lesion-manifest", default=None, help="For the supplementary mean-IoU metric")
    args = parser.parse_args()

    evaluate_yolo(args.weights, args.data_yaml, split=args.split)
    if args.lesion_manifest:
        compute_mean_iou(args.weights, load_lesion_manifest(args.lesion_manifest))
