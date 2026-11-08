"""Sequential training, validation-set model choice, and a reloadable checkpoint.

Test metrics are recorded after the choice. They are not used to pick the model.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from classification.data import CLASS_NAMES, load_split_rows, make_leakage_aware_split
from classification.models import COMPARISON_BACKBONES, PRIMARY_BACKBONE
from classification.train import release, score_split, train_one
from utils.hardware import current_rss_bytes, hardware_snapshot
from utils.seed import get_device


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return value
    return value


def _strip_curves(metrics: dict) -> dict:
    kept = {key: value for key, value in metrics.items() if key not in {"roc_curve", "probabilities", "labels", "image_ids"}}
    return kept


def run_classification(repo_root: Path, settings: dict) -> dict:
    """Train the primary model, then each comparison model, one at a time."""
    repo_root = Path(repo_root)
    snapshot = hardware_snapshot(repo_root)
    if snapshot["disk_free_bytes"] < 2 * 1024**3:
        raise RuntimeError("Less than 2 GB free disk. Training was not started.")
    split = make_leakage_aware_split(
        settings["config"], repo_root, hamming_max=settings.get("hamming_max", 5), seed=settings.get("seed", 42)
    )
    device = get_device(settings["config"].get("device"))
    backbones = [PRIMARY_BACKBONE, *COMPARISON_BACKBONES]
    rows = {name: load_split_rows(repo_root, name) for name in ("train", "val", "test")}
    trained = []
    for backbone in backbones:
        result = train_one(
            backbone=backbone,
            train_rows=rows["train"],
            val_rows=rows["val"],
            image_size=settings["image_size"],
            batch_size=settings["batch_size"],
            epochs=settings["epochs"],
            freeze_epochs=settings["freeze_epochs"],
            patience=settings["patience"],
            learning_rate=settings["learning_rate"],
            fine_tune_lr=settings["fine_tune_lr"],
            weight_decay=settings["weight_decay"],
            seed=settings.get("seed", 42),
            device=device,
            checkpoint_dir=repo_root / "checkpoints" / "classification" / backbone,
            pretrained=True,
        )
        model = result.pop("_model")
        result.pop("_val_labels", None)
        result.pop("_val_probs", None)
        test_metrics = score_split(
            model, rows["test"], settings["image_size"], result["batch_size"], device, result["threshold_from_validation"]
        )
        errors = _error_lists(test_metrics)
        result["test"] = _strip_curves(test_metrics)
        result["test_errors"] = errors
        result["test_probabilities"] = {
            "image_ids": test_metrics["image_ids"],
            "labels": test_metrics["labels"],
            "probabilities": test_metrics["probabilities"],
        }
        del model
        release(result)
        trained.append(result)
        if current_rss_bytes() > 16 * 1024**3:
            raise MemoryError("RSS stayed above 16 GiB after releasing a model.")

    selected = max(
        trained,
        key=lambda item: (
            item["validation"].get("roc_auc") or -1,
            item["validation"]["sensitivity_malignant"],
            -item["peak_rss_bytes"],
        ),
    )
    selected_dir = repo_root / "checkpoints" / "classification" / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected["checkpoint"], selected_dir / "best.pt")
    pointer = {
        "backbone": selected["backbone"],
        "checkpoint": str(selected_dir / "best.pt"),
        "source_checkpoint": selected["checkpoint"],
        "image_size": selected["image_size"],
        "threshold": selected["threshold_from_validation"],
        "class_names": CLASS_NAMES,
        "selection_rule": "highest validation ROC-AUC, then malignant sensitivity, then lower peak RSS. Test labels were not used.",
    }
    (selected_dir / "selected.json").write_text(json.dumps(pointer, indent=2))
    payload = {
        "hardware_before": snapshot,
        "hardware_after_rss_bytes": current_rss_bytes(),
        "device": str(device),
        "split": {key: value for key, value in split.items() if key != "splits"},
        "settings": {key: value for key, value in settings.items() if key != "config"},
        "models": [_jsonable(item) for item in trained],
        "selected": pointer,
        "disclaimer": "Research classification of folder labels. Not a clinical diagnosis and not a BI-RADS assessment.",
    }
    out = repo_root / "reports" / "classification_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2))
    _write_plots(payload, repo_root / "reports" / "figures" / "classification")
    _write_markdown(payload, repo_root / "reports" / "classification_evaluation.md")
    return payload


def _error_lists(metrics: dict) -> dict:
    threshold = metrics["threshold"]
    false_negatives = []
    false_positives = []
    for image_id, label, probability in zip(metrics["image_ids"], metrics["labels"], metrics["probabilities"]):
        predicted = int(probability >= threshold)
        if label == 1 and predicted == 0:
            false_negatives.append({"image_id": image_id, "probability_malignant": probability})
        if label == 0 and predicted == 1:
            false_positives.append({"image_id": image_id, "probability_malignant": probability})
    return {"false_negatives": false_negatives, "false_positives": false_positives}


def _write_plots(payload: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    names = [item["backbone"] for item in payload["models"]]
    val_auc = [item["validation"].get("roc_auc") or 0 for item in payload["models"]]
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(names, val_auc, color="#4C78A8")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Validation ROC-AUC")
    axis.set_title("Model comparison on the validation set")
    fig.tight_layout()
    fig.savefig(directory / "validation_roc_auc.png", dpi=120)
    plt.close(fig)

    selected_name = payload["selected"]["backbone"]
    selected = next(item for item in payload["models"] if item["backbone"] == selected_name)
    matrix = np.array(selected["test"]["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(4, 3.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], CLASS_NAMES)
    axis.set_yticks([0, 1], CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Folder label")
    for row in range(2):
        for col in range(2):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=axis, fraction=0.046)
    axis.set_title(f"{selected_name} test confusion")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5, 3.5))
    axis.plot(selected["history"]["train_loss"], label="train")
    axis.plot(selected["history"]["val_loss"], label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.set_title(f"{selected_name} loss")
    fig.tight_layout()
    fig.savefig(directory / "loss_curve.png", dpi=120)
    plt.close(fig)


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Classification evaluation",
        "",
        payload["disclaimer"],
        "",
        f"Device: `{payload['device']}`.",
        "",
        "The selected model is the one with the highest validation ROC-AUC. Test numbers are reported after that choice.",
        "",
        f"Selected backbone: **{payload['selected']['backbone']}**.",
        f"Validation threshold frozen for test: {payload['selected']['threshold']}.",
        "",
        "## Validation",
        "",
        "| Model | ROC-AUC | Malignant sensitivity | Benign specificity | Epochs | Peak RSS GiB |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload["models"]:
        val = item["validation"]
        auc = val.get("roc_auc")
        auc_text = "n/a" if auc is None else f"{auc:.3f}"
        lines.append(
            f"| {item['backbone']} | {auc_text} | {val['sensitivity_malignant']:.3f} | "
            f"{val['specificity_benign']:.3f} | {item['epochs_ran']} | {item['peak_rss_bytes'] / 1024**3:.2f} |"
        )
    selected = next(item for item in payload["models"] if item["backbone"] == payload["selected"]["backbone"])
    test = selected["test"]
    lines += [
        "",
        "## Selected model on the untouched test split",
        "",
        f"- Malignant sensitivity: {test['sensitivity_malignant']:.3f}",
        f"- Benign specificity: {test['specificity_benign']:.3f}",
        f"- Precision (malignant): {test['precision_malignant']:.3f}",
        f"- F1 (malignant): {test['f1_malignant']:.3f}",
        f"- Macro-F1: {test['macro_f1']:.3f}",
        f"- ROC-AUC: {test.get('roc_auc')}",
        f"- PR-AUC: {test.get('pr_auc')}",
        f"- Brier score: {test['brier']:.3f}",
        f"- Confusion matrix (rows true benign/malignant): {test['confusion_matrix']}",
        "",
        "Patient identifiers are not in this dataset. Groups are exact duplicates and near-duplicates. "
        f"Conflict groups held out: {payload['split']['n_conflict_groups']}.",
        "",
        "These figures describe folder labels on this split. They are not a BI-RADS assessment and not a clinical claim.",
        "",
    ]
    path.write_text("\n".join(lines))


def benchmark_selected(repo_root: Path, repeats: int = 20) -> dict:
    """Load the selected classifier alone and time a batch of one image."""
    import time

    from classification.models import build_image_classifier
    from utils.hardware import current_rss_bytes
    from utils.seed import get_device

    pointer = json.loads((Path(repo_root) / "checkpoints" / "classification" / "selected" / "selected.json").read_text())
    checkpoint = Path(pointer["checkpoint"])
    rss_before = current_rss_bytes()
    device = get_device(None)
    model = build_image_classifier(pointer["backbone"], num_classes=2, pretrained=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    rss_loaded = current_rss_bytes()
    sample = torch.zeros(1, 3, pointer["image_size"], pointer["image_size"], device=device)
    with torch.no_grad():
        model(sample)
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            model(sample)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    rss_after = current_rss_bytes()
    size_bytes = checkpoint.stat().st_size
    del model, sample
    release({})
    return {
        "backbone": pointer["backbone"],
        "checkpoint_bytes": size_bytes,
        "rss_before_bytes": rss_before,
        "rss_loaded_bytes": rss_loaded,
        "rss_after_inference_bytes": rss_after,
        "repeats": repeats,
        "seconds_per_image": elapsed / repeats,
        "within_8gb_checkpoint": size_bytes <= 8 * 1000**3,
        "within_18gib_process": rss_after < 18 * 1024**3,
    }


def predict_image(repo_root: Path, image_path: Path) -> dict:
    """Load only the selected checkpoint and score one image."""
    from classification.data import letterbox
    from classification.models import build_image_classifier
    from preprocessing.normalization import imagenet_normalize, load_image_rgb, to_unit_range

    pointer = json.loads((Path(repo_root) / "checkpoints" / "classification" / "selected" / "selected.json").read_text())
    device = get_device(None)
    model = build_image_classifier(pointer["backbone"], num_classes=2, pretrained=False).to(device)
    state = torch.load(pointer["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    image = letterbox(load_image_rgb(image_path), pointer["image_size"])
    tensor = torch.from_numpy(imagenet_normalize(to_unit_range(image)).transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    index = int(probability.argmax())
    del model, tensor
    return {
        "image": str(image_path),
        "predicted_class": CLASS_NAMES[index],
        "probability_malignant": float(probability[1]),
        "threshold": pointer["threshold"],
        "above_validation_threshold": bool(probability[1] >= pointer["threshold"]),
        "backbone": pointer["backbone"],
        "disclaimer": "Research output from a folder-label classifier. Not a diagnosis.",
    }
