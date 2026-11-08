"""Train one small classifier at a time, then drop it from memory.

Batch size starts at the configured value (8 or lower). If the process hits
the memory cap or the allocator raises an out-of-memory error, the batch is
halved and that model is trained again from the beginning.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from classification.data import class_weights, make_loader
from classification.metrics import binary_metrics, choose_threshold
from classification.models import build_image_classifier, freeze_backbone, parameter_counts
from utils.hardware import PEAK_PROCESS_LIMIT_BYTES, assert_process_budget, current_rss_bytes
from utils.seed import set_seed


def _empty_cache() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _forward_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses = []
    labels_all = []
    probs_all = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            probability = torch.softmax(logits.detach(), dim=1)[:, 1]
            losses.append(float(loss.item()) * labels.size(0))
            labels_all.append(labels.detach().cpu().numpy())
            probs_all.append(probability.cpu().numpy())
            del images, labels, logits, loss, probability
    count = sum(len(item) for item in labels_all)
    return (
        float(sum(losses) / max(count, 1)),
        np.concatenate(labels_all),
        np.concatenate(probs_all),
    )


def train_one(
    backbone: str,
    train_rows: list[dict],
    val_rows: list[dict],
    image_size: int,
    batch_size: int,
    epochs: int,
    freeze_epochs: int,
    patience: int,
    learning_rate: float,
    fine_tune_lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path,
    pretrained: bool = True,
) -> dict:
    """Train ``backbone`` and return validation probabilities for threshold selection."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    batch = batch_size
    attempt_notes = []
    while batch >= 1:
        try:
            return _train_attempt(
                backbone, train_rows, val_rows, image_size, batch, epochs, freeze_epochs, patience,
                learning_rate, fine_tune_lr, weight_decay, seed, device, best_path, pretrained, attempt_notes,
            )
        except (MemoryError, RuntimeError) as exc:
            message = str(exc).lower()
            if batch == 1 or not any(token in message for token in ("memory", "mps", "out of memory")):
                raise
            attempt_notes.append(f"batch {batch} failed: {exc}")
            batch = max(1, batch // 2)
            _empty_cache()
    raise RuntimeError("training failed after batch-size reductions")


def _train_attempt(
    backbone, train_rows, val_rows, image_size, batch_size, epochs, freeze_epochs, patience,
    learning_rate, fine_tune_lr, weight_decay, seed, device, best_path, pretrained, attempt_notes,
) -> dict:
    set_seed(seed, deterministic=True)
    model = build_image_classifier(backbone, num_classes=2, pretrained=pretrained).to(device)
    freeze_backbone(model, backbone, frozen=freeze_epochs > 0)
    counts = parameter_counts(model)
    weights = class_weights(train_rows).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    train_loader = make_loader(train_rows, image_size, batch_size, train=True, seed=seed)
    val_loader = make_loader(val_rows, image_size, batch_size, train=False, seed=seed)

    best_val_loss = float("inf")
    best_epoch = -1
    waited = 0
    history = {"train_loss": [], "val_loss": [], "val_balanced_accuracy": []}
    peak_rss = current_rss_bytes()
    started = time.perf_counter()
    frozen = freeze_epochs > 0

    for epoch in range(epochs):
        assert_process_budget()
        if frozen and epoch >= freeze_epochs:
            freeze_backbone(model, backbone, frozen=False)
            optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - epoch, 1))
            frozen = False
        train_loader.dataset.epoch = epoch
        train_loss, _, _ = _forward_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_labels, val_probs = _forward_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        balanced = binary_metrics(val_labels, val_probs, 0.5)
        balanced_accuracy = 0.5 * (
            balanced["sensitivity_malignant"] + balanced["specificity_benign"]
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_balanced_accuracy"].append(balanced_accuracy)
        peak_rss = max(peak_rss, current_rss_bytes())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            waited = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": backbone,
                    "image_size": image_size,
                    "class_names": ["benign", "malignant"],
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )
        else:
            waited += 1
            if waited >= patience:
                break
        if current_rss_bytes() > PEAK_PROCESS_LIMIT_BYTES * 0.9:
            raise MemoryError(f"RSS approached the cap during {backbone} epoch {epoch}")

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    _, val_labels, val_probs = _forward_epoch(model, val_loader, criterion, device, optimizer=None)
    threshold = choose_threshold(val_labels, val_probs)
    validation = binary_metrics(val_labels, val_probs, threshold)
    elapsed = time.perf_counter() - started
    result = {
        "backbone": backbone,
        "checkpoint": str(best_path),
        "parameters": counts,
        "batch_size": batch_size,
        "image_size": image_size,
        "epochs_ran": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "threshold_from_validation": threshold,
        "validation": {key: value for key, value in validation.items() if key != "roc_curve"},
        "history": history,
        "seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "attempt_notes": list(attempt_notes),
        "pretrained": pretrained,
    }
    # Keep probabilities only long enough for the caller to score the test set.
    result["_model"] = model
    result["_val_labels"] = val_labels
    result["_val_probs"] = val_probs
    return result


def score_split(model, rows, image_size, batch_size, device, threshold: float) -> dict:
    loader = make_loader(rows, image_size, batch_size, train=False, seed=0)
    model.eval()
    labels_all = []
    probs_all = []
    ids = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            probability = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels_all.append(batch["label"].numpy())
            probs_all.append(probability)
            ids.extend(batch["image_id"])
    labels = np.concatenate(labels_all)
    probs = np.concatenate(probs_all)
    metrics = binary_metrics(labels, probs, threshold)
    metrics["image_ids"] = ids
    metrics["probabilities"] = probs.tolist()
    metrics["labels"] = labels.tolist()
    return metrics


def release(result: dict) -> None:
    result.pop("_model", None)
    result.pop("_val_labels", None)
    result.pop("_val_probs", None)
    _empty_cache()
