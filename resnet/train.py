"""Stage 4 training entry point: ResNet50 benign/malignant classification.

Not executed automatically anywhere in this repo — run manually, e.g.:

    python train.py resnet
    python train.py resnet --overrides configs/quick_run.yaml   # smoke test
    python train.py resnet --resume

Two-phase transfer learning: ``freeze_backbone_epochs`` epochs of linear-probe
training (only the new FC head trained, backbone frozen) followed by full
fine-tuning at a lower learning rate — the standard, low-overfitting-risk
recipe for a dataset as small as BUSI.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim

from resnet.dataset import build_resnet_dataloader
from resnet.model import build_resnet50, set_backbone_trainable
from utils.logging import get_logger
from utils.seed import get_device, set_seed
from visualization.plots import plot_training_curves

logger = get_logger("resnet_train")


def _run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, float]:
    """One train (optimizer given) or val (optimizer=None) epoch. Returns (loss, accuracy)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_n = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_n += labels.size(0)

    return total_loss / total_n, total_correct / total_n


def train_resnet(config: dict, resume: bool = False) -> nn.Module:
    """Train Stage 4's ResNet50 classifier with a freeze-then-fine-tune schedule."""
    set_seed(config.get("seed", 42), config.get("deterministic", True))
    resnet_cfg = config["resnet"]
    device = get_device(config.get("device"))
    use_amp = resnet_cfg["amp"] and device.type == "cuda"

    logger.info("Training ResNet50 on device=%s", device)

    model = build_resnet50(
        pretrained=resnet_cfg["pretrained"],
        num_classes=resnet_cfg["num_classes"],
        freeze_backbone=(resnet_cfg["freeze_backbone_epochs"] > 0),
    ).to(device)

    train_loader = build_resnet_dataloader(config, "train")
    val_loader = build_resnet_dataloader(config, "val")

    criterion = nn.CrossEntropyLoss(label_smoothing=resnet_cfg["label_smoothing"])
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    checkpoint_dir = Path(resnet_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / resnet_cfg["checkpoint_name"]
    last_path = checkpoint_dir / f"last_{resnet_cfg['checkpoint_name']}"

    start_epoch = 0
    best_val_acc = -1.0
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    checkpoint = None
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        history = checkpoint["history"]

    # Rebuild the optimizer/scheduler to match the freeze/unfreeze state at
    # start_epoch *before* loading optimizer_state, since a checkpoint saved
    # after the unfreeze boundary has full-model parameter groups rather than
    # the frozen-backbone (FC-head-only) groups the model starts with.
    backbone_frozen = resnet_cfg["freeze_backbone_epochs"] > 0 and start_epoch < resnet_cfg["freeze_backbone_epochs"]
    if resnet_cfg["freeze_backbone_epochs"] > 0 and not backbone_frozen:
        set_backbone_trainable(model, True)
        optimizer = optim.Adam(
            model.parameters(), lr=resnet_cfg["fine_tune_lr"], weight_decay=resnet_cfg["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=resnet_cfg["epochs"] - resnet_cfg["freeze_backbone_epochs"]
        )
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=resnet_cfg["lr"],
            weight_decay=resnet_cfg["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=resnet_cfg["epochs"])

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info("Resumed from epoch %d (best val acc so far: %.4f)", start_epoch, best_val_acc)

    for epoch in range(start_epoch, resnet_cfg["epochs"]):
        if backbone_frozen and epoch >= resnet_cfg["freeze_backbone_epochs"]:
            logger.info("Unfreezing backbone at epoch %d, switching to fine_tune_lr", epoch + 1)
            set_backbone_trainable(model, True)
            optimizer = optim.Adam(
                model.parameters(), lr=resnet_cfg["fine_tune_lr"], weight_decay=resnet_cfg["weight_decay"]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=resnet_cfg["epochs"] - epoch
            )
            backbone_frozen = False

        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, None, scaler, use_amp)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            "epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch + 1, resnet_cfg["epochs"], train_loss, train_acc, val_loss, val_acc,
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "history": history,
                "config": resnet_cfg,
            },
            last_path,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "config": resnet_cfg}, best_path)
            logger.info("New best val acc %.4f -> saved %s", best_val_acc, best_path)
        else:
            patience_counter += 1
            if patience_counter >= resnet_cfg["patience"]:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, resnet_cfg["patience"])
                break

    plot_training_curves(
        {"train_loss": history["train_loss"], "val_loss": history["val_loss"]},
        title="ResNet50 loss",
        save_path=Path(config["paths"]["results"]) / "resnet" / "training_loss_curve.png",
    )
    plot_training_curves(
        {"train_acc": history["train_acc"], "val_acc": history["val_acc"]},
        title="ResNet50 accuracy",
        save_path=Path(config["paths"]["results"]) / "resnet" / "training_acc_curve.png",
    )

    logger.info("Training complete. Best checkpoint: %s (val_acc=%.4f)", best_path, best_val_acc)
    return model


if __name__ == "__main__":
    import argparse

    from utils.config import load_config

    parser = argparse.ArgumentParser(description="Train the Stage 4 ResNet50 classifier")
    parser.add_argument("--overrides", default=None, help="Optional config override YAML")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(overrides=args.overrides)
    train_resnet(cfg, resume=args.resume)
