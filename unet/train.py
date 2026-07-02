"""Stage 3 training entry point: U-Net tumor segmentation.

Not executed automatically anywhere in this repo — run manually, e.g.:

    python train.py unet
    python train.py unet --overrides configs/unet_attention_resnet34.yaml
    python train.py unet --resume
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim

from evaluation.metrics import dice_score
from unet.dataset import build_unet_dataloader
from unet.losses import DiceBCELoss
from unet.model import build_unet
from utils.logging import get_logger
from utils.seed import get_device, set_seed
from visualization.plots import plot_training_curves

logger = get_logger("unet_train")


def _build_scheduler(optimizer: optim.Optimizer, unet_cfg: dict):
    sched_cfg = unet_cfg["scheduler"]
    if sched_cfg["type"] == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=sched_cfg["factor"], patience=sched_cfg["patience"]
        )
    if sched_cfg["type"] == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=unet_cfg["epochs"])
    raise ValueError(f"Unknown scheduler type: {sched_cfg['type']}")


def _run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, float]:
    """One train (optimizer given) or val (optimizer=None) epoch. Returns (loss, dice)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_dice += dice_score(preds.cpu().numpy(), masks.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches, total_dice / n_batches


def train_unet(config: dict, resume: bool = False) -> nn.Module:
    """Train Stage 3's U-Net variant selected by ``config["unet"]``."""
    set_seed(config.get("seed", 42), config.get("deterministic", True))
    unet_cfg = config["unet"]
    device = get_device(config.get("device"))
    use_amp = unet_cfg["amp"] and device.type == "cuda"

    logger.info(
        "Training %s (encoder=%s) on device=%s", unet_cfg["architecture"], unet_cfg["encoder"], device
    )

    model = build_unet(
        architecture=unet_cfg["architecture"],
        encoder_name=unet_cfg["encoder"],
        encoder_weights=unet_cfg["encoder_weights"],
        in_channels=unet_cfg["in_channels"],
        classes=unet_cfg["classes"],
    ).to(device)

    train_loader = build_unet_dataloader(config, "train")
    val_loader = build_unet_dataloader(config, "val")

    criterion = DiceBCELoss(**unet_cfg["loss"])
    optimizer = optim.Adam(model.parameters(), lr=unet_cfg["lr"], weight_decay=unet_cfg["weight_decay"])
    scheduler = _build_scheduler(optimizer, unet_cfg)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    checkpoint_dir = Path(unet_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / unet_cfg["checkpoint_name"]
    last_path = checkpoint_dir / f"last_{unet_cfg['checkpoint_name']}"

    start_epoch = 0
    best_val_dice = -1.0
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_dice = checkpoint["best_val_dice"]
        history = checkpoint["history"]
        logger.info("Resumed from epoch %d (best val dice so far: %.4f)", start_epoch, best_val_dice)

    for epoch in range(start_epoch, unet_cfg["epochs"]):
        train_loss, train_dice = _run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss, val_dice = _run_epoch(model, val_loader, criterion, device, None, scaler, use_amp)

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_dice)
        else:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        logger.info(
            "epoch %d/%d | train_loss=%.4f train_dice=%.4f | val_loss=%.4f val_dice=%.4f",
            epoch + 1, unet_cfg["epochs"], train_loss, train_dice, val_loss, val_dice,
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_val_dice,
                "history": history,
                "config": unet_cfg,
            },
            last_path,
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "config": unet_cfg}, best_path)
            logger.info("New best val dice %.4f -> saved %s", best_val_dice, best_path)
        else:
            patience_counter += 1
            if patience_counter >= unet_cfg["patience"]:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, unet_cfg["patience"])
                break

    plot_training_curves(
        {"train_loss": history["train_loss"], "val_loss": history["val_loss"]},
        title=f"U-Net ({unet_cfg['architecture']}/{unet_cfg['encoder']}) loss",
        save_path=Path(config["paths"]["results"]) / "unet" / "training_loss_curve.png",
    )
    plot_training_curves(
        {"train_dice": history["train_dice"], "val_dice": history["val_dice"]},
        title=f"U-Net ({unet_cfg['architecture']}/{unet_cfg['encoder']}) dice",
        save_path=Path(config["paths"]["results"]) / "unet" / "training_dice_curve.png",
    )

    logger.info("Training complete. Best checkpoint: %s (val_dice=%.4f)", best_path, best_val_dice)
    return model


if __name__ == "__main__":
    import argparse

    from utils.config import load_config

    parser = argparse.ArgumentParser(description="Train a Stage 3 U-Net variant")
    parser.add_argument("--overrides", default=None, help="Optional config override YAML")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(overrides=args.overrides)
    train_unet(cfg, resume=args.resume)
