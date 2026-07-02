#!/usr/bin/env python
"""Root CLI entry point for training every stage of the Breast CAD pipeline.

    python train.py preprocess                    # Stage 1 only (no training)
    python train.py yolo                           # Stage 2
    python train.py unet                            # Stage 3
    python train.py resnet                          # Stage 4
    python train.py all                              # Stage 1 -> 2 -> 3 -> 4
    python train.py all --overrides configs/quick_run.yaml
    python train.py yolo --resume

This is a thin dispatcher: all real logic lives in ``preprocessing/``,
``yolo/``, ``unet/``, ``resnet/`` — this script just wires config + CLI args
to those modules' ``train_*`` functions.

IMPORTANT: this script trains real models when invoked with the yolo/unet/
resnet/all stages. It must always be run manually by a human, never invoked
automatically by an agent/notebook cell/CI job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset.splits import apply_quick_run_subsample, load_split
from preprocessing.pipeline import run_stage1
from utils.config import load_config
from utils.logging import get_logger
from yolo.dataset_builder import build_yolo_dataset

logger = get_logger("train")


def ensure_stage1(config: dict) -> None:
    """Run Stage 1 preprocessing if the split manifests don't exist yet."""
    splits_dir = Path(config["paths"]["splits_dir"])
    if (splits_dir / "train.json").exists():
        return
    logger.info("No existing split manifests found — running Stage 1 preprocessing first...")
    run_stage1(config)


def ensure_yolo_dataset(config: dict) -> Path:
    """Build the Ultralytics-format YOLO dataset if it doesn't exist yet."""
    data_yaml = Path(config["paths"]["yolo_dataset_dir"]) / "data.yaml"
    if data_yaml.exists():
        return data_yaml
    logger.info("No existing YOLO dataset found — building it from the split manifests...")
    splits = {
        name: apply_quick_run_subsample(load_split(config["paths"]["splits_dir"], name), config)
        for name in ("train", "val", "test")
    }
    return build_yolo_dataset(
        splits,
        config["paths"]["yolo_dataset_dir"],
        min_area_px=config["preprocessing"]["min_mask_area_px"],
        bbox_pad_ratio=config["preprocessing"]["bbox_pad_ratio"],
        class_names=tuple(config["yolo"]["class_names"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Breast CAD pipeline")
    parser.add_argument("stage", choices=["preprocess", "yolo", "unet", "resnet", "all"])
    parser.add_argument("--overrides", default=None, help="Optional config override YAML")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(overrides=args.overrides)

    if args.stage in ("preprocess", "all"):
        ensure_stage1(config)

    if args.stage in ("yolo", "all"):
        from yolo.train import train_yolo

        data_yaml = ensure_yolo_dataset(config)
        train_yolo(config, data_yaml, resume=args.resume)

    if args.stage in ("unet", "all"):
        from unet.train import train_unet

        ensure_stage1(config)
        train_unet(config, resume=args.resume)

    if args.stage in ("resnet", "all"):
        from resnet.train import train_resnet

        ensure_stage1(config)
        train_resnet(config, resume=args.resume)


if __name__ == "__main__":
    main()
