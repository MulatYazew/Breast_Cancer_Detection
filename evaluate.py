#!/usr/bin/env python
"""Root CLI entry point for evaluating trained checkpoints from every stage.

    python evaluate.py yolo --weights checkpoints/yolo/busi_tumor/weights/best.pt
    python evaluate.py unet --checkpoint checkpoints/unet/best_unet.pth
    python evaluate.py resnet --checkpoint checkpoints/resnet/best_resnet.pth
    python evaluate.py cascade                     # full Stage 2->3->4 pipeline

Every path here assumes forward-pass-only evaluation against an
already-trained checkpoint — safe to run any time, never trains anything.
"""

from __future__ import annotations

import argparse

from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Breast CAD checkpoints")
    parser.add_argument("stage", choices=["yolo", "unet", "resnet", "cascade"])
    parser.add_argument("--weights", help="YOLO checkpoint path (stage=yolo)")
    parser.add_argument("--checkpoint", help="U-Net/ResNet checkpoint path (stage=unet/resnet)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--overrides", default=None)
    args = parser.parse_args()

    config = load_config(overrides=args.overrides)

    if args.stage == "yolo":
        from yolo.evaluate import evaluate_yolo

        weights = args.weights or config["inference"]["yolo_weights"]
        data_yaml = f"{config['paths']['yolo_dataset_dir']}/data.yaml"
        evaluate_yolo(
            weights,
            data_yaml,
            imgsz=config["yolo"]["imgsz"],
            split=args.split,
            conf_threshold=config["inference"]["conf_threshold"],
        )

    elif args.stage == "unet":
        from unet.evaluate import evaluate_unet

        checkpoint = args.checkpoint or config["inference"]["unet_weights"]
        evaluate_unet(config, checkpoint, split=args.split)

    elif args.stage == "resnet":
        from resnet.evaluate import evaluate_resnet

        checkpoint = args.checkpoint or config["inference"]["resnet_weights"]
        evaluate_resnet(config, checkpoint, split=args.split)

    elif args.stage == "cascade":
        from evaluation.cascade import evaluate_cascade

        evaluate_cascade(config, split=args.split)


if __name__ == "__main__":
    main()
