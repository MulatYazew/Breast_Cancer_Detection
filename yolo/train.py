"""Stage 2 training entry point: YOLO26n tumor detection.

Not executed automatically anywhere in this repo — run manually, e.g.:

    python train.py yolo
    python train.py yolo --overrides configs/quick_run.yaml   # smoke test
    python train.py yolo --resume                              # resume last run

Ultralytics' own ``Trainer`` provides mixed precision (``amp``), early
stopping (``patience``), LR scheduling, and automatic best/last checkpointing
under ``<project>/<name>/weights/{best,last}.pt`` — this module just wires
our config schema to those arguments rather than re-implementing them.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from utils.config import REPO_ROOT
from utils.logging import get_logger
from utils.seed import get_device, set_seed
from yolo.model import build_yolo_model

logger = get_logger("yolo_train")


def train_yolo(config: dict, data_yaml: str | Path, resume: bool = False) -> YOLO:
    """Train YOLO26n on the Ultralytics-format dataset at ``data_yaml``.

    Args:
        config: Full merged config (see ``utils.config.load_config``).
        data_yaml: Path to the dataset yaml produced by
            ``yolo.dataset_builder.build_yolo_dataset``.
        resume: If True, resume the last run in ``<project>/<name>`` instead
            of starting fresh.
    """
    set_seed(config.get("seed", 42), config.get("deterministic", True))
    yolo_cfg = config["yolo"]
    device = get_device(config.get("device"))

    logger.info("Training YOLO26n on device=%s, data=%s", device, data_yaml)
    model = build_yolo_model(yolo_cfg["weights"])
    # Ultralytics resolves a relative ``project`` against its global ``runs_dir``
    # setting (not the cwd), so anchor it to the repo root explicitly.
    project_dir = Path(yolo_cfg["project"])
    if not project_dir.is_absolute():
        project_dir = REPO_ROOT / project_dir

    model.train(
        data=str(data_yaml),
        epochs=yolo_cfg["epochs"],
        imgsz=yolo_cfg["imgsz"],
        batch=yolo_cfg["batch"],
        patience=yolo_cfg["patience"],
        lr0=yolo_cfg["lr0"],
        optimizer=yolo_cfg["optimizer"],
        amp=yolo_cfg["amp"],
        workers=yolo_cfg["workers"],
        project=str(project_dir),
        name=yolo_cfg["name"],
        resume=resume or yolo_cfg.get("resume", False),
        device=str(device),
        seed=config.get("seed", 42),
        deterministic=config.get("deterministic", True),
        exist_ok=True,
    )

    best_path = project_dir / yolo_cfg["name"] / "weights" / "best.pt"
    logger.info("Training complete. Best checkpoint: %s", best_path)
    return model


if __name__ == "__main__":
    import argparse

    from utils.config import load_config

    parser = argparse.ArgumentParser(description="Train YOLO26n tumor detector (Stage 2)")
    parser.add_argument("--data-yaml", default="dataset/yolo_dataset/data.yaml")
    parser.add_argument("--overrides", default=None, help="Optional config override YAML")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(overrides=args.overrides)
    train_yolo(cfg, args.data_yaml, resume=args.resume)
