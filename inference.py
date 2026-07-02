#!/usr/bin/env python
"""Root CLI entry point: non-interactive batch/script inference (Stage 2->3->4).

    python inference.py --input path/to/image.png --output results/inference/
    python inference.py --input path/to/folder/ --output results/inference/

Thin wrapper: all cascade logic lives in ``inference/pipeline.py`` (imported
here and by ``app/app.py`` — never duplicated). No physician
accept/reject steps; every stage always uses the model's own prediction. For
clinician-in-the-loop review, use ``app/app.py`` instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.pipeline import build_pipeline
from inference.visualize import plot_case_result
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger("inference_cli")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def run_batch_inference(input_path: str | Path, output_dir: str | Path, config: dict) -> list[dict]:
    """Run the full cascade over every image at ``input_path`` (file or directory)."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(config)
    images = _collect_images(input_path)
    logger.info("Running inference on %d image(s)", len(images))

    summaries = []
    for image_path in images:
        try:
            result = pipeline.run(image_path)
        except RuntimeError as exc:
            logger.warning("Skipping %s: %s", image_path.name, exc)
            continue

        case_id = image_path.stem
        plot_case_result(result, save_path=output_dir / f"{case_id}_report.png")

        summary = {"image": str(image_path), **result.to_dict()}
        with open(output_dir / f"{case_id}_result.json", "w") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)

        logger.info("%s -> %s (%.1f%%)", case_id, result.predicted_class, result.confidence * 100)

    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference over the full Breast CAD cascade")
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument("--output", default="results/inference", help="Output directory")
    parser.add_argument("--overrides", default=None)
    args = parser.parse_args()

    config = load_config(overrides=args.overrides)
    run_batch_inference(args.input, args.output, config)


if __name__ == "__main__":
    main()
