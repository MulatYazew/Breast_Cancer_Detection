"""Stage 1 orchestrator: raw BUSI_Jpeg -> validated, split, ready-to-train dataset.

Chains together every other ``preprocessing``/``dataset`` module:
manifest discovery -> validation -> stratified split -> per-lesion manifests
-> sanity-check visualizations -> preprocessing report. Produces no trained
models; safe to run in full any time the raw data or split config changes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from dataset.lesion_manifest import build_lesion_manifest, save_lesion_manifest
from dataset.manifest import ManifestRecord, build_manifest
from dataset.splits import save_splits, split_class_counts, stratified_split
from preprocessing.mask_to_yolo import component_mask, image_lesion_components
from preprocessing.normalization import load_image_rgb
from preprocessing.report import build_preprocessing_report
from preprocessing.validation import validate_dataset
from utils.logging import get_logger

logger = get_logger("preprocessing")


def run_stage1(config: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """Run the full Stage 1 pipeline and return a summary dict.

    Side effects (all under paths from ``config["paths"]`` / ``results/``):
        - ``dataset/splits/{train,val,test}.json``: image-level split manifests.
        - ``dataset/splits/{train,val,test}_lesions.json``: per-lesion manifests.
        - ``results/preprocessing/report.json``: the preprocessing report.
        - ``results/preprocessing/*.png``: mask sanity-check and bbox overlay figures.
    """
    paths = config["paths"]
    prep_cfg = config["preprocessing"]
    split_cfg = config["split"]
    results_dir = Path(paths["results"]) / "preprocessing"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building manifest from raw BUSI_Jpeg directories...")
    records = build_manifest(
        paths["raw_benign"], paths["raw_benign_mask"], paths["raw_malignant"], paths["raw_malignant_mask"]
    )
    logger.info("Found %d images.", len(records))

    logger.info("Validating dataset (pairing, corruption, duplicates, dimensions, masks)...")
    validation_report = validate_dataset(records, stop_on_critical=True)
    logger.info(
        "%d critical issue(s), %d warning(s).",
        len(validation_report.critical_issues),
        len(validation_report.warnings),
    )
    for issue in validation_report.warnings:
        logger.warning("[%s] %s: %s", issue.kind, issue.image_id, issue.detail)

    logger.info("Splitting dataset (stratified 70/15/15, seed=%d)...", split_cfg["seed"])
    splits = stratified_split(
        records,
        train_frac=split_cfg["train"],
        val_frac=split_cfg["val"],
        test_frac=split_cfg["test"],
        seed=split_cfg["seed"],
    )
    save_splits(splits, paths["splits_dir"])
    logger.info("Split sizes: %s", {k: len(v) for k, v in splits.items()})
    logger.info("Split class counts: %s", split_class_counts(splits))

    logger.info("Building per-lesion manifests...")
    lesion_manifests: dict[str, list[dict]] = {}
    for split_name, split_records in splits.items():
        rows = build_lesion_manifest(
            split_records,
            min_area_px=prep_cfg["min_mask_area_px"],
            bbox_pad_ratio=prep_cfg["bbox_pad_ratio"],
        )
        save_lesion_manifest(rows, Path(paths["splits_dir"]) / f"{split_name}_lesions.json")
        lesion_manifests[split_name] = rows
        logger.info("  %s: %d lesion instances from %d images", split_name, len(rows), len(split_records))

    logger.info("Rendering sanity-check visualizations...")
    _visualize_mask_samples(records, results_dir / "mask_sanity_check.png", seed=seed)
    _visualize_bbox_samples(records, prep_cfg, results_dir / "bbox_overlay_check.png", seed=seed)

    report = build_preprocessing_report(
        records, validation_report, splits, save_path=results_dir / "report.json"
    )
    logger.info("Preprocessing report saved to %s", results_dir / "report.json")

    return {
        "records": records,
        "validation_report": validation_report,
        "splits": splits,
        "lesion_manifests": lesion_manifests,
        "report": report,
    }


def _visualize_mask_samples(
    records: list[ManifestRecord], save_path: Path, n_samples: int = 4, seed: int = 42
) -> None:
    from visualization.plots import plot_sample_grid

    rng = random.Random(seed)
    candidates = [r for r in records if r.mask_paths]
    sample_records = rng.sample(candidates, min(n_samples, len(candidates)))

    samples = []
    for record in sample_records:
        image = load_image_rgb(record.image_path)
        mask = component_mask(str(record.mask_paths[0]), 1)
        samples.append((image, mask, f"{record.label}: {record.image_id}"))

    plot_sample_grid(samples, save_path=save_path)


def _visualize_bbox_samples(
    records: list[ManifestRecord],
    prep_cfg: dict[str, Any],
    save_path: Path,
    n_samples: int = 8,
    seed: int = 42,
) -> None:
    from visualization.plots import plot_bbox_overlay

    rng = random.Random(seed)
    candidates = [r for r in records if r.mask_paths]
    sample_records = rng.sample(candidates, min(n_samples, len(candidates)))

    images_with_boxes = []
    for record in sample_records:
        image = load_image_rgb(record.image_path)
        components = image_lesion_components(
            record.mask_paths,
            min_area_px=prep_cfg["min_mask_area_px"],
            pad_ratio=prep_cfg["bbox_pad_ratio"],
        )
        images_with_boxes.append((image, [c.bbox for c in components]))

    plot_bbox_overlay(images_with_boxes, save_path=save_path)
