"""Sanity-check the lexicon thresholds and rule-based categories on a held-out split.

    python -m reporting.calibration --split val

Runs features -> lexicon -> rules on ground-truth masks (lexicon-only, and
with the trained ResNet50's malignant probability) and writes per-category
benign/malignant counts to ``results/reporting/calibration_<split>.json``.
This is a consistency check of the heuristics, not a clinical validation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dataset.lesion_manifest import load_lesion_manifest
from preprocessing.mask_to_yolo import component_mask
from preprocessing.normalization import load_image_rgb
from preprocessing.roi_utils import apply_mask, crop_roi_pair
from reporting.features import extract_features
from reporting.lexicon import map_to_lexicon
from reporting.scoring import CATEGORY_ORDER, assess
from utils.config import load_config
from utils.seed import get_device


def _load_resnet(config: dict, device: torch.device):
    from inference.pipeline import to_chw_tensor
    from preprocessing.normalization import to_unit_range
    from resnet.model import build_resnet50

    model = build_resnet50(pretrained=False, num_classes=config["resnet"]["num_classes"]).to(device)
    state = torch.load(config["inference"]["resnet_weights"], map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    def p_malignant(masked_roi: np.ndarray) -> float:
        with torch.no_grad():
            logits = model(to_chw_tensor(to_unit_range(masked_roi), device))
        return float(torch.softmax(logits, dim=1)[0, 1])

    return p_malignant


def run_calibration(config: dict, split: str = "val") -> dict:
    """Features -> lexicon -> rules on ``split``'s ground-truth masks; returns (and saves) the tallies."""
    roi_size = tuple(config["preprocessing"]["roi_size"])
    rows = load_lesion_manifest(Path(config["paths"]["splits_dir"]) / f"{split}_lesions.json")
    p_fn = _load_resnet(config, get_device(config.get("device")))

    tallies = {"lexicon_only": Counter(), "with_classifier": Counter()}
    descriptor_tally: Counter = Counter()
    for row in rows:
        image = load_image_rgb(row["image_path"])
        mask = component_mask(row["source_mask_path"], row["component_label"])
        roi, roi_mask, padded = crop_roi_pair(image, mask, tuple(row["bbox"]), pad_ratio=0.0, target_size=roi_size)
        features = extract_features(image, padded, roi_mask)
        descriptors = map_to_lexicon(features)
        for d in descriptors:
            if d.assessable:
                descriptor_tally[(d.key, d.value, row["label"])] += 1
        tallies["lexicon_only"][(assess(descriptors, features, None).category, row["label"])] += 1
        p = p_fn(apply_mask(roi, roi_mask))
        tallies["with_classifier"][(assess(descriptors, features, p).category, row["label"])] += 1

    summary: dict = {"split": split, "n_lesions": len(rows)}
    for mode, counter in tallies.items():
        summary[mode] = {
            cat: {"benign": counter[(cat, "benign")], "malignant": counter[(cat, "malignant")]}
            for cat in CATEGORY_ORDER if counter[(cat, "benign")] + counter[(cat, "malignant")]
        }
    summary["descriptors"] = {
        f"{key}={value}": {"benign": descriptor_tally[(key, value, "benign")],
                           "malignant": descriptor_tally[(key, value, "malignant")]}
        for key, value in sorted({(k, v) for k, v, _ in descriptor_tally})
    }

    out_path = Path(config["paths"]["results"]) / "reporting" / f"calibration_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    summary = run_calibration(load_config(), args.split)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
