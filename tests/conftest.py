"""Shared synthetic fixtures for the research-runtime tests (no dataset, checkpoints or network needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def synthetic_case():
    """A speckled 'ultrasound' image with a darker elliptical lesion and its aligned mask."""
    rng = np.random.RandomState(7)
    h, w = 160, 200
    image = rng.normal(120, 25, (h, w)).clip(0, 255)
    yy, xx = np.mgrid[:h, :w]
    lesion = ((xx - 100) / 40.0) ** 2 + ((yy - 70) / 25.0) ** 2 <= 1.0
    image[lesion] = rng.normal(55, 15, lesion.sum()).clip(0, 255)
    rgb = np.repeat(image.astype(np.uint8)[..., None], 3, axis=2)
    return rgb, lesion.astype(np.uint8), [50, 35, 150, 105]


@pytest.fixture
def settings():
    from bus_research.radiomics_extract import RadiomicsSettings

    return RadiomicsSettings()


@pytest.fixture
def make_record(synthetic_case, settings):
    from bus_research import birads
    from bus_research.feature_store import build_record
    from bus_research.radiomics_extract import extract_radiomics_features

    def _make(image=None, mask=None, key="case-1", lesion="1", labels=None, intensity_shift=0):
        img, msk, roi = synthetic_case
        img = img if image is None else image
        msk = msk if mask is None else mask
        if intensity_shift:
            img = np.clip(img.astype(int) + intensity_shift, 0, 255).astype(np.uint8)
        result = extract_radiomics_features(img, msk, {"roi_bbox": roi}, settings)
        return build_record(
            image_key=key, lesion_key=lesion, image=img, radiomics_result=result,
            segmentation={"model": "unet", "checkpoint_version": None, "mask_source": "predicted",
                          "activation": "sigmoid", "threshold": 0.5, "bbox_source": "yolo",
                          "detection_confidence": 0.9, "roi_bbox_xyxy": roi},
            descriptors=birads.empty_descriptors(), reference_labels=labels,
        )

    return _make
