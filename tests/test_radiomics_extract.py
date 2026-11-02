from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("radiomics")

from bus_research.radiomics_extract import (  # noqa: E402
    FEATURE_CLASSES,
    _coerce,
    binarize_unet_output,
    extract_radiomics_features,
    flat_features,
    native_mask_from_roi,
    validate_mask,
)


def test_extracts_every_family_on_synthetic_lesion(synthetic_case, settings):
    image, mask, roi = synthetic_case
    result = extract_radiomics_features(image, mask, {"roi_bbox": roi}, settings)
    assert result["status"] == "ok", result["errors"]
    assert result["qc"]["status"] == "pass"
    assert set(result["features"]) == set(FEATURE_CLASSES)
    assert all(result["features"][f] for f in FEATURE_CLASSES)
    assert result["unit_system"] == "pixel"
    assert result["preprocessing"]["resampling"] == "none"
    # Shape is measured on the native mask: area within ~2% of the drawn ellipse.
    assert abs(result["features"]["shape2D"]["PixelSurface"] - mask.sum()) / mask.sum() < 0.02
    # The lesion was drawn darker than the background.
    assert result["features"]["firstorder"]["Mean"] < 80


def test_reproducible_under_fixed_settings(synthetic_case, settings):
    image, mask, roi = synthetic_case
    a = extract_radiomics_features(image, mask, {"roi_bbox": roi}, settings)
    b = extract_radiomics_features(image, mask, {"roi_bbox": roi}, settings)
    assert a["settings_hash"] == b["settings_hash"]
    assert flat_features(a) == flat_features(b)


def test_settings_change_changes_hash(settings):
    from dataclasses import replace

    assert replace(settings, bin_width=16).settings_hash() != settings.settings_hash()


def test_empty_mask_fails_cleanly(synthetic_case, settings):
    image, mask, _ = synthetic_case
    result = extract_radiomics_features(image, np.zeros_like(mask), {}, settings)
    assert result["status"] == "failed"
    assert [e["code"] for e in result["errors"]] == ["empty_mask"]
    assert result["features"] == {}


def test_shape_mismatch_and_bad_geometry(synthetic_case, settings):
    image, mask, _ = synthetic_case
    assert extract_radiomics_features(image, mask[:-5], {}, settings)["errors"][0]["code"] == "shape_mismatch"
    four_channel = np.zeros(image.shape[:2] + (4,), np.uint8)
    assert (extract_radiomics_features(four_channel, mask, {}, settings)["errors"][0]["code"]
            == "unsupported_image_geometry")


def test_tiny_and_fragmented_masks(synthetic_case, settings):
    image, _, _ = synthetic_case
    tiny = np.zeros(image.shape[:2], np.uint8)
    tiny[10:14, 10:14] = 1
    assert validate_mask(image, tiny, settings).status == "fail"
    frag = np.zeros(image.shape[:2], np.uint8)
    frag[20:60, 20:60] = 1
    frag[100:140, 120:160] = 1
    qc = validate_mask(image, frag, settings)
    assert qc.status == "warn" and {"fragmented_mask", "no_dominant_component"} <= {i["code"] for i in qc.issues}
    result = extract_radiomics_features(image, frag, {}, settings)
    assert result["preprocessing"]["extracted_area_px"] == 1600  # largest component only


def test_misaligned_mask_is_rejected(synthetic_case, settings):
    image, mask, _ = synthetic_case
    result = extract_radiomics_features(image, mask, {"roi_bbox": [0, 0, 60, 60]}, settings)
    assert result["status"] == "failed"
    assert "mask_outside_roi" in [e["code"] for e in result["errors"]]
    oob = extract_radiomics_features(image, mask, {"roi_bbox": [50, 35, 900, 105]}, settings)
    assert "roi_out_of_bounds" in [e["code"] for e in oob["errors"]]


def test_roi_mask_maps_back_to_native_geometry():
    roi_mask = np.zeros((256, 256), np.uint8)
    roi_mask[64:192, 64:192] = 1                      # centre half of the ROI
    native = native_mask_from_roi(roi_mask, (100, 50, 200, 250), (300, 400, 3))
    ys, xs = np.nonzero(native)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (125, 174, 100, 199)
    with pytest.raises(ValueError):
        native_mask_from_roi(roi_mask, (100, 50, 500, 250), (300, 400, 3))


def test_unet_output_binarization_matches_pipeline():
    logits = np.array([[-2.0, 0.0, 0.1, 3.0]])
    assert binarize_unet_output(logits).tolist() == [[0, 0, 1, 1]]          # sigmoid > 0.5
    assert binarize_unet_output(np.array([0.2, 0.7]), activation="none").tolist() == [0, 1]


def test_coerce_handles_complex_and_nonfinite():
    assert _coerce(np.complex128(0.8 + 0j)) == 0.8
    assert _coerce(np.complex128(0.8 + 0.3j)) is None
    assert _coerce(np.float64("nan")) is None
    assert _coerce(np.array([1.0, 2.0])) is None
