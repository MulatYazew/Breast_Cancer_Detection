from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("jsonschema")

from bus_research import birads  # noqa: E402
from bus_research import feature_store as fs  # noqa: E402


def test_record_validates_and_round_trips(make_record, tmp_path):
    pytest.importorskip("radiomics")
    record = make_record(labels={"benign_malignant": "malignant", "label_source": "BUSI folder", "split": "train"})
    path = fs.save_record(tmp_path, record)
    assert fs.load_record(path) == json.loads(path.read_text())
    assert not list(path.parent.glob("*.tmp"))
    # Pseudonymous ids carry no part of the raw dataset id (BUSI ids contain the class name).
    for key in ("record_id", "image_id", "lesion_id"):
        assert "case" not in record[key] and "malignant" not in record[key]


def test_regeneration_is_deterministic(make_record):
    pytest.importorskip("radiomics")
    assert fs.content_hash(make_record()) == fs.content_hash(make_record())


def test_strict_json_rejects_nan_and_serializes_numpy():
    with pytest.raises(ValueError):
        fs.dumps_strict({"x": float("nan")})
    with pytest.raises(ValueError):
        fs.dumps_strict({"x": np.float32("inf")})
    out = json.loads(fs.dumps_strict({"a": np.int64(3), "b": np.float64(0.1 + 0.2), "c": np.arange(2),
                                      "d": np.bool_(True)}))
    assert out == {"a": 3, "b": 0.30000000000000004, "c": [0, 1], "d": True}   # full precision kept
    with pytest.raises(TypeError):
        fs.to_jsonable({"p": __import__("pathlib").Path("/tmp/x")})


def test_schema_rejects_bad_records(make_record):
    pytest.importorskip("radiomics")
    record = make_record()
    bad = json.loads(json.dumps(record))
    bad["radiomics"]["features"]["glcm"]["Contrast"] = "high"
    with pytest.raises(fs.RecordValidationError):
        fs.validate_record(bad)
    fabricated = json.loads(json.dumps(record))
    fabricated["birads_descriptors"]["margin"] = {"value": "spiculated", "status": "not_assessed",
                                                  "source": "not_annotated", "confidence": None}
    with pytest.raises(fs.RecordValidationError):
        fs.validate_record(fabricated)
    unsourced = json.loads(json.dumps(record))
    unsourced["birads_descriptors"]["margin"] = {"value": "spiculated", "status": "observed",
                                                 "source": "not_annotated", "confidence": None}
    with pytest.raises(fs.RecordValidationError):
        fs.validate_record(unsourced)


def test_failed_extraction_still_produces_valid_record(make_record, synthetic_case):
    pytest.importorskip("radiomics")
    image, mask, _ = synthetic_case
    record = make_record(mask=np.zeros_like(mask))
    assert record["radiomics"]["status"] == "failed"
    assert record["segmentation"]["quality_status"] == "fail"


def test_unknown_schema_version_needs_migration(make_record):
    pytest.importorskip("radiomics")
    record = make_record()
    record["schema_version"] = "0.9.0"
    with pytest.raises(fs.RecordValidationError):
        fs.migrate_record(record)


def test_manifest_refuses_absolute_paths(tmp_path):
    with pytest.raises(ValueError):
        fs.write_manifest(tmp_path, [{"record_id": "rec_x", "record_path": "/Users/someone/x.json"}])
    path = fs.write_manifest(tmp_path, [{"record_id": "rec_x", "record_path": "records/rec_x.json"}])
    assert json.loads(path.read_text())["n_records"] == 1


# --------------------------------------------------------------------------- #
# BI-RADS schema / label gating
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [("4a", "4A"), ("BI-RADS 4C", "4C"), (3, "3"), ("5.0", "5"),
                                          ("category 0", "0")])
def test_normalize_category(raw, expected):
    assert birads.normalize_category(raw) == expected


@pytest.mark.parametrize("raw", ["7", "4D", "", None, "benign", "malignant"])
def test_invalid_categories_rejected(raw):
    with pytest.raises(birads.InvalidBiradsLabel):
        birads.normalize_category(raw)


def test_training_gate_blocks_missing_or_inferred_labels():
    assert not birads.birads_training_gate(None, None).allowed
    inferred = ["2"] * 30 + ["3"] * 30 + ["5"] * 30
    gate = birads.birads_training_gate(inferred, "derived_from_benign_malignant")
    assert not gate.allowed and "radiologist" in gate.reasons[0]


def test_training_gate_checks_counts_and_non_risk_categories():
    small = birads.birads_training_gate(["3"] * 25 + ["4A"] * 5, "radiologist_assessment")
    assert not small.allowed
    with_six = birads.birads_training_gate(["2"] * 25 + ["3"] * 25 + ["5"] * 25 + ["6"] * 3,
                                           "radiologist_assessment")
    assert not with_six.allowed and any("0 and/or 6" in r for r in with_six.reasons)
    ok = birads.birads_training_gate(["2"] * 25 + ["3"] * 25 + ["4B"] * 25, "radiologist_assessment")
    assert ok.allowed and ok.class_counts == {"2": 25, "3": 25, "4B": 25}


def test_descriptor_validation_and_defaults():
    defaults = birads.empty_descriptors()
    assert defaults["margin"]["status"] == "not_assessed" and defaults["margin"]["value"] is None
    assert defaults["vascularity"]["status"] == "unavailable"
    ok = birads.validate_descriptor("margin", "Spiculated", "radiologist")
    assert ok == {"value": "spiculated", "status": "observed", "source": "radiologist", "confidence": None,
                  "note": ""}
    with pytest.raises(ValueError):
        birads.validate_descriptor("margin", "fuzzy", "radiologist")
    with pytest.raises(ValueError):
        birads.validate_descriptor("elasticity", "hard", "radiologist")
    with pytest.raises(ValueError):
        birads.validate_descriptor("shape", "oval", "model_predicted")   # needs a confidence
