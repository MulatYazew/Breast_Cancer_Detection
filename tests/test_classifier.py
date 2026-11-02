from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("radiomics")
pytest.importorskip("sklearn")

from bus_research.classifier import (  # noqa: E402
    CLASS_ORDER,
    RadiomicsClassifier,
    feature_vector,
    load_if_present,
)
from bus_research.radiomics_extract import flat_features  # noqa: E402


@pytest.fixture
def artifact(tmp_path, make_record):
    """Tiny artifact trained on synthetic records (darker = 'malignant'), saved like notebook 04 does."""
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    records = [make_record(key=f"c{i}", intensity_shift=s) for i, s in enumerate([-40, -30, -20, 20, 30, 40])]
    names = sorted(flat_features(records[0]["radiomics"]))
    X = np.stack([feature_vector(r, names) for r in records])
    y = np.array([1, 1, 1, 0, 0, 0])
    model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression()).fit(X, y)
    joblib.dump(model, tmp_path / "model.joblib")
    metadata = {
        "model_id": "test-lr", "prediction_type": "benign_vs_malignant", "class_order": CLASS_ORDER,
        "feature_names": names, "required_settings_hash": records[0]["radiomics"]["settings_hash"],
        "calibration": {"status": "uncalibrated", "method": None},
        "train_feature_stats": {"mean": X.mean(0).tolist(), "std": X.std(0).tolist()},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path, records


def test_load_and_predict(artifact, make_record):
    path, _ = artifact
    clf = load_if_present(path)
    out = clf.predict(make_record(key="new", intensity_shift=-35))
    assert out["available"] and out["prediction_type"] == "benign_vs_malignant"
    assert list(out["class_probabilities"]) == CLASS_ORDER
    assert abs(sum(out["class_probabilities"].values()) - 1) < 1e-9
    assert out["class_probabilities"]["malignant"] > 0.5
    assert out["calibration_status"] == "uncalibrated"


def test_inference_uses_training_preprocessing(artifact):
    """The loaded artifact reproduces the fitted pipeline's own output exactly."""
    import joblib

    path, records = artifact
    clf = RadiomicsClassifier.load(path)
    raw_model = joblib.load(path / "model.joblib")
    x = feature_vector(records[0], clf.metadata["feature_names"])
    assert clf.predict(records[0])["class_probabilities"]["malignant"] == pytest.approx(
        raw_model.predict_proba(x.reshape(1, -1))[0, 1])


def test_refuses_incompatible_settings_and_failed_records(artifact, make_record, synthetic_case):
    path, records = artifact
    clf = RadiomicsClassifier.load(path)
    other = json.loads(json.dumps(records[0]))
    other["radiomics"]["settings_hash"] = "deadbeefdeadbeef"
    out = clf.predict(other)
    assert not out["available"] and "settings" in out["reason"]
    failed = make_record(mask=np.zeros_like(synthetic_case[1]))
    assert not clf.predict(failed)["available"]


def test_class_order_mismatch_is_rejected(artifact):
    path, _ = artifact
    meta = json.loads((path / "metadata.json").read_text())
    meta["class_order"] = ["malignant", "benign"]
    (path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        RadiomicsClassifier.load(path)


def test_vector_excludes_labels_and_ids(make_record):
    record = make_record(labels={"benign_malignant": "malignant", "label_source": "BUSI", "split": "train"})
    with pytest.raises(ValueError):
        feature_vector(record, ["reference_labels.benign_malignant"])
    with pytest.raises(ValueError):
        feature_vector(record, ["segmentation.mask_area_px"])
    names = sorted(flat_features(record["radiomics"]))
    assert all(n.split(".")[0] in {"shape2D", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"}
               for n in names)


def test_missing_artifact_returns_none(tmp_path):
    assert load_if_present(tmp_path / "nope") is None
