"""Load and apply the radiomics benign-vs-malignant classifier trained in notebook 04.

Artifact layout (``research.models.classifier_dir``)::

    model.joblib     fitted sklearn Pipeline (imputer -> scaler -> estimator [-> calibrator])
    metadata.json    feature names, class order, settings hash, calibration, provenance, metrics

The predictor vector is built **only** from ``record["radiomics"]["features"]``
by the exact feature names stored at training time; ids, labels, QC fields and
file metadata can never enter it. A record extracted with different settings
(``settings_hash`` mismatch) is refused rather than silently mis-scored.

This is a benign/malignant model. It never outputs a BI-RADS category.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bus_research.radiomics_extract import FEATURE_CLASSES

PREDICTION_TYPE = "benign_vs_malignant"
CLASS_ORDER = ["benign", "malignant"]


class IncompatibleRecord(ValueError):
    pass


def feature_vector(record: dict, feature_names: list[str]) -> np.ndarray:
    """Record -> 1-D float vector in ``feature_names`` order (NaN where a feature is missing)."""
    for name in feature_names:
        family = name.split(".", 1)[0]
        if family not in FEATURE_CLASSES:
            raise ValueError(f"{name!r} is not a radiomics feature; refusing non-radiomics predictors")
    feats = (record.get("radiomics") or {}).get("features") or {}
    out = np.full(len(feature_names), np.nan)
    for i, name in enumerate(feature_names):
        family, feature = name.split(".", 1)
        value = feats.get(family, {}).get(feature)
        if value is not None:
            out[i] = float(value)
    return out


def unavailable_prediction(reason: str) -> dict:
    return {"available": False, "model_id": None, "prediction_type": None, "class_probabilities": None,
            "calibration_status": "not_available", "reason": reason}


@dataclass
class RadiomicsClassifier:
    model: object
    metadata: dict

    @property
    def model_id(self) -> str:
        return self.metadata["model_id"]

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "RadiomicsClassifier":
        import joblib

        artifact_dir = Path(artifact_dir)
        metadata = json.loads((artifact_dir / "metadata.json").read_text())
        if metadata.get("class_order") != CLASS_ORDER or metadata.get("prediction_type") != PREDICTION_TYPE:
            raise ValueError(f"artifact class order/type {metadata.get('class_order')}/"
                             f"{metadata.get('prediction_type')} does not match {CLASS_ORDER}/{PREDICTION_TYPE}")
        model = joblib.load(artifact_dir / "model.joblib")
        if list(getattr(model, "classes_", [0, 1])) != [0, 1]:
            raise ValueError("model classes_ must be [0, 1] (benign, malignant)")
        return cls(model, metadata)

    def check_compatible(self, record: dict) -> None:
        rad = record.get("radiomics") or {}
        if rad.get("status") != "ok":
            raise IncompatibleRecord("radiomics extraction did not succeed for this lesion")
        if rad.get("settings_hash") != self.metadata["required_settings_hash"]:
            raise IncompatibleRecord(
                f"record extracted with settings {rad.get('settings_hash')}, model needs "
                f"{self.metadata['required_settings_hash']}")

    def ood_check(self, x: np.ndarray) -> dict:
        stats = self.metadata["train_feature_stats"]
        mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
        z = np.abs((x - mean) / np.where(std > 0, std, 1.0))
        extreme = int(np.nansum(z > stats.get("z_limit", 6.0)))
        frac = extreme / len(x)
        return {"n_extreme_features": extreme, "fraction_extreme": round(frac, 4),
                "out_of_distribution": frac > stats.get("max_extreme_fraction", 0.1)}

    def predict(self, record: dict) -> dict:
        """Calibrated (if available) class probabilities, or an explicit unavailable/abstain result."""
        try:
            self.check_compatible(record)
        except IncompatibleRecord as exc:
            return unavailable_prediction(str(exc))
        names = self.metadata["feature_names"]
        x = feature_vector(record, names)
        missing = int(np.isnan(x).sum())
        if missing > 0.1 * len(names):
            return unavailable_prediction(f"{missing}/{len(names)} model features are missing for this lesion")
        ood = self.ood_check(x)
        probs = self.model.predict_proba(x.reshape(1, -1))[0]
        cal = self.metadata.get("calibration") or {}
        return {
            "available": True,
            "model_id": self.model_id,
            "prediction_type": PREDICTION_TYPE,
            "class_probabilities": {c: float(p) for c, p in zip(CLASS_ORDER, probs)},
            "calibration_status": cal.get("status", "uncalibrated"),
            "calibration_method": cal.get("method"),
            "n_missing_features": missing,
            "ood": ood,
        }


def load_if_present(artifact_dir: str | Path) -> RadiomicsClassifier | None:
    artifact_dir = Path(artifact_dir)
    if (artifact_dir / "model.joblib").exists() and (artifact_dir / "metadata.json").exists():
        return RadiomicsClassifier.load(artifact_dir)
    return None
