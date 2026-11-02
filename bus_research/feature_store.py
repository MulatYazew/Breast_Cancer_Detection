"""Versioned, schema-validated JSON records: one per lesion instance.

Value states are explicit, never implied by an empty object:

* radiomics feature present in ``radiomics.features``  -> observed (finite float)
* listed in ``radiomics.feature_status``               -> ``invalid_nonfinite`` / ``unsupported``
* ``radiomics.status == "failed"`` + ``errors``         -> extraction did not run
* BI-RADS descriptor ``status``                         -> observed / not_assessed / unavailable / invalid

Records hold no names, dates, file paths or raw dataset ids: BUSI ids such as
``"benign (12)"`` contain the class label, so ``pseudonymize`` hashes them.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0.0"
RECORD_TYPE = "breast_ultrasound_lesion"
DEFAULT_SALT = "busi-research-v1"

_NUM_OR_NULL = {"type": ["number", "null"]}
_DESCRIPTOR = {
    "type": "object",
    "required": ["value", "status", "source", "confidence"],
    "properties": {
        "value": {"type": ["string", "null"]},
        "status": {"enum": ["observed", "not_assessed", "unavailable", "invalid"]},
        "source": {"enum": ["not_annotated", "radiologist", "model_predicted"]},
        "confidence": _NUM_OR_NULL,
        "note": {"type": "string"},
    },
    "additionalProperties": False,
    # An observed descriptor must carry a value from someone; anything else must not.
    "if": {"properties": {"status": {"const": "observed"}}},
    "then": {"properties": {"value": {"type": "string"},
                            "source": {"enum": ["radiologist", "model_predicted"]}}},
    "else": {"properties": {"value": {"type": "null"}}},
}

LESION_RECORD_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Breast ultrasound lesion feature record",
    "type": "object",
    "required": ["schema_version", "record_type", "record_id", "image_id", "lesion_id", "image_metadata",
                 "segmentation", "radiomics", "birads_descriptors", "reference_labels", "prediction",
                 "limitations", "created_at_utc"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "record_type": {"const": RECORD_TYPE},
        "record_id": {"type": "string", "pattern": "^rec_[0-9a-f]{12}$"},
        "image_id": {"type": "string", "pattern": "^img_[0-9a-f]{12}$"},
        "lesion_id": {"type": "string", "pattern": "^les_[0-9a-f]{12}$"},
        "image_metadata": {
            "type": "object",
            "required": ["width_px", "height_px", "physical_pixel_spacing", "spacing_unit", "image_modality"],
            "properties": {
                "width_px": {"type": "integer", "minimum": 1},
                "height_px": {"type": "integer", "minimum": 1},
                "channels": {"type": "integer"},
                "dtype": {"type": "string"},
                "physical_pixel_spacing": {"type": ["array", "null"]},
                "spacing_unit": {"enum": ["pixel", "mm"]},
                "image_modality": {"const": "ultrasound"},
            },
            "additionalProperties": False,
        },
        "segmentation": {
            "type": "object",
            "required": ["model", "checkpoint_version", "mask_source", "threshold", "mask_area_px", "quality_status"],
            "properties": {
                "model": {"enum": ["unet", "ground_truth_annotation", "clinician"]},
                "checkpoint_version": {"type": ["string", "null"]},
                "mask_source": {"enum": ["predicted", "clinician_corrected", "ground_truth"]},
                "activation": {"type": ["string", "null"]},
                "threshold": _NUM_OR_NULL,
                "bbox_source": {"enum": ["yolo", "manual", "ground_truth", None]},
                "detection_confidence": _NUM_OR_NULL,
                "roi_bbox_xyxy": {"type": ["array", "null"], "items": {"type": "integer"}},
                "mask_area_px": {"type": ["integer", "null"]},
                "quality_status": {"enum": ["pass", "warn", "fail", "not_evaluated"]},
                "quality_issues": {"type": "array"},
            },
            "additionalProperties": False,
        },
        "radiomics": {
            "type": "object",
            "required": ["status", "extractor", "extractor_version", "extraction_version", "settings_hash",
                         "unit_system", "features", "feature_status", "errors"],
            "properties": {
                "status": {"enum": ["ok", "failed"]},
                "extractor": {"const": "PyRadiomics"},
                "extractor_version": {"type": ["string", "null"]},
                "extraction_version": {"type": "string"},
                "settings_hash": {"type": "string"},
                "settings": {"type": "object"},
                "unit_system": {"enum": ["pixel", "physical"]},
                "features": {
                    "type": "object",
                    "additionalProperties": {"type": "object", "additionalProperties": {"type": "number"}},
                },
                "feature_status": {"type": "object",
                                   "additionalProperties": {"enum": ["invalid_nonfinite", "unsupported"]}},
                "preprocessing": {"type": "object"},
                "diagnostics": {"type": "object"},
                "errors": {"type": "array", "items": {"type": "object", "required": ["code", "message"]}},
            },
            "additionalProperties": False,
        },
        "birads_descriptors": {"type": "object", "additionalProperties": _DESCRIPTOR},
        "reference_labels": {
            "type": "object",
            "required": ["benign_malignant", "birads_category", "label_source"],
            "properties": {
                "benign_malignant": {"enum": ["benign", "malignant", None]},
                "birads_category": {"type": ["string", "null"]},
                "label_source": {"type": ["string", "null"]},
                "split": {"enum": ["train", "val", "test", None]},
            },
            "additionalProperties": False,
        },
        "prediction": {
            "type": "object",
            "required": ["model_id", "prediction_type", "class_probabilities", "calibration_status"],
            "properties": {
                "model_id": {"type": ["string", "null"]},
                "prediction_type": {"enum": ["benign_vs_malignant", "birads_category", None]},
                "class_probabilities": {"type": ["object", "null"], "additionalProperties": {"type": "number"}},
                "calibration_status": {"enum": ["calibrated", "uncalibrated", "not_available"]},
                "calibration_method": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
        "created_at_utc": {"type": ["string", "null"]},
        "provenance": {"type": "object"},
    },
}


class RecordValidationError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Identifiers / serialization
# --------------------------------------------------------------------------- #
def pseudonymize(raw_id: str, kind: str = "img", salt: str | None = None) -> str:
    salt = salt if salt is not None else os.environ.get("BUS_PSEUDONYM_SALT", DEFAULT_SALT)
    return f"{kind}_{hashlib.sha256(f'{salt}|{kind}|{raw_id}'.encode()).hexdigest()[:12]}"


def image_content_id(image: np.ndarray, salt: str | None = None) -> str:
    """Pseudonymous id from pixel content (used for app uploads, whose filenames may hold PHI)."""
    arr = np.ascontiguousarray(image)
    return pseudonymize(hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest(), "img", salt)


def to_jsonable(obj):
    """Recursively convert NumPy scalars/arrays and tuples into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        raise TypeError("file paths must not be stored in feature records")
    return obj


def dumps_strict(obj, indent: int | None = 2) -> str:
    """JSON with full float precision (repr) that refuses NaN/Infinity."""
    return json.dumps(to_jsonable(obj), indent=indent, allow_nan=False, sort_keys=False)


def atomic_write_json(path: str | Path, obj) -> Path:
    """Write via temp file + fsync + rename, so a crash never leaves a half-written record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dumps_strict(obj)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)   # mkstemp creates 0600; records are ordinary project artifacts
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def validate_record(record: dict) -> dict:
    """Schema + strict-JSON validation; returns the record unchanged or raises ``RecordValidationError``."""
    import jsonschema

    try:
        json.dumps(to_jsonable(record), allow_nan=False)
    except ValueError as exc:
        raise RecordValidationError(f"record is not strict JSON: {exc}") from exc
    validator = jsonschema.Draft202012Validator(LESION_RECORD_SCHEMA)
    errors = sorted(validator.iter_errors(to_jsonable(record)), key=lambda e: list(e.path))
    if errors:
        msg = "; ".join(f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:5])
        raise RecordValidationError(msg)
    return record


def content_hash(record: dict) -> str:
    """Hash of everything except the timestamp: equal for a deterministic regeneration."""
    body = {k: v for k, v in record.items() if k != "created_at_utc"}
    return hashlib.sha256(dumps_strict(body, indent=None).encode()).hexdigest()[:16]


def build_record(
    *,
    image_key: str,
    lesion_key: str,
    image: np.ndarray,
    radiomics_result: dict,
    segmentation: dict,
    descriptors: dict,
    reference_labels: dict | None = None,
    prediction: dict | None = None,
    limitations: list[str] | None = None,
    provenance: dict | None = None,
    salt: str | None = None,
    timestamp: bool = True,
) -> dict:
    """Assemble and validate one lesion record.

    ``image_key``/``lesion_key`` are raw keys (dataset id or content hash);
    only their salted hashes are stored.
    """
    image_id = pseudonymize(image_key, "img", salt)
    lesion_id = pseudonymize(f"{image_key}#{lesion_key}", "les", salt)
    # One lesion can have several records (ground-truth / predicted mask, GT / YOLO box).
    record_id = pseudonymize(
        f"{image_key}#{lesion_key}#{segmentation.get('mask_source')}#{segmentation.get('bbox_source')}", "rec", salt)
    qc = radiomics_result.get("qc") or {}
    radiomics_block = {k: radiomics_result[k] for k in (
        "status", "extractor", "extractor_version", "extraction_version", "settings_hash", "settings",
        "unit_system", "features", "feature_status", "preprocessing", "errors") if k in radiomics_result}
    if "diagnostics" in radiomics_result:
        radiomics_block["diagnostics"] = radiomics_result["diagnostics"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "record_id": record_id,
        "image_id": image_id,
        "lesion_id": lesion_id,
        "image_metadata": {
            "width_px": int(image.shape[1]),
            "height_px": int(image.shape[0]),
            "channels": int(image.shape[2]) if image.ndim == 3 else 1,
            "dtype": str(image.dtype),
            "physical_pixel_spacing": None,
            "spacing_unit": "pixel",
            "image_modality": "ultrasound",
        },
        "segmentation": {
            "mask_area_px": qc.get("area_px"),
            "quality_status": qc.get("status", "not_evaluated"),
            "quality_issues": qc.get("issues", []),
            **segmentation,
        },
        "radiomics": radiomics_block,
        "birads_descriptors": descriptors,
        "reference_labels": {"benign_malignant": None, "birads_category": None, "label_source": None,
                             "split": None, **(reference_labels or {})},
        "prediction": {"model_id": None, "prediction_type": None, "class_probabilities": None,
                       "calibration_status": "not_available", **(prediction or {})},
        "limitations": list(limitations or []),
        "created_at_utc": datetime.now(timezone.utc).isoformat() if timestamp else None,
        "provenance": provenance or {},
    }
    return validate_record(to_jsonable(record))


def record_path(output_dir: str | Path, record: dict) -> Path:
    return Path(output_dir) / "records" / f"{record['record_id']}.json"


def save_record(output_dir: str | Path, record: dict) -> Path:
    validate_record(record)
    return atomic_write_json(record_path(output_dir, record), record)


def load_record(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text())
    return validate_record(migrate_record(record))


# Schema migrations: {from_version: fn(record) -> record at the next version}.
MIGRATIONS: dict[str, callable] = {}


def migrate_record(record: dict) -> dict:
    seen = set()
    while record.get("schema_version") != SCHEMA_VERSION:
        version = record.get("schema_version")
        if version not in MIGRATIONS or version in seen:
            raise RecordValidationError(f"no migration from schema version {version!r} to {SCHEMA_VERSION}")
        seen.add(version)
        record = MIGRATIONS[version](record)
    return record


def write_manifest(output_dir: str | Path, entries: list[dict], extra: dict | None = None) -> Path:
    """Index linking record ids to splits. Paths are relative to ``output_dir`` (no private paths)."""
    for e in entries:
        if Path(e["record_path"]).is_absolute():
            raise ValueError("manifest record paths must be relative")
    manifest = {"schema_version": SCHEMA_VERSION, "n_records": len(entries), **(extra or {}), "records": entries}
    return atomic_write_json(Path(output_dir) / "manifest.json", manifest)
