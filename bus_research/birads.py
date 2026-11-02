"""BI-RADS (ACR 5th edition) assessment categories, ultrasound descriptor schema, and label gating.

Vocabulary is transcribed from the ACR BI-RADS Atlas 5th Edition Quick
Reference card shipped with the dataset (``dataset/BUSI_Jpeg/BIRADS Reference
Card (1).pdf``, ultrasound column + assessment-category table).

Nothing here *derives* a BI-RADS category or descriptor from model output.
Descriptors are separately annotated (or separately predicted) fields with
explicit provenance, and a BI-RADS classifier may only be trained when
``birads_training_gate`` confirms genuine radiologist labels exist.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Literal

# Assessment categories (card, page 2). 4A/4B/4C subdivide 4 for mammography & ultrasound.
CATEGORIES: dict[str, str] = {
    "0": "Incomplete – need additional imaging evaluation",
    "1": "Negative",
    "2": "Benign",
    "3": "Probably benign",
    "4": "Suspicious",
    "4A": "Low suspicion for malignancy",
    "4B": "Moderate suspicion for malignancy",
    "4C": "High suspicion for malignancy",
    "5": "Highly suggestive of malignancy",
    "6": "Known biopsy-proven malignancy",
}
# Categories that are not points on an image-based risk scale: 0 means the work-up
# is incomplete and 6 is a pathology fact. A risk classifier must not emit them.
NON_RISK_CATEGORIES = frozenset({"0", "6"})
ORDINAL_RISK_ORDER = ["1", "2", "3", "4A", "4B", "4C", "5"]

# Ultrasound lexicon (card, page 1). Margin subtypes all mean "not circumscribed".
US_DESCRIPTORS: dict[str, list[str]] = {
    "shape": ["oval", "round", "irregular"],
    "orientation": ["parallel", "not parallel"],
    "margin": ["circumscribed", "indistinct", "angular", "microlobulated", "spiculated"],
    "echo_pattern": ["anechoic", "hyperechoic", "complex cystic and solid", "hypoechoic", "isoechoic",
                     "heterogeneous"],
    "posterior_features": ["no posterior features", "enhancement", "shadowing", "combined pattern"],
    "calcifications": ["calcifications in a mass", "calcifications outside of a mass",
                       "intraductal calcifications", "none"],
    "vascularity": ["absent", "internal vascularity", "vessels in rim"],
    "elasticity": ["soft", "intermediate", "hard"],
}
# What a single grayscale B-mode still (BUSI) cannot show at all.
NOT_ASSESSABLE_ON_BMODE_STILL = frozenset({"vascularity", "elasticity"})

DescriptorSource = Literal["not_annotated", "radiologist", "model_predicted"]
DescriptorStatus = Literal["observed", "not_assessed", "unavailable", "invalid"]


@dataclass
class DescriptorValue:
    """One BI-RADS descriptor with provenance. ``value`` is set only when ``status == "observed"``."""

    value: str | None = None
    status: DescriptorStatus = "not_assessed"
    source: DescriptorSource = "not_annotated"
    confidence: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def empty_descriptors() -> dict[str, dict]:
    """The default for every record: nothing assessed, nothing fabricated.

    Doppler vascularity and elastography are marked ``unavailable`` rather
    than ``not_assessed``: no amount of annotation of a B-mode still fixes that.
    """
    out = {}
    for key in US_DESCRIPTORS:
        if key in NOT_ASSESSABLE_ON_BMODE_STILL:
            out[key] = DescriptorValue(status="unavailable", note="Not visible on a grayscale B-mode still image.")
        else:
            out[key] = DescriptorValue()
    return {k: v.to_dict() for k, v in out.items()}


def validate_descriptor(key: str, value: str | None, source: str, confidence: float | None = None) -> dict:
    """Validate an annotated/predicted descriptor value against the lexicon; return the record form."""
    if key not in US_DESCRIPTORS:
        raise ValueError(f"unknown ultrasound descriptor {key!r}")
    if source not in ("radiologist", "model_predicted"):
        raise ValueError(f"descriptor source must be 'radiologist' or 'model_predicted', got {source!r}")
    if key in NOT_ASSESSABLE_ON_BMODE_STILL:
        raise ValueError(f"{key!r} cannot be assessed from a B-mode still image")
    normalized = (value or "").strip().lower()
    if normalized not in US_DESCRIPTORS[key]:
        raise ValueError(f"{value!r} is not a BI-RADS {key} term; expected one of {US_DESCRIPTORS[key]}")
    if source == "model_predicted" and confidence is None:
        raise ValueError("a model-predicted descriptor needs a confidence")
    return DescriptorValue(normalized, "observed", source, confidence).to_dict()


class InvalidBiradsLabel(ValueError):
    pass


def normalize_category(label: object) -> str:
    """'4a' / 'BI-RADS 4A' / 4 -> canonical category string; raises on anything else."""
    if label is None:
        raise InvalidBiradsLabel("missing BI-RADS label")
    text = str(label).strip().upper().replace("BI-RADS", "").replace("BIRADS", "").replace("CATEGORY", "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text not in CATEGORIES:
        raise InvalidBiradsLabel(f"{label!r} is not a BI-RADS assessment category")
    return text


@dataclass
class TrainingGate:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    invalid_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def birads_training_gate(
    labels: list[object] | None,
    label_source: str | None,
    min_per_class: int = 20,
    min_classes: int = 3,
) -> TrainingGate:
    """Decide whether a BI-RADS category classifier may be trained on ``labels``.

    Requires lesion-level labels assigned by a radiologist (``label_source ==
    "radiologist_assessment"``), every label a valid category, no 0/6 in the
    risk-scale target, and at least ``min_classes`` categories with
    ``min_per_class`` lesions each. Labels inferred from pathology, model
    output, or LLM text are rejected by design.
    """
    reasons: list[str] = []
    if not labels:
        return TrainingGate(False, ["No BI-RADS reference labels are present in the dataset."])
    if label_source != "radiologist_assessment":
        reasons.append(
            f"Label source is {label_source!r}; only lesion-level radiologist assessments are accepted "
            "(not pathology, benign/malignant folders, model probabilities, or generated text)."
        )
    valid, invalid = [], []
    for label in labels:
        try:
            valid.append(normalize_category(label))
        except InvalidBiradsLabel:
            invalid.append(str(label))
    if invalid:
        reasons.append(f"{len(invalid)} label(s) are not valid BI-RADS categories.")
    counts = Counter(valid)
    if counts.keys() & NON_RISK_CATEGORIES:
        reasons.append("Categories 0 and/or 6 are present; exclude or model them separately — they are not "
                       "points on an image-based malignancy-risk scale.")
    usable = [c for c, n in counts.items() if c not in NON_RISK_CATEGORIES and n >= min_per_class]
    if len(usable) < min_classes:
        reasons.append(f"Only {len(usable)} category(ies) have >= {min_per_class} lesions; need {min_classes}.")
    return TrainingGate(not reasons, reasons, dict(sorted(counts.items())), invalid)


def unavailable_assessment(reason: str = "validated BI-RADS labels/model not configured") -> dict:
    return {"status": "unavailable", "category": None, "source": None,
            "reason": f"BI-RADS assessment unavailable: {reason}."}
