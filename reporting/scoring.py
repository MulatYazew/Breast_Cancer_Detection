"""Stage 6c: deterministic BI-RADS category proposal ("rules propose, LLM explains").

The category, likelihood-of-malignancy range and confidence come from here,
not from the language model: the LLM only explains this proposal, and the
validator rejects any draft whose category disagrees. Category 6 (biopsy-proven
malignancy) and category 1 (negative) are never proposed — 6 can only be set by
the clinician, and a report is only generated for an outlined lesion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from reporting.features import RadiomicFeatures
from reporting.lexicon import Descriptor, descriptor_map, lesion_truncated

CATEGORY_ORDER = ["0", "1", "2", "3", "4A", "4B", "4C", "5", "6"]
MODEL_CATEGORIES = ["0", "2", "3", "4A", "4B", "4C", "5"]   # what the rules/LLM may propose
CLINICIAN_CATEGORIES = ["0", "1", "2", "3", "4A", "4B", "4C", "5", "6"]

LIKELIHOOD = {
    "0": "N/A (incomplete)",
    "1": "essentially 0%",
    "2": "essentially 0%",
    "3": ">0% to ≤2%",
    "4A": ">2% to ≤10%",
    "4B": ">10% to ≤50%",
    "4C": ">50% to <95%",
    "5": "≥95%",
    "6": "N/A (biopsy-proven)",
}

DEFAULT_RECOMMENDATION = {
    "0": "Incomplete: additional imaging evaluation is needed to fully characterize the lesion.",
    "2": "Benign finding: routine age-appropriate screening.",
    "3": "Probably benign: short-interval follow-up ultrasound in 6 months (or continued surveillance).",
    "4A": "Low suspicion for malignancy: tissue diagnosis (ultrasound-guided core biopsy) is recommended.",
    "4B": "Moderate suspicion for malignancy: tissue diagnosis (ultrasound-guided core biopsy) is recommended.",
    "4C": "High suspicion for malignancy: tissue diagnosis (ultrasound-guided core biopsy) is recommended.",
    "5": "Highly suggestive of malignancy: tissue diagnosis is recommended; a benign result would be discordant.",
}

# Points per suspicious lexicon value. Any single suspicious feature is enough
# to leave category 3 (ACR: a probably-benign mass has *no* suspicious features).
SUSPICIOUS_POINTS = {
    ("shape", "irregular"): 1.0,
    ("orientation", "not parallel"): 1.0,
    ("margin", "not circumscribed - spiculated"): 2.0,
    ("margin", "not circumscribed - angular"): 1.0,
    ("margin", "not circumscribed - microlobulated"): 1.0,
    ("margin", "not circumscribed - indistinct"): 1.0,
    ("posterior", "shadowing"): 1.0,
    ("posterior", "combined pattern"): 0.5,
    ("echo_pattern", "complex cystic and solid"): 1.0,
    ("echo_pattern", "heterogeneous"): 0.25,
}
# Upper score bound for each category (score > last bound -> 5).
CATEGORY_BOUNDS = [("3", 0.25), ("4A", 1.5), ("4B", 3.0), ("4C", 4.5)]


@dataclass
class RuleAssessment:
    category: str
    likelihood: str
    score: float
    points: list[dict] = field(default_factory=list)   # [{descriptor, value, points}]
    malignant_probability: float | None = None
    confidence: str = "high"                           # high | moderate | low
    confidence_reasons: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _model_points(p_malignant: float | None) -> float:
    if p_malignant is None:
        return 0.0
    if p_malignant >= 0.9:
        return 1.5
    if p_malignant >= 0.7:
        return 1.0
    if p_malignant >= 0.5:
        return 0.5
    if p_malignant < 0.1:
        return -0.5
    return 0.0


def assess(
    descriptors: list[Descriptor],
    features: RadiomicFeatures,
    p_malignant: float | None,
    mask_source: str = "model",
) -> RuleAssessment:
    d = descriptor_map(descriptors)
    reasons: list[str] = []

    if lesion_truncated(features) or not (d["shape"].assessable and d["margin"].assessable):
        return RuleAssessment(
            category="0", likelihood=LIKELIHOOD["0"], score=0.0, malignant_probability=p_malignant,
            confidence="low",
            confidence_reasons=["Lesion outline touches the ROI border, so the lesion may be only partially imaged."],
            recommendation=DEFAULT_RECOMMENDATION["0"],
        )

    is_simple_cyst = (
        d["shape"].value in ("oval", "round") and d["margin"].value == "circumscribed"
        and d["echo_pattern"].value == "anechoic" and d["posterior"].value == "enhancement"
        and (p_malignant is None or p_malignant < 0.5)
    )

    points = [
        {"descriptor": key, "value": d[key].value, "points": pts}
        for (key, value), pts in SUSPICIOUS_POINTS.items() if key in d and d[key].value == value
    ]
    model_pts = _model_points(p_malignant)
    if model_pts:
        points.append({"descriptor": "classifier", "value": f"p(malignant)={p_malignant:.2f}", "points": model_pts})
    score = sum(p["points"] for p in points)

    if is_simple_cyst:
        category = "2"
    else:
        category = next((cat for cat, bound in CATEGORY_BOUNDS if score <= bound), "5")
        # A mass with any suspicious lexicon feature is never "probably benign".
        if category == "3" and any(p["descriptor"] != "classifier" for p in points if p["points"] >= 1.0):
            category = "4A"

    # --- confidence -------------------------------------------------------- #
    suspicious_features = sum(1 for p in points if p["descriptor"] != "classifier" and p["points"] >= 1.0)
    if p_malignant is not None:
        if category in ("2", "3") and p_malignant >= 0.5:
            reasons.append(f"The classifier leans malignant (p={p_malignant:.2f}) but the lexicon features look benign.")
        if category in ("4B", "4C", "5") and p_malignant < 0.2:
            reasons.append(f"The lexicon features are suspicious but the classifier leans benign (p={p_malignant:.2f}).")
        if suspicious_features == 0 and p_malignant >= 0.7:
            reasons.append("The category is driven mainly by the classifier, without suspicious lexicon features.")
    if not is_simple_cyst and any(abs(score - bound) < 0.25 for _, bound in CATEGORY_BOUNDS):
        reasons.append(f"The score ({score:.2f}) lies near a category boundary.")
    if not d["posterior"].assessable:
        reasons.append("Posterior features could not be assessed.")
    if mask_source == "manual":
        reasons.append("Margin descriptors come from a hand-painted mask, so the boundary detail is approximate.")
    if features.quality["mask_components"] > 1:
        reasons.append("The approved mask has several disconnected parts; only the largest was analysed.")
    confidence = "high" if not reasons else ("moderate" if len(reasons) == 1 else "low")

    return RuleAssessment(
        category=category, likelihood=LIKELIHOOD[category], score=round(score, 2), points=points,
        malignant_probability=p_malignant, confidence=confidence, confidence_reasons=reasons,
        recommendation=DEFAULT_RECOMMENDATION[category],
    )
