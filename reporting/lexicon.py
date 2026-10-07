"""Stage 6b: map radiomic features onto the ACR BI-RADS ultrasound lexicon.

One ``Descriptor`` per lexicon category, with the chosen value, the numbers
that support it, whether it can be assessed from a single B-mode image at all,
and its polarity (supports benign / suspicious / neutral). Lexicon categories a
still grayscale image can't assess (vascularity, elasticity, skin, edema,
special cases) are listed explicitly as not assessable, so they are never
silently missing from a report.

Thresholds were set by inspecting feature distributions on the *training*
split's ground-truth masks (BUSI); they are heuristics, not validated
cut-offs. ``reporting/calibration.py`` re-checks them on the validation split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from reporting.features import RadiomicFeatures

Polarity = Literal["benign", "suspicious", "neutral"]

THRESHOLDS = {
    "irregular_solidity": 0.92,        # contour / convex-hull area
    "irregular_ellipse_iou": 0.86,     # lesion vs. best-fit ellipse
    "round_axis_ratio": 0.85,
    "not_parallel_depth_to_width": 1.0,
    "spiculated_spikes": 5,
    "spiculated_roughness": 0.09,
    "angular_concavities": 2,
    "microlobulated_lobules": 8,
    "microlobulated_roughness": 0.05,
    "indistinct_cnr": 0.6,
    "anechoic_fraction": 0.8,
    "anechoic_ratio": 0.3,
    "complex_anechoic_low": 0.25,
    "complex_solid_fraction": 0.25,
    "heterogeneous_local": 0.28,
    "isoechoic_low": 0.8,
    "isoechoic_high": 1.2,
    "enhancement_ratio": 1.2,
    "shadowing_ratio": 0.85,
    "echogenic_foci": 3,
    "truncated_edge_touch": 0.15,
}


@dataclass
class Descriptor:
    key: str                     # stable id, e.g. "margin"; rationale sources cite "feature:<key>"
    category: str                # lexicon category as shown on the ACR card
    value: str                   # lexicon term, or "not assessable"
    assessable: bool
    polarity: Polarity
    kb_id: str | None            # knowledge-base chunk defining this term
    evidence: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(float(x), nd)


def map_to_lexicon(features: RadiomicFeatures) -> list[Descriptor]:
    t = THRESHOLDS
    s, o, m, e, p, c = (features.shape, features.orientation, features.margin,
                        features.echo, features.posterior, features.calcification)
    out: list[Descriptor] = []

    # --- shape ------------------------------------------------------------ #
    if s["solidity"] < t["irregular_solidity"] or s["ellipse_fit_iou"] < t["irregular_ellipse_iou"]:
        shape, pol = "irregular", "suspicious"
    elif s["axis_ratio"] >= t["round_axis_ratio"]:
        shape, pol = "round", "neutral"
    else:
        shape, pol = "oval", "benign"
    out.append(Descriptor("shape", "Mass shape", shape, True, pol, f"us-shape-{shape}",
                          {"solidity": _r(s["solidity"]), "ellipse_fit_iou": _r(s["ellipse_fit_iou"]),
                           "axis_ratio": _r(s["axis_ratio"]), "circularity": _r(s["circularity"])}))

    # --- orientation ------------------------------------------------------ #
    not_parallel = o["depth_to_width_ratio"] > t["not_parallel_depth_to_width"]
    out.append(Descriptor(
        "orientation", "Orientation", "not parallel" if not_parallel else "parallel", True,
        "suspicious" if not_parallel else "benign",
        "us-orientation-not-parallel" if not_parallel else "us-orientation-parallel",
        {"depth_to_width_ratio": _r(o["depth_to_width_ratio"]), "major_axis_angle_deg": _r(o["major_axis_angle_deg"], 1)},
        "Assumes the image's top edge is the skin surface (standard probe orientation).",
    ))

    # --- margin (most suspicious subtype wins, per the lexicon) ----------- #
    margin_ev = {"spike_count": m["spike_count"], "lobule_count": m["lobule_count"],
                 "deep_concavities": m["deep_concavities"],
                 "boundary_roughness": _r(m["boundary_roughness"]), "boundary_cnr": _r(m["boundary_cnr"])}
    if m["spike_count"] >= t["spiculated_spikes"] and m["boundary_roughness"] >= t["spiculated_roughness"]:
        margin = "spiculated"
    elif m["deep_concavities"] >= t["angular_concavities"]:
        margin = "angular"
    elif m["lobule_count"] >= t["microlobulated_lobules"] and m["boundary_roughness"] >= t["microlobulated_roughness"]:
        margin = "microlobulated"
    elif m["boundary_cnr"] < t["indistinct_cnr"]:
        margin = "indistinct"
    else:
        margin = "circumscribed"
    out.append(Descriptor(
        "margin", "Margin", margin if margin == "circumscribed" else f"not circumscribed - {margin}", True,
        "benign" if margin == "circumscribed" else "suspicious", f"us-margin-{margin}", margin_ev,
    ))

    # --- echo pattern ----------------------------------------------------- #
    if e["anechoic_fraction"] >= t["anechoic_fraction"] and e["echogenicity_ratio"] < t["anechoic_ratio"]:
        echo, pol, kb = "anechoic", "benign", "us-echo-anechoic"
    elif e["anechoic_fraction"] >= t["complex_anechoic_low"] and e["solid_fraction"] >= t["complex_solid_fraction"]:
        echo, pol, kb = "complex cystic and solid", "suspicious", "us-echo-complex"
    elif e["local_heterogeneity"] >= t["heterogeneous_local"]:
        echo, pol, kb = "heterogeneous", "neutral", "us-echo-heterogeneous"
    elif e["echogenicity_ratio"] < t["isoechoic_low"]:
        echo, pol, kb = "hypoechoic", "neutral", "us-echo-hypoechoic"
    elif e["echogenicity_ratio"] <= t["isoechoic_high"]:
        echo, pol, kb = "isoechoic", "neutral", "us-echo-isoechoic"
    else:
        echo, pol, kb = "hyperechoic", "benign", "us-echo-hyperechoic"
    out.append(Descriptor(
        "echo_pattern", "Echo pattern", echo, True, pol, kb,
        {"echogenicity_ratio": _r(e["echogenicity_ratio"]), "anechoic_fraction": _r(e["anechoic_fraction"]),
         "solid_fraction": _r(e["solid_fraction"]), "local_heterogeneity": _r(e["local_heterogeneity"])},
        "Echogenicity is relative to the surrounding tissue ring, not to subcutaneous fat.",
    ))

    # --- posterior features ---------------------------------------------- #
    ratio = p.get("posterior_ratio")
    post_ev = {"posterior_ratio": _r(ratio), "left_ratio": _r(p.get("posterior_left_ratio")),
               "right_ratio": _r(p.get("posterior_right_ratio")),
               "available_fraction": _r(p.get("posterior_available_fraction"))}
    if ratio is None:
        out.append(Descriptor("posterior", "Posterior features", "not assessable", False, "neutral", None, post_ev,
                              "Too little tissue imaged deep to or beside the lesion."))
    else:
        left, right = p.get("posterior_left_ratio"), p.get("posterior_right_ratio")
        halves = [h for h in (left, right) if h is not None]
        if (len(halves) == 2 and max(halves) >= t["enhancement_ratio"] and min(halves) <= t["shadowing_ratio"]):
            post, pol = "combined pattern", "neutral"
        elif ratio <= t["shadowing_ratio"]:
            post, pol = "shadowing", "suspicious"
        elif ratio >= t["enhancement_ratio"]:
            post, pol = "enhancement", "neutral"
        else:
            post, pol = "no posterior features", "neutral"
        kb = {"combined pattern": "us-posterior-combined", "shadowing": "us-posterior-shadowing",
              "enhancement": "us-posterior-enhancement", "no posterior features": "us-posterior-none"}[post]
        out.append(Descriptor("posterior", "Posterior features", post, True, pol, kb, post_ev))

    # --- calcifications: limited on B-mode, informational only ----------- #
    foci = c["echogenic_foci_count"]
    out.append(Descriptor(
        "calcifications", "Calcifications",
        "possible echogenic foci in mass" if foci >= t["echogenic_foci"] else "none identified",
        True, "neutral", "us-calcifications", {"echogenic_foci_count": foci},
        "Limited reliability: speckle can mimic foci. Correlate with mammography.",
    ))

    # --- lexicon categories a single B-mode still can't assess ----------- #
    for key, category in (
        ("vascularity", "Vascularity (Doppler)"),
        ("elasticity", "Elasticity assessment"),
        ("skin_changes", "Skin changes"),
        ("edema", "Edema"),
        ("special_cases", "Special cases"),
    ):
        out.append(Descriptor(key, category, "not assessable", False, "neutral", "us-not-assessable-bmode",
                              note="Not assessable from a single grayscale B-mode image."))
    return out


def lesion_truncated(features: RadiomicFeatures) -> bool:
    """True when the mask runs along the ROI border, i.e. the lesion may be cut off."""
    return features.quality["edge_touch_fraction"] > THRESHOLDS["truncated_edge_touch"]


def descriptor_map(descriptors: list[Descriptor]) -> dict[str, Descriptor]:
    return {d.key: d for d in descriptors}
