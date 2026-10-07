"""Stage 6a: radiomic features from the clinician-approved lesion mask.

Everything is computed from the *approved* mask (model mask the clinician
accepted, or the clinician's manual mask), never from a rejected prediction.

The ROI crop/mask the app works with is resized to ``roi_size`` (256x256),
which destroys the lesion's aspect ratio — so the mask is first mapped back
into native image geometry via the padded bbox it was cropped from. Shape,
orientation ("taller than wide") and posterior features are only meaningful
there.

These are hand-crafted, interpretable measurements chosen to back the BI-RADS
ultrasound lexicon (``reporting.lexicon``). Their thresholds are heuristics,
not values validated against radiologist reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

Bbox = tuple[int, int, int, int]


class EmptyMaskError(ValueError):
    """The approved mask contains no usable lesion pixels."""


@dataclass
class RadiomicFeatures:
    shape: dict[str, float] = field(default_factory=dict)
    orientation: dict[str, float] = field(default_factory=dict)
    margin: dict[str, float] = field(default_factory=dict)
    echo: dict[str, float] = field(default_factory=dict)
    posterior: dict[str, float | None] = field(default_factory=dict)
    calcification: dict[str, float] = field(default_factory=dict)
    texture: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def roi_mask_to_native(roi_mask: np.ndarray, padded_bbox: Bbox, image_shape: tuple[int, ...]) -> np.ndarray:
    """Undo the ROI resize: paste the (roi_size) mask back into full-image coordinates."""
    x1, y1, x2, y2 = padded_bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    native = cv2.resize((roi_mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    full = np.zeros(image_shape[:2], dtype=np.uint8)
    full[y1:y2, x1:x2] = native[: full[y1:y2, x1:x2].shape[0], : full[y1:y2, x1:x2].shape[1]]
    return full


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise EmptyMaskError("approved mask is empty")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8), n - 1


def _smooth_mask(mask: np.ndarray, eq_diameter: float) -> np.ndarray:
    """Remove pixel stair-steps (from the 256->native resize, or a painted brush edge)
    without erasing real lobulations: blur radius ~2% of the lesion diameter."""
    k = max(3, int(round(eq_diameter * 0.02)) * 2 + 1)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    smoothed = (blurred >= 0.5).astype(np.uint8)
    return smoothed if smoothed.sum() > 0 else mask


def _ring(mask: np.ndarray, inner_px: int, outer_px: int) -> np.ndarray:
    """Pixels between ``inner_px`` and ``outer_px`` outside the mask boundary."""
    def dilate(px: int) -> np.ndarray:
        if px <= 0:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        return cv2.dilate(mask, kernel)

    return (dilate(outer_px) > 0) & ~(dilate(inner_px) > 0)


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    eroded = cv2.erode(mask, kernel) > 0
    return eroded if eroded.any() else mask > 0


def _radial_profile(contour: np.ndarray, centroid: tuple[float, float], bins: int = 360) -> np.ndarray:
    """Boundary distance from the centroid as a function of angle (resampled, circular)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    dx, dy = pts[:, 0] - centroid[0], pts[:, 1] - centroid[1]
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    r = np.hypot(dx, dy)
    order = np.argsort(theta)
    theta, r = theta[order], r[order]
    grid = np.linspace(0, 2 * np.pi, bins, endpoint=False)
    return np.interp(grid, theta, r, period=2 * np.pi)


def _circular_smooth(signal: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.concatenate([signal[-window:], signal, signal[:window]])
    return np.convolve(padded, kernel, mode="same")[window:-window]


def _count_peaks(residual: np.ndarray, threshold: float, min_separation: int) -> int:
    """Local maxima above ``threshold`` at least ``min_separation`` bins apart (circular)."""
    n = len(residual)
    candidates = [
        i for i in range(n)
        if residual[i] > threshold and residual[i] >= residual[i - 1] and residual[i] >= residual[(i + 1) % n]
    ]
    peaks: list[int] = []
    for i in sorted(candidates, key=lambda j: -residual[j]):
        if all(min(abs(i - p), n - abs(i - p)) >= min_separation for p in peaks):
            peaks.append(i)
    return len(peaks)


def _glcm_features(gray: np.ndarray, region: np.ndarray, levels: int = 32) -> dict[str, float]:
    """Gray-level co-occurrence (distance 1, 0° and 90°, symmetric) inside ``region``."""
    q = np.clip((gray.astype(np.float32) / 256.0 * levels).astype(np.int32), 0, levels - 1)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    for dy, dx in ((0, 1), (1, 0)):
        a_valid = region[: region.shape[0] - dy, : region.shape[1] - dx]
        b_valid = region[dy:, dx:]
        both = a_valid & b_valid
        a = q[: q.shape[0] - dy, : q.shape[1] - dx][both]
        b = q[dy:, dx:][both]
        np.add.at(glcm, (a, b), 1)
        np.add.at(glcm, (b, a), 1)
    total = glcm.sum()
    if total == 0:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_energy": 0.0, "glcm_correlation": 0.0}
    p = glcm / total
    i, j = np.indices(p.shape)
    mu_i, mu_j = (i * p).sum(), (j * p).sum()
    sd_i = np.sqrt(((i - mu_i) ** 2 * p).sum())
    sd_j = np.sqrt(((j - mu_j) ** 2 * p).sum())
    corr = ((i - mu_i) * (j - mu_j) * p).sum() / (sd_i * sd_j) if sd_i > 0 and sd_j > 0 else 0.0
    return {
        "glcm_contrast": float(((i - j) ** 2 * p).sum()),
        "glcm_homogeneity": float((p / (1.0 + np.abs(i - j))).sum()),
        "glcm_energy": float(np.sqrt((p**2).sum())),
        "glcm_correlation": float(corr),
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def extract_features(image: np.ndarray, padded_bbox: Bbox, roi_mask: np.ndarray) -> RadiomicFeatures:
    """Compute lexicon-oriented radiomic features for one lesion.

    Args:
        image: Full HxWx3 (RGB) or HxW ultrasound image, uint8.
        padded_bbox: The (x1, y1, x2, y2) box the ROI crop was taken from.
        roi_mask: The clinician-approved binary mask in ROI-crop space.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
    gray = gray.astype(np.float32)
    h_img, w_img = gray.shape

    native = roi_mask_to_native(roi_mask, padded_bbox, gray.shape)
    if native.sum() < 30:
        raise EmptyMaskError("approved mask has fewer than 30 lesion pixels")
    lesion, n_components = _largest_component(native)
    eq_diameter = float(np.sqrt(4 * lesion.sum() / np.pi))
    lesion = _smooth_mask(lesion, eq_diameter)
    lesion, _ = _largest_component(lesion)
    lesion_bool = lesion > 0

    feats = RadiomicFeatures()

    # --- quality: is the lesion fully inside the crop / image? ------------ #
    x1, y1, x2, y2 = padded_bbox
    crop = lesion[y1:y2, x1:x2]
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    contours, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, closed=True))
    feats.quality = {
        "mask_components": float(n_components),
        "edge_touch_fraction": float(border.sum() / max(perimeter, 1.0)),
        "lesion_area_px": float(lesion.sum()),
        "equivalent_diameter_px": eq_diameter,
    }

    # --- shape ------------------------------------------------------------ #
    area = float(lesion.sum())
    hull = cv2.convexHull(contour)
    # Contour-polygon area for both terms: pixel-count area vs. hull-polygon area
    # would push solidity above 1 for smooth lesions.
    contour_area = float(cv2.contourArea(contour)) or area
    hull_area = float(cv2.contourArea(hull)) or contour_area
    ellipse_iou = 1.0
    axis_ratio = 1.0
    if len(contour) >= 5:
        (cx_e, cy_e), (d1, d2), angle = cv2.fitEllipse(contour)
        ellipse_mask = np.zeros_like(lesion)
        cv2.ellipse(ellipse_mask, ((cx_e, cy_e), (d1, d2), angle), 1, thickness=-1)
        inter = float((ellipse_mask & lesion).sum())
        union = float((ellipse_mask | lesion).sum())
        ellipse_iou = inter / union if union else 0.0
        axis_ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 1.0
    feats.shape = {
        "area_px": area,
        "perimeter_px": perimeter,
        "circularity": float(4 * np.pi * area / perimeter**2) if perimeter else 0.0,
        "solidity": contour_area / hull_area,
        "ellipse_fit_iou": float(ellipse_iou),
        "axis_ratio": float(axis_ratio),
    }

    # --- orientation ------------------------------------------------------ #
    ys, xs = np.nonzero(lesion_bool)
    width = float(xs.max() - xs.min() + 1)
    depth = float(ys.max() - ys.min() + 1)
    coords = np.stack([xs - xs.mean(), ys - ys.mean()])
    eigvals, eigvecs = np.linalg.eigh(np.cov(coords))
    major = eigvecs[:, int(np.argmax(eigvals))]
    major_angle = float(np.degrees(np.arctan2(abs(major[1]), abs(major[0]))))  # 0 = parallel to skin
    feats.orientation = {
        "depth_to_width_ratio": depth / width,
        "major_axis_angle_deg": major_angle,
        "width_px": width,
        "depth_px": depth,
    }

    # --- margin ----------------------------------------------------------- #
    m = cv2.moments(lesion, binaryImage=True)
    centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    radial = _radial_profile(contour, centroid)
    mean_r = float(radial.mean()) or 1.0
    trend = _circular_smooth(radial, window=45)          # ~45° low-pass = overall shape
    residual = (radial - trend) / mean_r                  # high-frequency boundary detail
    spike_count = _count_peaks(residual, threshold=0.08, min_separation=10)
    lobule_count = _count_peaks(residual, threshold=0.03, min_separation=8)

    deep_defects = 0
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3:
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
        except cv2.error:
            defects = None
        if defects is not None:
            depths = defects[:, 0, 3] / 256.0
            deep_defects = int((depths > 0.08 * eq_diameter).sum())

    ring_px = max(2, int(round(eq_diameter * 0.06)))
    inner = lesion_bool & ~_erode(lesion, ring_px)
    outer = _ring(lesion, 1, ring_px + 1)
    inner_vals, outer_vals = gray[inner], gray[outer]
    pooled_sd = float(np.sqrt((inner_vals.var() + outer_vals.var()) / 2)) if inner_vals.size and outer_vals.size else 0.0
    boundary_cnr = (
        abs(float(outer_vals.mean()) - float(inner_vals.mean())) / pooled_sd
        if pooled_sd > 0 and inner_vals.size and outer_vals.size
        else 0.0
    )
    feats.margin = {
        "radial_cv": float(radial.std() / mean_r),
        "boundary_roughness": float(np.sqrt((residual**2).mean())),
        "spike_count": float(spike_count),
        "lobule_count": float(lobule_count),
        "deep_concavities": float(deep_defects),
        "boundary_cnr": float(boundary_cnr),
    }

    # --- echo pattern ----------------------------------------------------- #
    interior = _erode(lesion, max(1, int(round(eq_diameter * 0.04))))
    reference = _ring(lesion, int(round(eq_diameter * 0.05)) + 1, int(round(eq_diameter * 0.30)) + 2)
    lesion_vals = gray[interior]
    ref_vals = gray[reference]
    ref_mean = float(ref_vals.mean()) if ref_vals.size else float(gray.mean())
    ref_mean = max(ref_mean, 1.0)
    local_mean = cv2.blur(gray, (5, 5))
    anechoic_level = max(20.0, 0.25 * ref_mean)
    feats.echo = {
        "lesion_mean": float(lesion_vals.mean()),
        "reference_mean": ref_mean,
        "echogenicity_ratio": float(lesion_vals.mean() / ref_mean),
        "internal_cv": float(lesion_vals.std() / max(lesion_vals.mean(), 1.0)),
        "local_heterogeneity": float(local_mean[interior].std() / ref_mean),
        "anechoic_fraction": float((local_mean[interior] < anechoic_level).mean()),
        "solid_fraction": float((local_mean[interior] > 0.5 * ref_mean).mean()),
    }

    # --- posterior features ---------------------------------------------- #
    lx1, lx2 = int(xs.min()), int(xs.max())
    ly2 = int(ys.max())
    lesion_h, lesion_w = depth, width
    gap = max(2, int(0.05 * lesion_h))
    py1 = ly2 + gap
    py2 = min(h_img, py1 + max(5, int(0.6 * lesion_h)))
    available = (py2 - py1) / max(1.0, 0.6 * lesion_h)
    excluded = cv2.dilate(native, np.ones((5, 5), np.uint8)) > 0
    posterior: dict[str, float | None] = {"posterior_available_fraction": float(min(available, 1.0))}

    def region_mean(c1: int, c2: int) -> float | None:
        c1, c2 = max(0, c1), min(w_img, c2)
        if c2 - c1 < 3 or py2 - py1 < 3:
            return None
        vals = gray[py1:py2, c1:c2][~excluded[py1:py2, c1:c2]]
        return float(vals.mean()) if vals.size >= 10 else None

    inset = int(0.2 * lesion_w)
    mid = (lx1 + lx2) // 2
    behind = region_mean(lx1 + inset, lx2 - inset)
    behind_left = region_mean(lx1 + inset, mid)
    behind_right = region_mean(mid, lx2 - inset)
    side_w = max(5, int(0.5 * lesion_w))
    side_gap = max(2, int(0.1 * lesion_w))
    lateral = [v for v in (region_mean(lx1 - side_gap - side_w, lx1 - side_gap),
                           region_mean(lx2 + side_gap, lx2 + side_gap + side_w)) if v is not None]
    if behind is not None and lateral and available >= 0.5:
        lateral_mean = max(float(np.mean(lateral)), 1.0)
        posterior.update({
            "posterior_ratio": behind / lateral_mean,
            "posterior_left_ratio": (behind_left / lateral_mean) if behind_left is not None else None,
            "posterior_right_ratio": (behind_right / lateral_mean) if behind_right is not None else None,
            "lateral_reference_count": float(len(lateral)),
        })
    else:
        posterior.update({"posterior_ratio": None, "posterior_left_ratio": None,
                          "posterior_right_ratio": None, "lateral_reference_count": float(len(lateral))})
    feats.posterior = posterior

    # --- calcification proxy: small echogenic foci inside the mass -------- #
    bright_level = max(float(np.percentile(ref_vals, 95)) if ref_vals.size else 200.0,
                       float(lesion_vals.mean() + 3 * lesion_vals.std()))
    bright = ((gray > bright_level) & interior).astype(np.uint8)
    n_b, _, stats_b, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    foci = int(((stats_b[1:, cv2.CC_STAT_AREA] >= 2) & (stats_b[1:, cv2.CC_STAT_AREA] <= 40)).sum()) if n_b > 1 else 0
    feats.calcification = {"echogenic_foci_count": float(foci), "bright_threshold": bright_level}

    # --- first-order + GLCM texture (raw radiomics, display only) --------- #
    vals = gray[lesion_bool]
    hist, _ = np.histogram(vals, bins=32, range=(0, 256))
    prob = hist / max(hist.sum(), 1)
    nz = prob[prob > 0]
    sd = float(vals.std()) or 1.0
    feats.texture = {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "skewness": float(((vals - vals.mean()) ** 3).mean() / sd**3),
        "kurtosis": float(((vals - vals.mean()) ** 4).mean() / sd**4 - 3.0),
        "entropy_bits": float(-(nz * np.log2(nz)).sum()),
        **_glcm_features(gray, lesion_bool),
    }
    return feats
