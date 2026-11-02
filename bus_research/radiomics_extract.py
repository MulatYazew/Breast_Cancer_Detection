"""Mask quality control + PyRadiomics feature extraction for one breast-ultrasound lesion.

Extraction always runs on the **original-resolution grayscale image** and a
mask aligned to it — never on the 256x256 resized, ImageNet-normalized ROI
tensor the U-Net/ResNet50 consume, and never on the black-background masked
ROI. U-Net masks live in ROI-crop space, so ``native_mask_from_roi`` pastes
them back into image coordinates through the padded box they were cropped
from (reusing ``reporting.features.roi_mask_to_native``).

BUSI has no physical pixel calibration, so the image is given spacing (1, 1)
and every shape feature is reported in **pixels** (``unit_system="pixel"``),
never millimetres.

Only the ``Original`` image type is enabled: wavelet/LoG features multiply the
feature count ~8x on a ~450-lesion training set, and LoG sigmas in mm are
meaningless without calibration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import cv2
import numpy as np

EXTRACTION_VERSION = "bus-radiomics-1.0.0"
FEATURE_CLASSES = ("shape2D", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
# Mirrors ``CADPipeline.segment``: sigmoid(logits) > 0.5.
UNET_ACTIVATION = "sigmoid"
UNET_THRESHOLD = 0.5

QCStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class RadiomicsSettings:
    bin_width: float = 8.0                  # 8-bit intensities -> at most 32 gray levels
    feature_classes: tuple[str, ...] = FEATURE_CLASSES
    image_types: tuple[str, ...] = ("Original",)
    normalize: bool = False                 # keep native echo intensities; no z-scoring
    resampled_pixel_spacing: tuple[float, float] | None = None  # no resampling
    interpolator: str = "sitkBSpline"       # unused while resampling is off; recorded for the hash
    label: int = 1
    keep_largest_component: bool = True
    min_area_px: int = 50
    max_area_fraction: float = 0.6
    min_largest_component_fraction: float = 0.9

    @classmethod
    def from_config(cls, config: dict) -> "RadiomicsSettings":
        cfg = dict((config.get("research") or {}).get("radiomics") or {})
        qc = cfg.pop("qc", {}) or {}
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in {**cfg, **qc}.items() if k in known}
        for key in ("feature_classes", "image_types"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        if kwargs.get("resampled_pixel_spacing") is not None:
            kwargs["resampled_pixel_spacing"] = tuple(kwargs["resampled_pixel_spacing"])
        return cls(**kwargs)

    def pyradiomics_settings(self) -> dict:
        return {
            "binWidth": self.bin_width,
            "force2D": True,
            "force2Ddimension": 0,
            "normalize": self.normalize,
            "resampledPixelSpacing": list(self.resampled_pixel_spacing) if self.resampled_pixel_spacing else None,
            "interpolator": self.interpolator,
            "label": self.label,
            "minimumROIDimensions": 2,
            "minimumROISize": None,
            "correctMask": False,
            "additionalInfo": True,
        }

    def settings_hash(self) -> str:
        """Identity of everything that changes feature values (settings + extractor versions)."""
        payload = {"settings": asdict(self), "pyradiomics": pyradiomics_version(), "extraction": EXTRACTION_VERSION}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()[:16]


def pyradiomics_version() -> str | None:
    try:
        import radiomics
    except ImportError:
        return None
    return radiomics.__version__


# --------------------------------------------------------------------------- #
# Geometry / masks
# --------------------------------------------------------------------------- #
def binarize_unet_output(output: np.ndarray, activation: str = UNET_ACTIVATION,
                         threshold: float = UNET_THRESHOLD) -> np.ndarray:
    """Logits ("sigmoid") or probabilities ("none") -> {0,1} uint8 mask, matching ``CADPipeline.segment``."""
    values = np.asarray(output, dtype=np.float64)
    if activation == "sigmoid":
        values = 1.0 / (1.0 + np.exp(-values))
    elif activation != "none":
        raise ValueError(f"unknown activation {activation!r}")
    return (values > threshold).astype(np.uint8)


def native_mask_from_roi(roi_mask: np.ndarray, padded_bbox, image_shape) -> np.ndarray:
    """ROI-crop-space mask (e.g. 256x256 U-Net output) -> mask aligned with the original image."""
    from reporting.features import roi_mask_to_native

    x1, y1, x2, y2 = (int(v) for v in padded_bbox)
    h, w = image_shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"ROI box {padded_bbox} lies outside the {w}x{h} image")
    return roi_mask_to_native(np.asarray(roi_mask), (x1, y1, x2, y2), image_shape)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0]
    raise ValueError(f"unsupported image shape {image.shape}")


@dataclass
class MaskQC:
    status: QCStatus = "pass"
    issues: list[dict] = field(default_factory=list)   # [{code, severity, message}]
    area_px: int = 0
    area_fraction: float = 0.0
    n_components: int = 0
    largest_component_fraction: float = 0.0
    bbox_xyxy: list[int] | None = None

    def add(self, code: str, severity: Literal["warn", "fail"], message: str) -> None:
        self.issues.append({"code": code, "severity": severity, "message": message})
        if severity == "fail":
            self.status = "fail"
        elif self.status == "pass":
            self.status = "warn"

    def to_dict(self) -> dict:
        return asdict(self)


def validate_mask(image: np.ndarray, mask: np.ndarray, settings: RadiomicsSettings,
                  roi_bbox=None) -> MaskQC:
    """Empty / too small / too large / fragmented / misaligned / truncated checks."""
    qc = MaskQC()
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3)):
        qc.add("unsupported_image_geometry", "fail", f"image shape {image.shape} is not HxW or HxWx3")
        return qc
    if image.dtype != np.uint8:
        qc.add("unexpected_image_dtype", "warn", f"expected uint8 ultrasound, got {image.dtype}")
    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        qc.add("shape_mismatch", "fail", f"mask {mask.shape} is not aligned with image {image.shape[:2]}")
        return qc
    unique = np.unique(mask)
    if not set(unique.tolist()) <= {0, 1, 255}:
        qc.add("non_binary_mask", "warn", f"mask has values {unique[:6].tolist()}...; binarized with > 0")
    binary = (mask > 0).astype(np.uint8)
    qc.area_px = int(binary.sum())
    qc.area_fraction = qc.area_px / binary.size
    if qc.area_px == 0:
        qc.add("empty_mask", "fail", "mask contains no lesion pixels")
        return qc
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    qc.n_components = n - 1
    qc.largest_component_fraction = float(stats[1:, cv2.CC_STAT_AREA].max() / qc.area_px)
    ys, xs = np.nonzero(binary)
    qc.bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if qc.area_px < settings.min_area_px:
        qc.add("mask_too_small", "fail", f"{qc.area_px} px < minimum {settings.min_area_px} px")
    if qc.area_fraction > settings.max_area_fraction:
        qc.add("mask_too_large", "warn", f"mask covers {qc.area_fraction:.0%} of the image")
    if qc.n_components > 1:
        qc.add("fragmented_mask", "warn", f"{qc.n_components} disconnected components")
    if qc.largest_component_fraction < settings.min_largest_component_fraction:
        qc.add("no_dominant_component", "warn",
               f"largest component is only {qc.largest_component_fraction:.0%} of the mask")
    h, w = binary.shape
    if xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
        qc.add("touches_image_border", "warn", "lesion touches the image border and may be truncated")
    if roi_bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in roi_bbox)
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            qc.add("roi_out_of_bounds", "fail", f"ROI box {list(roi_bbox)} lies outside the {w}x{h} image")
        else:
            outside = binary.copy()
            outside[y1:y2, x1:x2] = 0
            if outside.any():
                qc.add("mask_outside_roi", "fail", "mask pixels fall outside the ROI box (misaligned mask)")
            inner = binary[y1:y2, x1:x2]
            edge = np.concatenate([inner[0], inner[-1], inner[:, 0], inner[:, -1]])
            if edge.mean() > 0.15:
                qc.add("touches_roi_border", "warn", "mask runs along the ROI box edge; lesion may be cut off")
    return qc


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _coerce(value) -> float | None:
    """PyRadiomics scalar -> finite float, or None if non-finite / genuinely complex."""
    arr = np.asarray(value)
    if arr.size != 1:
        return None
    scalar = arr.reshape(()).item()
    if isinstance(scalar, complex):
        if abs(scalar.imag) > 1e-9 * max(1.0, abs(scalar.real)):
            return None
        scalar = scalar.real
    try:
        scalar = float(scalar)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _largest_component(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return binary
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def extract_radiomics_features(
    image: np.ndarray,
    mask: np.ndarray,
    metadata: dict | None = None,
    settings: RadiomicsSettings | None = None,
) -> dict:
    """Validate the mask and extract PyRadiomics features for one lesion.

    Args:
        image: Original-resolution ultrasound, HxW or HxWx3 (RGB) uint8.
        mask: Lesion mask aligned with ``image`` (HxW). Use ``native_mask_from_roi``
            first if it came from the U-Net in ROI-crop space.
        metadata: Optional ``{"roi_bbox": [x1,y1,x2,y2], "mask_source": ...}``.
        settings: Extraction settings; defaults to ``RadiomicsSettings()``.

    Returns:
        A JSON-ready dict: ``status`` ("ok" | "failed"), grouped ``features``,
        ``feature_status`` for invalid/missing values, ``qc``, ``preprocessing``,
        versions, ``settings_hash`` and machine-readable ``errors``. Never raises
        for bad masks or extractor failures.
    """
    settings = settings or RadiomicsSettings()
    metadata = metadata or {}
    result: dict = {
        "status": "failed",
        "extraction_version": EXTRACTION_VERSION,
        "extractor": "PyRadiomics",
        "extractor_version": pyradiomics_version(),
        "settings": settings.pyradiomics_settings() | {
            "feature_classes": list(settings.feature_classes), "image_types": list(settings.image_types)},
        "settings_hash": settings.settings_hash(),
        "unit_system": "pixel",
        "features": {},
        "feature_status": {},
        "qc": None,
        "preprocessing": {},
        "errors": [],
    }
    image = np.asarray(image)
    mask = np.asarray(mask)
    qc = validate_mask(image, mask, settings, metadata.get("roi_bbox"))
    result["qc"] = qc.to_dict()
    if qc.status == "fail":
        result["errors"] = [{"code": i["code"], "message": i["message"]} for i in qc.issues if i["severity"] == "fail"]
        return result
    if result["extractor_version"] is None:
        result["errors"].append({"code": "dependency_missing", "message": "pyradiomics is not installed"})
        return result

    gray = to_grayscale(image).astype(np.float32)
    binary = (mask > 0).astype(np.uint8)
    if settings.keep_largest_component:
        binary = _largest_component(binary)
    result["preprocessing"] = {
        "image_source": "original_resolution_grayscale",
        "color_conversion": "RGB->gray (OpenCV BT.601)" if image.ndim == 3 else "none",
        "intensity_range": [float(gray.min()), float(gray.max())],
        "intensity_normalization": "none" if not settings.normalize else "pyradiomics normalize",
        "resampling": "none" if settings.resampled_pixel_spacing is None else list(settings.resampled_pixel_spacing),
        "pixel_spacing": [1.0, 1.0],
        "pixel_spacing_note": "BUSI has no physical calibration; spacing set to 1 so shape features are in pixels.",
        "image_size_hw": list(gray.shape),
        "mask_component_policy": "largest_component" if settings.keep_largest_component else "all_components",
        "extracted_area_px": int(binary.sum()),
    }

    try:
        import radiomics
        import SimpleITK as sitk
        from radiomics import featureextractor

        radiomics.setVerbosity(logging.ERROR)
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings.pyradiomics_settings())
        extractor.disableAllFeatures()
        for name in settings.feature_classes:
            extractor.enableFeatureClassByName(name)
        extractor.disableAllImageTypes()
        for name in settings.image_types:
            extractor.enableImageTypeByName(name)
        sitk_image = sitk.GetImageFromArray(gray)
        sitk_mask = sitk.GetImageFromArray(binary)
        raw = extractor.execute(sitk_image, sitk_mask, label=settings.label)
    except Exception as exc:  # noqa: BLE001 - PyRadiomics raises plain ValueError/RuntimeError
        result["errors"].append({"code": "extraction_failed", "message": f"{type(exc).__name__}: {exc}"})
        return result

    features: dict[str, dict[str, float]] = {name: {} for name in settings.feature_classes}
    for key, value in raw.items():
        if key.startswith("diagnostics_"):
            continue
        image_type, family, feature = key.split("_", 2)
        name = family if image_type == "original" else f"{image_type}_{family}"
        coerced = _coerce(value)
        if coerced is None:
            result["feature_status"][f"{name}.{feature}"] = "invalid_nonfinite"
        else:
            features.setdefault(name, {})[feature] = coerced
    for family in settings.feature_classes:
        if not features.get(family):
            result["feature_status"][f"{family}.*"] = "unsupported"
    result["features"] = features
    result["diagnostics"] = {
        "mask_voxel_count": int(raw.get("diagnostics_Mask-original_VoxelNum", 0)),
        "mask_bbox_xywh": [int(v) for v in raw.get("diagnostics_Mask-original_BoundingBox", ())],
        "dimensionality": str(raw.get("diagnostics_Image-original_Dimensionality", "")),
    }
    result["status"] = "ok"
    return result


def flat_features(radiomics_block: dict) -> dict[str, float]:
    """``{"glcm": {"Contrast": x}}`` -> ``{"glcm.Contrast": x}`` (predictor naming)."""
    return {f"{family}.{name}": value
            for family, values in (radiomics_block.get("features") or {}).items()
            for name, value in values.items()}
