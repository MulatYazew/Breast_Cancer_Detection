"""The single source of truth for chaining YOLO26n -> U-Net -> ResNet50.

``CADPipeline`` loads all three trained checkpoints once and exposes methods
that accept either the model's own output *or* a physician-supplied manual
override at each stage. ``inference.py`` (root CLI) and ``app/app.py``
both import this class rather than re-implementing the chaining logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from dataset.busi_dataset import CLASS_NAMES
from preprocessing.normalization import imagenet_normalize, load_image_rgb, to_unit_range
from preprocessing.roi_utils import apply_mask, crop_roi
from resnet.model import build_resnet50
from unet.model import build_unet
from yolo.model import load_trained_yolo
from yolo.predict import detect_top_box

Bbox = tuple[int, int, int, int]
Source = Literal["model", "manual"]


@dataclass
class CaseResult:
    """Full per-case output of the cascaded pipeline, stage-by-stage."""

    image: np.ndarray
    bbox: Bbox
    bbox_source: Source
    detection_confidence: float | None
    roi_crop: np.ndarray
    mask: np.ndarray
    mask_source: Source
    masked_roi: np.ndarray
    predicted_class: str
    confidence: float
    class_probs: dict[str, float]
    stage_sources: dict[str, Source] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable summary (arrays excluded) — the audit-log record."""
        return {
            "bbox": list(self.bbox),
            "bbox_source": self.bbox_source,
            "detection_confidence": self.detection_confidence,
            "mask_source": self.mask_source,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_probs": self.class_probs,
        }


def to_chw_tensor(image_float01: np.ndarray, device: torch.device) -> torch.Tensor:
    """ImageNet-normalize an HxWx3 float32 [0,1] image and return a (1,C,H,W) tensor.

    Public so callers outside this module (the Streamlit app's Grad-CAM panel)
    can prepare a model-input tensor identically to ``segment``/``classify``.
    """
    normalized = imagenet_normalize(image_float01)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).float().unsqueeze(0)
    return tensor.to(device)


class CADPipeline:
    """Loads YOLO26n, U-Net, and ResNet50 once; chains detect -> crop -> segment -> classify."""

    def __init__(self, config: dict, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        inf_cfg = config["inference"]
        prep_cfg = config["preprocessing"]
        unet_cfg = config["unet"]
        resnet_cfg = config["resnet"]

        self.roi_size = tuple(prep_cfg["roi_size"])
        self.conf_threshold = inf_cfg["conf_threshold"]
        self.iou_threshold = inf_cfg["iou_threshold"]
        self.bbox_pad_ratio = prep_cfg["bbox_pad_ratio"]

        self.yolo_model = load_trained_yolo(inf_cfg["yolo_weights"])

        self.unet_model = build_unet(
            architecture=unet_cfg["architecture"],
            encoder_name=unet_cfg["encoder"],
            encoder_weights=None,
            in_channels=unet_cfg["in_channels"],
            classes=unet_cfg["classes"],
        ).to(self.device)
        unet_checkpoint = torch.load(inf_cfg["unet_weights"], map_location=self.device)
        self.unet_model.load_state_dict(unet_checkpoint["model_state"])
        self.unet_model.eval()

        self.resnet_model = build_resnet50(pretrained=False, num_classes=resnet_cfg["num_classes"]).to(
            self.device
        )
        resnet_checkpoint = torch.load(inf_cfg["resnet_weights"], map_location=self.device)
        self.resnet_model.load_state_dict(resnet_checkpoint["model_state"])
        self.resnet_model.eval()

    def detect(self, image: np.ndarray) -> tuple[Bbox, float] | None:
        """Stage 2: run YOLO26n, return the top (bbox, confidence) or None."""
        return detect_top_box(
            self.yolo_model,
            image,
            imgsz=self.config["yolo"]["imgsz"],
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )

    def segment(self, roi_crop: np.ndarray) -> np.ndarray:
        """Stage 3: run U-Net on an ROI crop, return a binary mask (same size as ``roi_crop``)."""
        image_float = to_unit_range(roi_crop)
        tensor = to_chw_tensor(image_float, self.device)
        with torch.no_grad():
            logits = self.unet_model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return (prob > 0.5).astype(np.uint8)

    def classify(self, masked_roi: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Stage 4: run ResNet50 on a masked ROI, return (predicted_class, confidence, probs)."""
        image_float = to_unit_range(masked_roi)
        tensor = to_chw_tensor(image_float, self.device)
        with torch.no_grad():
            logits = self.resnet_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        class_probs = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
        return CLASS_NAMES[pred_idx], float(probs[pred_idx]), class_probs

    def run(
        self,
        image_path: str | Path,
        manual_bbox: Bbox | None = None,
        manual_mask: np.ndarray | None = None,
    ) -> CaseResult:
        """Full Stage 2->3->4 cascade for one image.

        Args:
            image_path: Path to a raw ultrasound image.
            manual_bbox: If given, skip detection and use this physician-drawn
                box instead of the YOLO prediction.
            manual_mask: If given (same size as the ROI crop that would result
                from ``manual_bbox`` or the detected box), skip segmentation
                and use this physician-corrected mask instead of U-Net's.
        """
        image = load_image_rgb(image_path)
        return self._run_on_array(image, manual_bbox=manual_bbox, manual_mask=manual_mask)

    def _run_on_array(
        self,
        image: np.ndarray,
        manual_bbox: Bbox | None = None,
        manual_mask: np.ndarray | None = None,
    ) -> CaseResult:
        stage_sources: dict[str, Source] = {}

        if manual_bbox is not None:
            bbox = manual_bbox
            bbox_source: Source = "manual"
            detection_confidence = None
        else:
            detection = self.detect(image)
            if detection is None:
                raise RuntimeError("YOLO26n found no tumor candidate; a manual bbox is required")
            bbox, detection_confidence = detection
            bbox_source = "model"
        stage_sources["detection"] = bbox_source

        roi_crop, padded_bbox = crop_roi(
            image, bbox, pad_ratio=(0.0 if bbox_source == "manual" else self.bbox_pad_ratio),
            target_size=self.roi_size,
        )

        if manual_mask is not None:
            mask = manual_mask
            mask_source: Source = "manual"
        else:
            mask = self.segment(roi_crop)
            mask_source = "model"
        stage_sources["segmentation"] = mask_source

        masked_roi = apply_mask(roi_crop, mask)
        predicted_class, confidence, class_probs = self.classify(masked_roi)
        stage_sources["classification"] = "model"

        return CaseResult(
            image=image,
            bbox=padded_bbox,
            bbox_source=bbox_source,
            detection_confidence=detection_confidence,
            roi_crop=roi_crop,
            mask=mask,
            mask_source=mask_source,
            masked_roi=masked_roi,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probs=class_probs,
            stage_sources=stage_sources,
        )

    def run_from_manual_roi(
        self,
        image: np.ndarray,
        manual_bbox: Bbox,
        manual_mask: np.ndarray | None = None,
    ) -> CaseResult:
        """Convenience entry point for the Streamlit app: image already in memory
        (uploaded file), manual bbox always given (physician rejected detection).
        """
        return self._run_on_array(image, manual_bbox=manual_bbox, manual_mask=manual_mask)

    def resegment(self, roi_crop: np.ndarray) -> np.ndarray:
        """Re-run just Stage 3 on an already-cropped ROI (used after a physician
        rejects the detected box and draws a new one, before mask review)."""
        return self.segment(roi_crop)

    def reclassify(self, roi_crop: np.ndarray, mask: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Re-run just Stage 4 after a physician corrects the segmentation mask."""
        masked_roi = apply_mask(roi_crop, mask)
        return self.classify(masked_roi)


def build_pipeline(config: dict) -> CADPipeline:
    """Convenience factory using ``utils.seed.get_device`` for device selection."""
    from utils.seed import get_device

    device = get_device(config.get("device"))
    return CADPipeline(config, device=device)
