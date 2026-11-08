"""In-memory research studies. Images are not written to disk.

Review routes require ``REVIEWER_TOKEN``. The service never writes a BI-RADS
category by itself.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

import numpy as np

from bus_research.geometry import geometry_from_mask
from bus_research.review_workflow import DISCLAIMER, ReviewWorkflow, WorkflowError
from bus_research.structured_report import (
    AssessmentBlock,
    ClassificationBlock,
    DescriptorBlock,
    DetectionBlock,
    ImageQualityBlock,
    ReportText,
    SegmentationBlock,
    StructuredStudyReport,
    WorkflowBlock,
    unsupported_claim_flags,
)
from preprocessing.roi_utils import apply_mask, crop_roi


class StudyError(RuntimeError):
    pass


def image_quality_comments(image: np.ndarray) -> list[str]:
    comments = []
    if image.size == 0 or min(image.shape[:2]) < 64:
        comments.append("Image is empty or smaller than 64 pixels on one side.")
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    if float(np.std(gray)) < 2:
        comments.append("Intensity variation is extremely low.")
    return comments


@dataclass
class Study:
    study_id: str
    image: np.ndarray
    workflow: ReviewWorkflow
    detection: dict = field(default_factory=dict)
    segmentation: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)
    geometry: dict | None = None
    assessment: str | None = None
    references: list[dict] = field(default_factory=list)


class StudyService:
    def __init__(self, pipeline=None) -> None:
        self.pipeline = pipeline
        self.studies: dict[str, Study] = {}

    def create(self, image: np.ndarray) -> Study:
        study_id = uuid.uuid4().hex[:12]
        study = Study(study_id=study_id, image=np.asarray(image), workflow=ReviewWorkflow(study_id))
        self.studies[study_id] = study
        return study

    def review(self, study_id: str, stage: str, action: str, reviewer_id: str, reason: str = "", detail: dict | None = None) -> dict:
        study = self._get(study_id)
        if stage == "assessment" and action in ("approve", "edit"):
            category = (detail or {}).get("physician_final_assessment")
            if category is None:
                raise StudyError("assessment approval requires physician_final_assessment in detail")
            from bus_research.birads import normalize_category

            study.assessment = normalize_category(category)
        entry = study.workflow.record(stage, action, reviewer_id, reason=reason, detail=detail)
        if stage == "detection" and action in ("reject", "uncertain"):
            study.detection = {}
            study.segmentation = {}
            study.classification = {}
            study.geometry = None
        return entry

    def run_detection(self, study_id: str) -> dict:
        study = self._get(study_id)
        study.workflow.assert_ready("detection")
        if self.pipeline is None:
            raise StudyError("No detection model is loaded.")
        found = self.pipeline.detect(study.image)
        if found is None:
            study.detection = {"lesion_detected": False, "bounding_box": None, "confidence": None, "source": "model"}
        else:
            box, confidence = found
            study.detection = {
                "lesion_detected": True,
                "bounding_box": [int(v) for v in box],
                "confidence": float(confidence),
                "source": "model",
            }
        return study.detection

    def run_segmentation(self, study_id: str) -> dict:
        study = self._get(study_id)
        study.workflow.assert_ready("segmentation")
        if not study.detection.get("lesion_detected"):
            raise StudyError("Segmentation requires an approved detection with a box.")
        if self.pipeline is None:
            raise StudyError("No segmentation model is loaded.")
        box = tuple(int(v) for v in study.detection["bounding_box"])
        roi, padded = crop_roi(study.image, box, pad_ratio=self.pipeline.bbox_pad_ratio, target_size=self.pipeline.roi_size)
        mask = self.pipeline.segment(roi)
        study.segmentation = {
            "mask_reference": "in-memory predicted mask",
            "roi_box": [int(v) for v in padded],
            "mask": mask,
            "roi": roi,
            "source": "model",
        }
        return {"mask_reference": study.segmentation["mask_reference"], "roi_box": study.segmentation["roi_box"]}

    def run_classification(self, study_id: str) -> dict:
        study = self._get(study_id)
        study.workflow.assert_ready("classification")
        if "mask" not in study.segmentation:
            raise StudyError("Classification requires a segmentation mask.")
        if self.pipeline is None:
            raise StudyError("No classification model is loaded.")
        masked = apply_mask(study.segmentation["roi"], study.segmentation["mask"])
        label, confidence, probs = self.pipeline.classify(masked)
        study.classification = {
            "predicted_class": label,
            "probability": confidence,
            "class_probs": probs,
            "source": "model",
        }
        study.geometry = geometry_from_mask(study.segmentation["mask"])
        return study.classification

    def build_report(self, study_id: str) -> StructuredStudyReport:
        study = self._get(study_id)
        study.workflow.assert_ready("report")
        quality = study.workflow.stages["image_quality"]
        detection_state = study.workflow.stages["detection"]
        segmentation_state = study.workflow.stages["segmentation"]
        class_state = study.workflow.stages["classification"]
        findings = []
        if detection_state["status"] == "approved" and study.detection.get("lesion_detected"):
            findings.append(
                f"Approved detection box {study.detection.get('bounding_box')} "
                f"(model confidence {study.detection.get('confidence')})."
            )
        if class_state["status"] == "approved" and study.classification:
            findings.append(
                f"Model class {study.classification.get('predicted_class')} "
                f"with probability {study.classification.get('probability')}. "
                "This is not a BI-RADS assessment."
            )
        else:
            findings.append("Classification was not approved.")
        findings.append("BI-RADS descriptors were not predicted: the dataset has no descriptor labels.")
        if study.assessment:
            impression = f"Physician-entered BI-RADS assessment: {study.assessment}."
            assessment_status = "physician_final"
        else:
            impression = "No physician BI-RADS assessment has been entered."
            assessment_status = "pending"
        report = StructuredStudyReport(
            study_id=study.study_id,
            image_quality=ImageQualityBlock(
                status="adequate" if quality["status"] == "approved" else quality["status"],
                comments=image_quality_comments(study.image),
                physician_review_status=quality["status"],
            ),
            detection=DetectionBlock(
                lesion_detected=study.detection.get("lesion_detected"),
                bounding_box=study.detection.get("bounding_box"),
                confidence=study.detection.get("confidence"),
                physician_review_status=detection_state["status"],
            ),
            segmentation=SegmentationBlock(
                mask_reference=study.segmentation.get("mask_reference"),
                confidence=None,
                physician_review_status=segmentation_state["status"],
            ),
            classification=ClassificationBlock(
                predicted_class=study.classification.get("predicted_class") if class_state["status"] == "approved" else None,
                probability=study.classification.get("probability") if class_state["status"] == "approved" else None,
                physician_reviewed_class=(class_state.get("detail") or {}).get("physician_reviewed_class"),
                physician_review_status=class_state["status"],
            ),
            birads_descriptors=DescriptorBlock(),
            birads_assessment=AssessmentBlock(
                model_suggestion=None,
                physician_final_assessment=study.assessment,
                status=assessment_status,
            ),
            auxiliary_geometry=study.geometry,
            report=ReportText(
                findings=" ".join(findings),
                impression=impression,
                limitations=[
                    DISCLAIMER,
                    "No patient identifier is available, so this is an image-level research record.",
                    "Descriptor heads were not trained.",
                    "Folder benign/malignant labels are not BI-RADS categories.",
                ],
                source_references=study.references,
            ),
            workflow=WorkflowBlock(
                physician_approved=study.workflow.approved("report") and study.assessment is not None,
                approval_timestamp=study.workflow.stages["report"]["updated_at"],
                model_versions={"detection": "yolo26n", "segmentation": "unet", "classification": "resnet50"},
                audit_log=list(study.workflow.audit_log),
            ),
        )
        flags = unsupported_claim_flags(report)
        if flags:
            raise StudyError("report failed the unsupported-claim check: " + "; ".join(flags))
        return report

    def _get(self, study_id: str) -> Study:
        try:
            return self.studies[study_id]
        except KeyError as exc:
            raise StudyError(f"unknown study {study_id}") from exc


def reviewer_token_ok(presented: str | None) -> bool:
    expected = os.environ.get("REVIEWER_TOKEN")
    if not expected:
        return False
    return presented is not None and presented == expected
