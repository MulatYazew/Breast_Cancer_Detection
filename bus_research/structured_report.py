"""Schema for a research ultrasound report.

Missing findings stay null. The model suggestion for BI-RADS assessment stays
null unless a physician types a final category. This schema does not accept a
model-finalized category.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from bus_research.review_workflow import DISCLAIMER

ReviewStatus = str


class ImageQualityBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str | None = None
    comments: list[str] = Field(default_factory=list)
    physician_review_status: str = "pending"


class DetectionBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lesion_detected: bool | None = None
    bounding_box: list[int] | None = None
    confidence: float | None = None
    physician_review_status: str = "pending"


class SegmentationBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mask_reference: str | None = None
    confidence: float | None = None
    physician_review_status: str = "pending"


class ClassificationBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicted_class: str | None = None
    probability: float | None = None
    physician_reviewed_class: str | None = None
    physician_review_status: str = "pending"
    label_source: str = "BUSI folder label when a reference label is present; otherwise unknown"


class DescriptorBlock(BaseModel):
    """Lexicon slots. Null means not annotated and not predicted."""

    model_config = ConfigDict(extra="forbid")
    tissue_composition: str | None = None
    shape: str | None = None
    orientation: str | None = None
    margin: list[str] = Field(default_factory=list)
    echo_pattern: str | None = None
    posterior_features: str | None = None
    calcifications: str | None = None
    associated_features: list[str] = Field(default_factory=list)
    special_case: str | None = None
    physician_review_status: str = "pending"
    unavailable_reason: str | None = (
        "No radiologist descriptor labels are in this dataset, so no descriptor model was trained."
    )


class AssessmentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_suggestion: str | None = None
    physician_final_assessment: str | None = None
    status: str = "pending"
    note: str = (
        "Only a physician may set physician_final_assessment. "
        "Categories 0 and 6 are not predicted from the image."
    )


class ReportText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: str = ""
    impression: str = ""
    limitations: list[str] = Field(default_factory=list)
    source_references: list[dict] = Field(default_factory=list)


class WorkflowBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    physician_approved: bool = False
    approval_timestamp: str | None = None
    model_versions: dict = Field(default_factory=dict)
    audit_log: list[dict] = Field(default_factory=list)


class StructuredStudyReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    study_id: str
    disclaimer: str = DISCLAIMER
    image_quality: ImageQualityBlock = Field(default_factory=ImageQualityBlock)
    detection: DetectionBlock = Field(default_factory=DetectionBlock)
    segmentation: SegmentationBlock = Field(default_factory=SegmentationBlock)
    classification: ClassificationBlock = Field(default_factory=ClassificationBlock)
    birads_descriptors: DescriptorBlock = Field(default_factory=DescriptorBlock)
    birads_assessment: AssessmentBlock = Field(default_factory=AssessmentBlock)
    auxiliary_geometry: dict | None = None
    report: ReportText = Field(default_factory=ReportText)
    workflow: WorkflowBlock = Field(default_factory=WorkflowBlock)


_CATEGORY_IN_PROSE = re.compile(r"\bbi-?rads\s*[0-6]\b", re.IGNORECASE)
_DESCRIPTOR_WORDS = re.compile(
    r"\b(hypoechoic|anechoic|spiculated|microlobulated|not parallel|circumscribed)\b",
    re.IGNORECASE,
)


def unsupported_claim_flags(report: StructuredStudyReport) -> list[str]:
    """Flag prose that states a finding the structured fields do not support."""
    flags = []
    text = f"{report.report.findings}\n{report.report.impression}"
    if report.birads_assessment.physician_final_assessment is None and _CATEGORY_IN_PROSE.search(text):
        flags.append("Prose names a BI-RADS category but physician_final_assessment is null.")
    if report.birads_assessment.model_suggestion is not None:
        flags.append("model_suggestion must stay null; this dataset has no assessment model.")
    descriptors_unset = (
        report.birads_descriptors.shape is None
        and report.birads_descriptors.echo_pattern is None
        and not report.birads_descriptors.margin
    )
    if descriptors_unset and _DESCRIPTOR_WORDS.search(text):
        flags.append("Prose states a BI-RADS descriptor that was not approved.")
    if report.workflow.physician_approved and report.birads_assessment.status != "physician_final":
        flags.append("The report is marked approved without a physician-final assessment status.")
    if report.detection.physician_review_status != "approved" and report.detection.lesion_detected:
        flags.append("A detection is present in the report without physician approval.")
    return flags
