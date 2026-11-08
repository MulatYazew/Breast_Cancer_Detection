"""Geometry, surface distance, review gates, report schema, and classifier input shapes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bus_research.geometry import geometry_from_mask
from bus_research.review_workflow import ReviewWorkflow, WorkflowError
from bus_research.structured_report import StructuredStudyReport, unsupported_claim_flags
from evaluation.surface_distance import surface_distances
from reporting.llm_selection import MEDGEMMA_27B, assert_requested_model_can_load


def test_geometry_does_not_assign_orientation_for_a_disk():
    yy, xx = np.mgrid[:80, :80]
    mask = ((xx - 40) ** 2 + (yy - 40) ** 2) <= 18**2
    measured = geometry_from_mask(mask.astype(np.uint8))
    assert measured["empty_mask"] is False
    assert measured["area_px"] > 0
    assert measured["orientation_assignment"] is None
    assert "round" in measured["orientation_reason"].lower() or "no skin" in measured["orientation_reason"].lower()


def test_geometry_empty_mask():
    measured = geometry_from_mask(np.zeros((32, 32), dtype=np.uint8))
    assert measured["empty_mask"] is True
    assert measured["area_px"] == 0
    assert measured["orientation_assignment"] is None


def test_surface_distance_identical_and_empty():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 12:28] = 1
    same = surface_distances(mask, mask)
    assert same["defined"] is True
    assert same["hd95_px"] == pytest.approx(0.0, abs=1e-6)
    empty = surface_distances(mask, np.zeros_like(mask))
    assert empty["defined"] is False
    assert empty["hd95_px"] is None


def test_review_blocks_downstream_until_approval_and_clears_after_rejection():
    flow = ReviewWorkflow("study")
    with pytest.raises(WorkflowError):
        flow.record("detection", "approve", "reviewer-a")
    flow.record("image_quality", "approve", "reviewer-a")
    flow.record("detection", "approve", "reviewer-a", detail={"bounding_box": [1, 2, 3, 4]})
    flow.record("segmentation", "approve", "reviewer-a")
    flow.record("detection", "reject", "reviewer-a", reason="box missed the lesion")
    assert flow.status("detection") == "rejected"
    assert flow.status("segmentation") == "pending"
    with pytest.raises(WorkflowError):
        flow.record("classification", "approve", "reviewer-a")


def test_report_flags_invented_descriptor_and_category():
    report = StructuredStudyReport(study_id="s1")
    report.report.findings = "The mass is spiculated and BI-RADS 5."
    flags = unsupported_claim_flags(report)
    assert any("BI-RADS" in flag for flag in flags)
    assert any("descriptor" in flag for flag in flags)
    assert report.birads_assessment.model_suggestion is None
    assert report.birads_descriptors.shape is None


def test_medgemma_27b_is_refused_on_this_machine():
    with pytest.raises(RuntimeError, match="Refusing to download or load"):
        assert_requested_model_can_load(MEDGEMMA_27B, ram_gib_available=64)


def test_medgemma_4b_is_over_the_checkpoint_cap():
    from reporting.llm_selection import MEDGEMMA_4B

    with pytest.raises(RuntimeError, match="8 GB"):
        assert_requested_model_can_load(MEDGEMMA_4B, ram_gib_available=24)


def test_untrained_backbones_forward_and_mask_channel():
    from resnet.backbones import build_classifier

    for name in ("resnet50", "densenet121", "efficientnet_b0", "convnext_tiny"):
        model = build_classifier(name, pretrained=False, in_channels=4)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 4, 64, 64))
        assert out.shape == (1, 2)


def test_api_review_gate(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("REVIEWER_TOKEN", "test-token")
    from api.main import app
    from api.service import StudyService

    class FakePipeline:
        bbox_pad_ratio = 0.0
        roi_size = (32, 32)

        def detect(self, image):
            return (8, 8, 40, 40), 0.91

        def segment(self, roi):
            return np.ones(roi.shape[:2], dtype=np.uint8)

        def classify(self, masked):
            return "malignant", 0.66, {"benign": 0.34, "malignant": 0.66}

    app.state.service = StudyService(FakePipeline())
    client = TestClient(app)
    image = np.zeros((64, 80, 3), dtype=np.uint8)
    image[16:48, 20:60] = 80
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    created = client.post("/studies", files={"file": ("frame.png", buffer.getvalue(), "image/png")})
    assert created.status_code == 200
    study_id = created.json()["study_id"]
    assert "Research use only" in created.json()["disclaimer"]

    denied = client.post(
        f"/studies/{study_id}/reviews/image_quality",
        json={"action": "approve", "reviewer_id": "dr-1"},
    )
    assert denied.status_code == 401

    headers = {"x-reviewer-token": "test-token"}
    blocked = client.post(f"/studies/{study_id}/detection", headers=headers)
    assert blocked.status_code == 409

    approved = client.post(
        f"/studies/{study_id}/reviews/image_quality",
        headers=headers,
        json={"action": "approve", "reviewer_id": "dr-1"},
    )
    assert approved.status_code == 200
    detected = client.post(f"/studies/{study_id}/detection", headers=headers)
    assert detected.status_code == 200
    assert detected.json()["physician_review_status"] == "pending"
    assert detected.json()["detection"]["lesion_detected"] is True

    # Segmentation before detection approval must fail.
    too_soon = client.post(f"/studies/{study_id}/segmentation", headers=headers)
    assert too_soon.status_code == 409

    for stage in ("detection", "segmentation", "classification", "descriptors"):
        if stage == "segmentation":
            response = client.post(f"/studies/{study_id}/segmentation", headers=headers)
            assert response.status_code == 200
        if stage == "classification":
            response = client.post(f"/studies/{study_id}/classification", headers=headers)
            assert response.status_code == 200
        body = {"action": "approve", "reviewer_id": "dr-1", "detail": {}}
        response = client.post(f"/studies/{study_id}/reviews/{stage}", headers=headers, json=body)
        assert response.status_code == 200, response.text

    assessment = client.post(
        f"/studies/{study_id}/reviews/assessment",
        headers=headers,
        json={
            "action": "approve",
            "reviewer_id": "dr-1",
            "detail": {"physician_final_assessment": "4B"},
        },
    )
    assert assessment.status_code == 200
    client.post(
        f"/studies/{study_id}/reviews/report",
        headers=headers,
        json={"action": "approve", "reviewer_id": "dr-1"},
    )
    built = client.post(f"/studies/{study_id}/report", headers=headers)
    assert built.status_code == 200
    payload = built.json()
    assert payload["birads_assessment"]["model_suggestion"] is None
    assert payload["birads_assessment"]["physician_final_assessment"] == "4B"
    assert payload["birads_descriptors"]["shape"] is None
    assert payload["classification"]["predicted_class"] == "malignant"
    assert payload["auxiliary_geometry"]["orientation_assignment"] is None
