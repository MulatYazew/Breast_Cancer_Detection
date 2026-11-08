"""Local research API. It does not finalize a diagnosis.

Start with the reviewer token set:

    export REVIEWER_TOKEN=local-research-token
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import io
import os

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from api.service import StudyError, StudyService, image_quality_comments, reviewer_token_ok
from bus_research.review_workflow import DISCLAIMER, WorkflowError

app = FastAPI(title="Breast ultrasound research API", version="0.1.0")
app.state.service = StudyService(pipeline=None)


def get_service() -> StudyService:
    return app.state.service


def require_reviewer(x_reviewer_token: str | None = Header(default=None)) -> None:
    if not reviewer_token_ok(x_reviewer_token):
        raise HTTPException(
            status_code=401,
            detail="Review routes require REVIEWER_TOKEN. The API does not accept unsigned approvals.",
        )


class ReviewBody(BaseModel):
    action: str
    reviewer_id: str = Field(min_length=1)
    reason: str = ""
    detail: dict = Field(default_factory=dict)


def _read_upload(payload: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            array = np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"unreadable image: {exc}") from exc
    return array


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "disclaimer": DISCLAIMER,
        "review_token_configured": bool(os.environ.get("REVIEWER_TOKEN")),
        "models_loaded": get_service().pipeline is not None,
    }


@app.post("/studies")
async def create_study(file: UploadFile = File(...), service: StudyService = Depends(get_service)) -> dict:
    array = _read_upload(await file.read())
    study = service.create(array)
    return {
        "study_id": study.study_id,
        "disclaimer": DISCLAIMER,
        "image_quality_comments": image_quality_comments(array),
        "image_quality_status": "pending",
        "note": "The upload name is discarded. Approve image quality before detection.",
    }


@app.post("/studies/{study_id}/reviews/{stage}")
def review_stage(
    study_id: str,
    stage: str,
    body: ReviewBody,
    _: None = Depends(require_reviewer),
    service: StudyService = Depends(get_service),
) -> dict:
    try:
        entry = service.review(study_id, stage, body.action, body.reviewer_id, body.reason, body.detail)
    except (StudyError, WorkflowError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"disclaimer": DISCLAIMER, "review": entry}


@app.post("/studies/{study_id}/detection")
def detect(study_id: str, _: None = Depends(require_reviewer), service: StudyService = Depends(get_service)) -> dict:
    try:
        result = service.run_detection(study_id)
    except (StudyError, WorkflowError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"disclaimer": DISCLAIMER, "detection": result, "physician_review_status": "pending"}


@app.post("/studies/{study_id}/segmentation")
def segment(study_id: str, _: None = Depends(require_reviewer), service: StudyService = Depends(get_service)) -> dict:
    try:
        result = service.run_segmentation(study_id)
    except (StudyError, WorkflowError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"disclaimer": DISCLAIMER, "segmentation": result, "physician_review_status": "pending"}


@app.post("/studies/{study_id}/classification")
def classify(study_id: str, _: None = Depends(require_reviewer), service: StudyService = Depends(get_service)) -> dict:
    try:
        result = service.run_classification(study_id)
    except (StudyError, WorkflowError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"disclaimer": DISCLAIMER, "classification": result, "physician_review_status": "pending"}


@app.post("/studies/{study_id}/report")
def report(study_id: str, _: None = Depends(require_reviewer), service: StudyService = Depends(get_service)) -> dict:
    try:
        built = service.build_report(study_id)
    except (StudyError, WorkflowError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return built.model_dump()
