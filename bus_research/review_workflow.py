"""Staged physician review. Later stages cannot consume a rejected earlier stage.

Model output and physician approval are stored separately. Nothing in this
module assigns a BI-RADS category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

STAGES = (
    "image_quality",
    "detection",
    "segmentation",
    "classification",
    "descriptors",
    "assessment",
    "report",
)
ACTIONS = ("approve", "reject", "edit", "uncertain")
DISCLAIMER = (
    "Research use only — not validated for autonomous clinical diagnosis or patient management."
)


class WorkflowError(RuntimeError):
    """A stage was used before its predecessor was approved."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReviewWorkflow:
    """One study's approval state and audit log."""

    study_id: str
    stages: dict[str, dict] = field(default_factory=dict)
    audit_log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        for stage in STAGES:
            self.stages.setdefault(
                stage,
                {"status": "pending", "reviewer_id": None, "updated_at": None, "detail": {}},
            )

    def status(self, stage: str) -> str:
        self._check_stage(stage)
        return self.stages[stage]["status"]

    def approved(self, stage: str) -> bool:
        return self.status(stage) == "approved"

    def assert_ready(self, stage: str) -> None:
        """Refuse to enter ``stage`` until every earlier stage is approved."""
        self._check_stage(stage)
        index = STAGES.index(stage)
        for previous in STAGES[:index]:
            if self.stages[previous]["status"] != "approved":
                raise WorkflowError(
                    f"{stage} is blocked: {previous} is {self.stages[previous]['status']}, not approved"
                )

    def record(
        self,
        stage: str,
        action: str,
        reviewer_id: str,
        *,
        reason: str = "",
        detail: dict | None = None,
    ) -> dict:
        """Record a physician action. Reject or uncertain clears every later stage."""
        self._check_stage(stage)
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        if not reviewer_id or not str(reviewer_id).strip():
            raise ValueError("reviewer_id is required")
        self.assert_ready(stage)
        status = {"approve": "approved", "edit": "approved", "reject": "rejected", "uncertain": "uncertain"}[action]
        entry = {
            "stage": stage,
            "action": action,
            "status": status,
            "reviewer_id": str(reviewer_id),
            "reason": reason,
            "detail": detail or {},
            "timestamp": _now(),
        }
        self.stages[stage] = {
            "status": status,
            "reviewer_id": entry["reviewer_id"],
            "updated_at": entry["timestamp"],
            "detail": entry["detail"],
            "reason": reason,
            "action": action,
        }
        if action in ("reject", "uncertain", "edit"):
            self._clear_downstream(stage, entry["timestamp"], entry["reviewer_id"])
        self.audit_log.append(entry)
        return entry

    def _clear_downstream(self, stage: str, timestamp: str, reviewer_id: str) -> None:
        index = STAGES.index(stage)
        for later in STAGES[index + 1 :]:
            if self.stages[later]["status"] == "pending" and not self.stages[later]["detail"]:
                continue
            self.stages[later] = {
                "status": "pending",
                "reviewer_id": None,
                "updated_at": timestamp,
                "detail": {},
                "reason": f"cleared because {stage} was changed",
            }
            self.audit_log.append(
                {
                    "stage": later,
                    "action": "invalidate",
                    "status": "pending",
                    "reviewer_id": reviewer_id,
                    "reason": f"cleared because {stage} changed",
                    "detail": {},
                    "timestamp": timestamp,
                }
            )

    @staticmethod
    def _check_stage(stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
