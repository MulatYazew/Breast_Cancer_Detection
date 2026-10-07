"""Stage 6 data contracts: the LLM's draft (strictly validated) and the full report result."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ModelCategory = Literal["0", "2", "3", "4A", "4B", "4C", "5"]   # never "6" from the model
ReportStatus = Literal["ok", "invalid", "error"]


class RationaleItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    descriptor: str = Field(min_length=1, max_length=60)
    value: str = Field(min_length=1, max_length=80)
    supports: Literal["benign", "suspicious", "neutral"]
    source: str = Field(min_length=1, max_length=80)   # "kb:<chunk id>" or "feature:<descriptor key>"


class LLMDraft(BaseModel):
    """Exactly what the language model must return (as a single JSON object)."""

    model_config = ConfigDict(extra="forbid")
    category: ModelCategory
    rationale: list[RationaleItem] = Field(min_length=3, max_length=12)
    recommendation: str = Field(min_length=20, max_length=800)
    patient_summary: str = Field(min_length=60, max_length=1500)


# Clinical jargon, named diagnoses, category numbers and percentages don't belong in the patient-facing text.
PATIENT_JARGON = re.compile(
    r"bi-?rads|categor(?:y|ies)|malignan\w*|suspicio\w*|hypoechoic|anechoic|posterior|lexicon|\d+\s*%"
    # named diagnoses: the summary must not tell the patient what the lesion is
    r"|fibroadenoma\w*|carcinoma\w*",
    re.IGNORECASE,
)


class DraftValidationError(ValueError):
    """The draft is not valid JSON, violates the schema, or contradicts the computed evidence."""


def extract_json_object(text: str) -> dict:
    """Pull the first top-level JSON object out of a model response (tolerates ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        if start < 0:
            raise DraftValidationError("no JSON object found in the response")
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    break
        if candidate is None:
            raise DraftValidationError("unterminated JSON object in the response")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise DraftValidationError(f"invalid JSON: {exc}") from exc


def parse_draft(
    text: str,
    expected_category: str,
    allowed_sources: set[str],
    descriptor_values: dict[str, str],
) -> LLMDraft:
    """Parse and validate a draft against the schema *and* the computed evidence.

    Args:
        expected_category: The rule-based category; the draft must not change it.
        allowed_sources: ``kb:<id>`` for each retrieved chunk + ``feature:<key>``
            for each descriptor — citations outside this set are rejected.
        descriptor_values: descriptor key -> computed lexicon value; a rationale
            row citing ``feature:<key>`` must restate that exact value.
    """
    data = extract_json_object(text)
    try:
        draft = LLMDraft.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
        raise DraftValidationError(f"schema violation: {problems}") from exc

    problems: list[str] = []
    if draft.category != expected_category:
        problems.append(f"category must be {expected_category!r} (rule-based), got {draft.category!r}")
    for i, item in enumerate(draft.rationale):
        if item.source not in allowed_sources:
            problems.append(f"rationale[{i}].source {item.source!r} is not one of the provided ids")
        elif item.source.startswith("feature:"):
            key = item.source.split(":", 1)[1]
            if item.value.strip().lower() != descriptor_values.get(key, "").lower():
                problems.append(
                    f"rationale[{i}] cites {item.source} with value {item.value!r}, "
                    f"but the computed value is {descriptor_values.get(key)!r}"
                )
    if re.search(r"\b(category|bi-?rads)\s*6\b", draft.patient_summary + draft.recommendation, re.IGNORECASE):
        problems.append("category 6 may only be assigned by the clinician")
    jargon = PATIENT_JARGON.findall(draft.patient_summary)
    if jargon:
        problems.append(f"patient_summary must be plain language; remove: {sorted(set(j.lower() for j in jargon))}")
    if problems:
        raise DraftValidationError("; ".join(problems))
    return draft


@dataclass
class ReportResult:
    """Everything Stage 6 produced for one case: evidence, rule proposal, and (if valid) the draft."""

    case_id: str
    report_key: str
    status: ReportStatus
    category: str | None = None
    likelihood: str | None = None
    confidence: str | None = None
    confidence_reasons: list[str] = field(default_factory=list)
    rule_score: float | None = None
    rule_points: list[dict] = field(default_factory=list)
    malignant_probability: float | None = None
    rationale: list[dict] = field(default_factory=list)
    recommendation: str | None = None
    limitations: list[str] = field(default_factory=list)
    patient_summary: str | None = None
    descriptors: list[dict] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    knowledge_hits: list[dict] = field(default_factory=list)
    similar_cases: list[dict] = field(default_factory=list)
    model_id: str | None = None
    backend: str | None = None
    prompt_version: str | None = None
    kb_version: str | None = None
    mask_source: str | None = None
    bbox_source: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    attempts: int = 0
    error: str | None = None
    error_kind: str | None = None      # rate_limit | timeout | llm | validation | features
    generated_at: str | None = None

    @property
    def retrieved_ids(self) -> list[str]:
        return [h["id"] for h in self.knowledge_hits]

    @property
    def similar_case_ids(self) -> list[str]:
        return [c["id"] for c in self.similar_cases]

    def to_dict(self) -> dict:
        return asdict(self)
