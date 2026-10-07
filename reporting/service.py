"""Stage 6 orchestration: approved mask -> features -> lexicon -> rules -> retrieval -> LLM -> validated report.

``ReportService.generate`` is the only entry point the Streamlit app calls; no
reporting logic lives in app.py. The service is constructed once (the app
caches it with ``st.cache_resource``) and holds the loaded embedding model,
FAISS indexes and LLM.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from inference.pipeline import CaseResult
from preprocessing.roi_utils import apply_mask
from reporting import report_key
from reporting.features import EmptyMaskError, extract_features
from reporting.lexicon import map_to_lexicon
from reporting.llm import ChatLLM, LLMError
from reporting.prompt import PROMPT_VERSION, build_messages, retry_message
from reporting.retrieval import CaseIndex, KnowledgeIndex
from reporting.schema import DraftValidationError, ReportResult, parse_draft
from reporting.scoring import assess

# Progress steps, in order (shown verbatim by the app's st.status).
STEPS = (
    "Extracting radiomic features",
    "Mapping to BI-RADS lexicon",
    "Retrieving knowledge and similar cases",
    "Drafting report",
    "Validating output",
)

BASE_LIMITATIONS = [
    "Automated analysis of a single B-mode still image; real-time scanning, Doppler, elastography, "
    "prior imaging and clinical history were not available.",
    "Lexicon thresholds are heuristics calibrated on the BUSI training set, not validated clinical cut-offs.",
]


class ReportService:
    def __init__(
        self,
        reporting_cfg: dict,
        llm: ChatLLM,
        knowledge_index: KnowledgeIndex,
        kb_version: str,
        case_index: CaseIndex | None = None,
        case_embedder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.cfg = reporting_cfg
        self.llm = llm
        self.knowledge_index = knowledge_index
        self.kb_version = kb_version
        self.case_index = case_index
        self.case_embedder = case_embedder

    def generate(
        self,
        case_result: CaseResult,
        approved_mask: np.ndarray,
        case_id: str = "",
        progress: Callable[[str], None] | None = None,
    ) -> ReportResult:
        """Generate a draft report from the clinician-approved mask.

        Args:
            case_result: The case as it stands after clinician review; its
                ``bbox`` must be the padded box the ROI crop came from.
            approved_mask: Clinician-approved binary mask in ROI-crop space.
                Features always come from this, never from a rejected model mask.
        """
        notify = progress or (lambda _step: None)
        timings: dict[str, float] = {}
        result = ReportResult(
            case_id=case_id,
            report_key=report_key(approved_mask, case_result.bbox),
            status="error",
            model_id=self.llm.model_id,
            backend=self.llm.backend,
            prompt_version=PROMPT_VERSION,
            kb_version=self.kb_version,
            mask_source=case_result.mask_source,
            bbox_source=case_result.bbox_source,
            malignant_probability=case_result.class_probs.get("malignant"),
            generated_at=datetime.now(timezone.utc).isoformat(),
            timings=timings,
        )

        def timed(step: str):
            notify(step)
            return _Timer(timings, step)

        with timed(STEPS[0]):
            try:
                features = extract_features(case_result.image, case_result.bbox, approved_mask)
            except EmptyMaskError as exc:
                result.error, result.error_kind = f"Cannot extract features: {exc}", "features"
                return result
            result.features = features.to_dict()

        with timed(STEPS[1]):
            descriptors = map_to_lexicon(features)
            rules = assess(descriptors, features, result.malignant_probability, case_result.mask_source)
            result.descriptors = [d.to_dict() for d in descriptors]
            result.category, result.likelihood = rules.category, rules.likelihood
            result.confidence, result.confidence_reasons = rules.confidence, rules.confidence_reasons
            result.rule_score, result.rule_points = rules.score, rules.points
            result.limitations = build_limitations(descriptors, rules)

        with timed(STEPS[2]):
            assessable = [d for d in descriptors if d.assessable]
            query = "Breast ultrasound mass: " + "; ".join(f"{d.category} {d.value}" for d in assessable)
            pinned = [d.kb_id for d in assessable if d.kb_id] + [f"cat-{rules.category.lower()}"]
            if any(not d.assessable for d in descriptors):
                pinned.append("us-not-assessable-bmode")
            result.knowledge_hits = self.knowledge_index.retrieve(query, self.cfg["top_k_knowledge"], pinned)
            if self.case_index is not None and self.case_embedder is not None:
                embedding = self.case_embedder(apply_mask(case_result.roi_crop, approved_mask))
                result.similar_cases = self.case_index.search(embedding, self.cfg["top_k_cases"])

        allowed = {f"kb:{h['id']}" for h in result.knowledge_hits} | {f"feature:{d.key}" for d in descriptors}
        values = {d.key: d.value for d in descriptors}
        messages = build_messages(descriptors, rules, result.knowledge_hits, result.similar_cases)
        max_attempts = 1 + int(self.cfg["llm"]["max_retries"])
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            result.attempts = attempt
            with timed(STEPS[3]):
                try:
                    raw = self.llm.generate(messages)
                except LLMError as exc:
                    result.error, result.error_kind = str(exc), exc.kind
                    return result
            with timed(STEPS[4]):
                try:
                    draft = parse_draft(raw, rules.category, allowed, values)
                except DraftValidationError as exc:
                    last_error = str(exc)
                    messages = messages + [{"role": "assistant", "content": raw}, retry_message(last_error)]
                    continue
            result.status = "ok"
            result.rationale = [item.model_dump() for item in draft.rationale]
            result.recommendation = draft.recommendation
            result.patient_summary = draft.patient_summary
            return result

        result.status = "invalid"
        result.error = f"The model output failed validation after {max_attempts} attempt(s): {last_error}"
        result.error_kind = "validation"
        return result


class _Timer:
    def __init__(self, timings: dict[str, float], step: str) -> None:
        self.timings, self.step = timings, step

    def __enter__(self) -> "_Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.timings[self.step] = round(self.timings.get(self.step, 0.0) + time.perf_counter() - self.start, 3)


def build_limitations(descriptors, rules) -> list[str]:
    """Deterministic, so the limitations can never contradict the computed evidence."""
    not_assessable = [d.category for d in descriptors if not d.assessable]
    items = list(BASE_LIMITATIONS)
    if not_assessable:
        items.append("Not assessable here: " + ", ".join(not_assessable) + ".")
    items += [f"{d.category}: {d.note}" for d in descriptors if d.assessable and d.note]
    items += rules.confidence_reasons
    return _dedupe(items)


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item.strip() and item.strip().lower() not in seen:
            seen.add(item.strip().lower())
            out.append(item.strip())
    return out


def build_report_service(
    config: dict,
    hf_token: str | None = None,
    case_embedder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ReportService:
    """Load the embedding model, FAISS indexes and LLM (slow; call once and cache)."""
    from reporting.llm import build_llm
    from reporting.retrieval import CaseIndex, KnowledgeIndex, load_knowledge, sentence_embedder
    from utils.config import REPO_ROOT

    cfg = config["reporting"]
    chunks, kb_version = load_knowledge(REPO_ROOT / cfg["knowledge_dir"])
    knowledge_index = KnowledgeIndex(chunks, sentence_embedder(cfg["embedding_model_id"]))
    case_index = CaseIndex.load_if_present(REPO_ROOT / cfg["index_dir"])
    llm = build_llm(cfg, device=config.get("device"), hf_token=hf_token)
    return ReportService(cfg, llm, knowledge_index, kb_version, case_index=case_index, case_embedder=case_embedder)
