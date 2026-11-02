"""Evidence-grounded, schema-validated LLM report for one lesion record ("code states facts, LLM explains").

The report's factual fields — ``observed_features``,
``quantitative_radiomics_summary``, ``classifier_output``, ``birads_assessment``
and ``requires_radiologist_review`` — are filled deterministically from the
validated record and classifier output. The LLM writes only the narrative
(``summary``, ``supporting_evidence``, ``uncertainty_and_limitations``,
``recommended_review``), and ``validate_narrative`` rejects any narrative that

* cites a chunk id that was not retrieved,
* states a BI-RADS category (none is available without validated labels),
* turns a model output into a biopsy/surgery/follow-up directive,
* quotes a decimal number that is not in the facts, or
* describes a lexicon finding (e.g. "spiculated") that was not observed.

If extraction failed or the mask failed QC, the service abstains without
calling the LLM. Payloads hold only pseudonymous ids and numbers.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bus_research import DISCLAIMER
from bus_research.birads import US_DESCRIPTORS, unavailable_assessment

PROMPT_VERSION = "bus-report-v3"

# Units for the shape features shown in the summary (spacing is 1 px, uncalibrated).
KEY_FEATURES = {
    "shape2D.PixelSurface": "px^2",
    "shape2D.Perimeter": "px",
    "shape2D.MajorAxisLength": "px",
    "shape2D.MinorAxisLength": "px",
    "shape2D.Elongation": "ratio",
    "shape2D.Sphericity": "ratio",
    "firstorder.Mean": "gray level",
    "firstorder.Entropy": "bits",
    "glcm.Contrast": "unitless",
    "glcm.Idm": "unitless",
    "glszm.ZoneEntropy": "bits",
}
# Lexicon terms that describe *this lesion*; generic words (oval, round, soft...) are excluded.
_DISTINCTIVE_TERMS = sorted({t for key in ("shape", "orientation", "margin", "echo_pattern", "posterior_features")
                             for t in US_DESCRIPTORS[key]} - {"oval", "round", "no posterior features"},
                            key=len, reverse=True)
BIRADS_CLAIM = re.compile(r"bi-?rads\s*(?:category\s*)?[0-6]|\bcategory\s*[0-6][abc]?\b|\b4[abc]\b", re.IGNORECASE)
DIRECTIVE = re.compile(
    r"(indicat|recommend|requir|necessar|should|must|need)\w*\W+(?:\w+\W+){0,6}?"
    r"(biops|surg|excision|aspiration|mastectom|lumpectom|follow-?up|further evaluation|additional imaging)"
    r"|(biops|surg|excision|aspiration|mastectom|lumpectom|follow-?up|further evaluation|additional imaging)"
    r"\w*\W+(?:\w+\W+){0,4}?(is|are|may be|might be|could be)\s+(indicated|required|recommended|necessary|"
    r"warranted|needed)",
    re.IGNORECASE)
# Evidence items are for reference knowledge; computed values citing a reference would borrow its authority.
COMPUTED_VALUE = re.compile(r"\d+\.\d+|\d+\s*%|\b(?:shape2D|firstorder|glcm|glrlm|glszm|gldm|ngtdm)\.|p\(malignant\)",
                            re.IGNORECASE)
DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d+)(%?)")


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=10, max_length=400)
    citations: list[str] = Field(min_length=1, max_length=4)


class Narrative(BaseModel):
    """Exactly what the LLM must return."""

    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=40, max_length=1500)
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list, max_length=8)
    uncertainty_and_limitations: list[str] = Field(min_length=1, max_length=10)
    recommended_review: str = Field(min_length=20, max_length=600)


class NarrativeValidationError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #
def _sig(x: float) -> float:
    return float(f"{x:.6g}")


def build_facts(record: dict, prediction: dict, radiologist_assessment: dict | None = None) -> dict:
    """Everything the report may state, computed without the LLM."""
    feats = record["radiomics"].get("features") or {}
    quantitative = []
    for name, unit in KEY_FEATURES.items():
        family, feature = name.split(".")
        value = feats.get(family, {}).get(feature)
        if value is not None:
            quantitative.append({"feature": name, "value": _sig(value), "unit": unit})
    observed = []
    for key, d in record["birads_descriptors"].items():
        observed.append({"descriptor": key, "status": d["status"], "source": d["source"],
                         "value": d["value"] if d["status"] == "observed" else None})
    if prediction.get("available"):
        p = prediction["class_probabilities"]["malignant"]
        classifier = {"available": True, "model_id": prediction["model_id"],
                      "prediction_type": prediction["prediction_type"],
                      "p_malignant": round(p, 3), "calibration_status": prediction["calibration_status"],
                      "out_of_distribution": bool((prediction.get("ood") or {}).get("out_of_distribution")),
                      "label": "Research model output — not a clinical diagnosis."}
    else:
        classifier = {"available": False, "reason": prediction.get("reason", "no compatible trained model")}
    if radiologist_assessment and radiologist_assessment.get("category"):
        birads = {"status": "provided_by_radiologist", "category": radiologist_assessment["category"],
                  "source": "radiologist", "reason": "Entered by the reviewing radiologist."}
    else:
        birads = unavailable_assessment()
    qc = record["segmentation"]
    limitations = list(record.get("limitations") or [])
    if not any("pixel" in item.lower() for item in limitations):
        limitations.append("Shape features are in pixels; BUSI images have no physical calibration.")
    if qc["quality_status"] == "warn":
        limitations += [f"Mask QC: {i['message']}" for i in qc.get("quality_issues", [])]
    if classifier.get("out_of_distribution"):
        limitations.append("Radiomics features are outside the training distribution; the classifier "
                           "output is unreliable for this lesion.")
    if classifier.get("available") and classifier["calibration_status"] != "calibrated":
        limitations.append("Probability calibration unavailable.")
    if not any(d["status"] == "observed" for d in observed):
        limitations.append("No BI-RADS descriptors were annotated; descriptors are reported as not assessed.")
    return {
        "record_id": record["record_id"],
        "mask_source": qc["mask_source"],
        "mask_quality": qc["quality_status"],
        "observed_features": observed,
        "quantitative_radiomics_summary": quantitative,
        "classifier_output": classifier,
        "birads_assessment": birads,
        "known_limitations": limitations,
    }


def abstain_reason(record: dict) -> str | None:
    if record["radiomics"]["status"] != "ok":
        errs = "; ".join(e["message"] for e in record["radiomics"].get("errors", [])) or "unknown error"
        return f"Radiomics extraction failed ({errs})."
    if record["segmentation"]["quality_status"] == "fail":
        return "The lesion mask failed quality control."
    return None


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You write the narrative part of a research report about one breast ultrasound lesion
for a radiologist to review. The facts are already computed; you explain them and never change them.
Rules:
- Output exactly one JSON object with keys: summary, supporting_evidence, uncertainty_and_limitations,
  recommended_review. Nothing else.
- Use only the FACTS and EVIDENCE given. Do not invent findings, measurements, history, Doppler or
  elastography results. Descriptors whose status is not "observed" must be called "not assessed".
- Never state or imply a BI-RADS category or subcategory (no "BI-RADS 4", "category 3", "4A").
- The classifier is a benign-vs-malignant research model, not a BI-RADS assessment. If you mention its
  probability, copy the number exactly and say whether it is calibrated.
- Never say that biopsy, surgery, aspiration, follow-up, further evaluation or additional imaging is
  indicated, warranted, required or recommended.
  recommended_review describes what a radiologist should review, not patient management.
- supporting_evidence holds ONLY statements taken from EVIDENCE text (no numbers, no feature names, no
  classifier output). Each item cites 1-4 ids copied
  exactly from ALLOWED_CITATION_IDS. Never cite FACTS keys. Computed measurements belong in summary.
- uncertainty_and_limitations is a list of plain strings; recommended_review is one plain string.
- Do not use lexicon words such as irregular, spiculated, angular, indistinct, hypoechoic or shadowing
  to describe this lesion: no descriptor was observed.
- Quote numbers only as they appear in FACTS.
Output shape (types are exact):
{"summary": "string", "supporting_evidence": [{"claim": "string", "citations": ["<allowed id>"]}],
 "uncertainty_and_limitations": ["string", "string"], "recommended_review": "string"}"""


def build_messages(facts: dict, hits: list[dict]) -> list[dict]:
    evidence = [{"id": h["chunk_id"], "tier": h["evidence_tier"], "source": h["citation"],
                 "page": h["page"], "section": h["section"], "text": h["text"]} for h in hits]
    user = ("FACTS_JSON:\n" + json.dumps(facts, indent=1) + "\n\nEVIDENCE:\n" + json.dumps(evidence, indent=1)
            + "\n\nALLOWED_CITATION_IDS: " + json.dumps([h["chunk_id"] for h in hits])
            + "\n\nWrite the JSON object now.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def retry_message(error: str) -> dict:
    return {"role": "user", "content": f"Your JSON failed validation: {error}\nReturn the corrected JSON object only."}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _fact_numbers(facts: dict) -> set[float]:
    nums: set[float] = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            nums.add(float(x))
    walk(facts)
    return nums


def _number_supported(text_value: str, percent: bool, facts_nums: set[float]) -> bool:
    """A quoted decimal must equal some fact (or fact x 100 for a percentage) to its quoted precision."""
    value = float(text_value)
    tol = 0.5 * 10 ** -len(text_value.split(".")[1]) + 1e-9
    scales = (100.0,) if percent else (1.0, 100.0)
    return any(abs(fact * s - value) <= tol for fact in facts_nums for s in scales)


def validate_narrative(raw: str, facts: dict, allowed_citations: set[str]) -> Narrative:
    from reporting.schema import DraftValidationError, extract_json_object

    try:
        data = extract_json_object(raw)
    except DraftValidationError as exc:
        raise NarrativeValidationError(str(exc)) from exc
    try:
        narrative = Narrative.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
        raise NarrativeValidationError(f"schema violation: {problems}") from exc
    problems: list[str] = []
    texts = [narrative.summary, narrative.recommended_review, *narrative.uncertainty_and_limitations,
             *(e.claim for e in narrative.supporting_evidence)]
    for i, item in enumerate(narrative.supporting_evidence):
        bad = [c for c in item.citations if c not in allowed_citations]
        if bad:
            problems.append(f"supporting_evidence[{i}] cites unknown ids {bad}")
        if COMPUTED_VALUE.search(item.claim):
            problems.append(f"supporting_evidence[{i}] restates a computed value; evidence claims must come from "
                            "the reference text (put measurements in summary)")
    joined = " \n".join(texts)
    if facts["birads_assessment"]["status"] == "unavailable" and BIRADS_CLAIM.search(joined):
        problems.append(f"states a BI-RADS category ({BIRADS_CLAIM.search(joined).group(0)!r}) but none is available")
    if DIRECTIVE.search(joined):
        problems.append(f"contains a management directive ({DIRECTIVE.search(joined).group(0)!r})")
    nums = _fact_numbers(facts)
    for text in (narrative.summary, *(e.claim for e in narrative.supporting_evidence)):
        for value, pct in DECIMAL.findall(text):
            if not _number_supported(value, bool(pct), nums):
                problems.append(f"number {value}{pct} is not in the facts")
    observed = {(d["value"] or "").lower() for d in facts["observed_features"] if d["status"] == "observed"}
    summary_low = narrative.summary.lower()
    for term in _DISTINCTIVE_TERMS:
        if term in observed:
            continue
        for m in re.finditer(rf"\b{re.escape(term)}\b", summary_low):
            # "not parallel" contains "parallel"; ignore the substring hit and negated mentions.
            window = summary_low[max(0, m.start() - 25): m.end() + 20]
            if "not assessed" in window or "not annotated" in window or "not available" in window:
                continue
            if term == "parallel" and summary_low[max(0, m.start() - 4): m.start()] == "not ":
                continue
            problems.append(f"summary describes {term!r}, which was not observed")
            break
    if problems:
        raise NarrativeValidationError("; ".join(dict.fromkeys(problems)))
    return narrative


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def template_narrative(facts: dict, hits: list[dict]) -> dict:
    """Deterministic narrative used by the offline stub provider and as the abstention text."""
    clf = facts["classifier_output"]
    quant = {q["feature"]: q for q in facts["quantitative_radiomics_summary"]}
    parts = [f"Quantitative analysis of the lesion outlined by the {facts['mask_source']} mask "
             f"(mask quality: {facts['mask_quality']})."]
    if "shape2D.PixelSurface" in quant:
        parts.append(f"The outlined area is {quant['shape2D.PixelSurface']['value']} px^2 "
                     "(pixel units; no physical calibration).")
    if clf.get("available"):
        parts.append(f"The radiomics benign-vs-malignant research model gives p(malignant) = {clf['p_malignant']} "
                     f"(calibration status: {clf['calibration_status']}).")
    else:
        parts.append("No compatible benign-vs-malignant model output is available.")
    parts.append("BI-RADS descriptors were not assessed and no BI-RADS assessment is available.")
    evidence = []
    by_tier = sorted(hits, key=lambda h: (h["evidence_tier"] != "primary_reference", -h["score"]))
    if by_tier:
        evidence.append({"claim": "The ACR BI-RADS ultrasound lexicon describes masses by shape, orientation, "
                                  "margin, echo pattern and posterior features, which require radiologist "
                                  "assessment.", "citations": [by_tier[0]["chunk_id"]]})
    return {
        "summary": " ".join(parts),
        "supporting_evidence": evidence,
        "uncertainty_and_limitations": facts["known_limitations"] or ["Single grayscale still image only."],
        "recommended_review": "A radiologist should review the segmentation outline, assess the BI-RADS "
                              "descriptors directly on the images, and interpret the research model output "
                              "in clinical context.",
    }


class StubLLM:
    """Offline deterministic provider: reads FACTS_JSON from the prompt, returns the template narrative."""

    backend = "stub"
    model_id = "template-narrative-v1"

    def generate(self, messages: list[dict]) -> str:
        user = messages[-1]["content"] if messages[-1]["role"] == "user" else messages[1]["content"]
        facts_text = user.split("FACTS_JSON:\n", 1)[1].split("\n\nEVIDENCE:\n", 1)
        facts = json.loads(facts_text[0])
        hits_raw = json.loads(facts_text[1].split("\n\nALLOWED_CITATION_IDS:")[0])
        hits = [{"chunk_id": h["id"], "evidence_tier": h["tier"], "score": 0.0} for h in hits_raw]
        return json.dumps(template_narrative(facts, hits))


def build_provider(config: dict, hf_token: str | None = None):
    """``research.reports.llm_backend``: stub | local | hf_api. hf_api sends data off-machine,
    so it also needs ``research.reports.allow_external_llm: true``."""
    rcfg = config["research"]["reports"]
    backend = rcfg.get("llm_backend", "stub")
    if backend == "stub":
        return StubLLM()
    if backend == "hf_api" and not rcfg.get("allow_external_llm", False):
        raise PermissionError("hf_api sends the de-identified payload off-machine; set "
                              "research.reports.allow_external_llm: true to authorize it")
    from reporting.llm import build_llm

    reporting_cfg = {**config["reporting"], "backend": backend}
    return build_llm(reporting_cfg, device=config.get("device"), hf_token=hf_token)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_report(record: dict, prediction: dict, hits: list[dict], llm=None, max_retries: int = 2,
                    index_version: str | None = None, radiologist_assessment: dict | None = None) -> dict:
    """Facts + validated narrative (or an explicit abstention/failure). Never raises on LLM errors."""
    facts = build_facts(record, prediction, radiologist_assessment)
    report = {
        "report_version": PROMPT_VERSION,
        "disclaimer": DISCLAIMER,
        "record_id": record["record_id"],
        "status": "error",
        "observed_features": facts["observed_features"],
        "quantitative_radiomics_summary": facts["quantitative_radiomics_summary"],
        "classifier_output": facts["classifier_output"],
        "birads_assessment": facts["birads_assessment"],
        "summary": None,
        "supporting_evidence": [],
        "uncertainty_and_limitations": facts["known_limitations"],
        "recommended_review": None,
        "requires_radiologist_review": True,
        "retrieved_evidence": [{k: h[k] for k in ("chunk_id", "title", "citation", "evidence_tier", "page",
                                                  "section", "score", "text")} for h in hits],
        "provider": getattr(llm, "backend", None),
        "llm_model_id": getattr(llm, "model_id", None),
        "rag_index_version": index_version,
        "attempts": 0,
        "error": None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    reason = abstain_reason(record)
    if reason:
        report.update(status="abstained", summary=f"No report generated: {reason}",
                      recommended_review="A radiologist should review the image and outline the lesion manually.",
                      uncertainty_and_limitations=[reason, *facts["known_limitations"]])
        return report
    if llm is None:
        report.update(status="error", error="No LLM provider configured.")
        return report
    allowed = {h["chunk_id"] for h in hits}
    messages = build_messages(facts, hits)
    last_error = ""
    for attempt in range(1, max_retries + 2):
        report["attempts"] = attempt
        try:
            raw = llm.generate(messages)
        except Exception as exc:  # noqa: BLE001 - provider/network/OOM errors become a failed report
            report.update(status="error", error=f"{type(exc).__name__}: {exc}")
            return report
        try:
            narrative = validate_narrative(raw, facts, allowed)
        except NarrativeValidationError as exc:
            last_error = str(exc)
            messages = messages + [{"role": "assistant", "content": raw}, retry_message(last_error)]
            continue
        report.update(status="ok", summary=narrative.summary,
                      supporting_evidence=[e.model_dump() for e in narrative.supporting_evidence],
                      uncertainty_and_limitations=list(dict.fromkeys(
                          [*facts["known_limitations"], *narrative.uncertainty_and_limitations])),
                      recommended_review=narrative.recommended_review)
        return report
    report.update(status="invalid", error=f"narrative failed validation after {report['attempts']} attempt(s): "
                                          f"{last_error}")
    return report


REPORT_SCHEMA_KEYS = ("summary", "observed_features", "quantitative_radiomics_summary", "classifier_output",
                      "birads_assessment", "supporting_evidence", "uncertainty_and_limitations",
                      "recommended_review", "requires_radiologist_review")


def report_to_markdown(report: dict) -> str:
    """Readable rendering for downloads; every reference statement keeps its chunk-id citation."""
    clf = report["classifier_output"]
    lines = [f"# Quantitative ultrasound research report — {report['record_id']}", "",
             f"> {report['disclaimer']}", "", f"- **Status:** {report['status']}",
             f"- **Requires radiologist review:** {'yes' if report['requires_radiologist_review'] else 'no'}",
             f"- **BI-RADS:** {report['birads_assessment']['reason']}"]
    if clf.get("available"):
        lines.append(f"- **Classifier ({clf['prediction_type']}, {clf['model_id']}):** p(malignant) = "
                     f"{clf['p_malignant']} · calibration: {clf['calibration_status']}"
                     + (" · OUT OF DISTRIBUTION" if clf.get("out_of_distribution") else ""))
    else:
        lines.append(f"- **Classifier:** unavailable ({clf.get('reason')})")
    lines.append(f"- **Provider:** {report['provider']} / {report['llm_model_id']} · RAG index "
                 f"{report['rag_index_version']} · generated {report['generated_at_utc']}")
    lines += ["", "## Summary", "", report["summary"] or f"_{report.get('error') or 'not available'}_", "",
              "## BI-RADS descriptors", "", "| Descriptor | Status | Value | Source |", "|---|---|---|---|"]
    for d in report["observed_features"]:
        status = {"not_assessed": "Descriptor not assessed"}.get(d["status"], d["status"])
        lines.append(f"| {d['descriptor']} | {status} | {d['value'] or '—'} | {d['source']} |")
    lines += ["", "## Quantitative radiomics (computed)", "", "| Feature | Value | Unit |", "|---|---|---|"]
    lines += [f"| {q['feature']} | {q['value']} | {q['unit']} |" for q in report["quantitative_radiomics_summary"]]
    lines += ["", "## Supporting evidence (retrieved references)", ""]
    lines += [f"- {e['claim']} [{', '.join(e['citations'])}]" for e in report["supporting_evidence"]] or ["- none"]
    lines += ["", "## Uncertainty and limitations", ""]
    lines += [f"- {x}" for x in report["uncertainty_and_limitations"]]
    lines += ["", "## Recommended review", "", report["recommended_review"] or "—", "", "## Citations", ""]
    lines += [f"- `{h['chunk_id']}` — {h['citation']} ({h['evidence_tier']}"
              + (f", p{h['page']}" if h.get("page") else "") + (f", {h['section']}" if h.get("section") else "") + ")"
              for h in report["retrieved_evidence"]]
    return "\n".join(lines) + "\n"
