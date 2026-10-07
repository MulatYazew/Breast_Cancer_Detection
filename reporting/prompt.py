"""Stage 6f: prompt construction. Bump ``PROMPT_VERSION`` whenever the wording changes
(it is written to every report and audit-log entry)."""

from __future__ import annotations

import json

from reporting.lexicon import Descriptor
from reporting.scoring import RuleAssessment

PROMPT_VERSION = "stage6-prompt-v1"

SYSTEM_PROMPT = """You draft structured breast ultrasound reports for a radiologist to review.
You explain an assessment that has ALREADY been computed; you never change it.

Rules:
- Output exactly one JSON object and nothing else.
- "category" must be exactly the proposed BI-RADS category you are given.
- Use only the findings listed. Do not invent findings, measurements, history, or Doppler/elastography results.
- Every rationale item cites one source id from the provided list: "feature:<key>" for a computed descriptor
  (its "value" must be copied exactly) or "kb:<id>" for a knowledge chunk.
- "supports" is "benign", "suspicious", or "neutral".
- "patient_summary" is written TO the patient ("you"), 3-6 short sentences at a 6th-grade reading level:
  what the scan looked at, what was seen in everyday words (e.g. "a small area with smooth, even edges"),
  and that your doctor will review the images with you and explain the next step (say what that step is,
  e.g. "a follow-up scan in a few months" or "taking a small tissue sample, called a biopsy").
  No medical jargon, no category numbers, no percentages, and never state or imply a diagnosis
  (do not name a condition such as "fibroadenoma" or "carcinoma").

JSON shape:
{"category": "...", "rationale": [{"descriptor": "...", "value": "...", "supports": "...", "source": "..."}],
 "recommendation": "...", "patient_summary": "..."}"""


def build_messages(
    descriptors: list[Descriptor],
    rules: RuleAssessment,
    knowledge_hits: list[dict],
    similar_cases: list[dict],
) -> list[dict]:
    findings = [
        {"source": f"feature:{d.key}", "descriptor": d.category, "value": d.value,
         "supports": d.polarity, "assessable": d.assessable, "measurements": d.evidence, "note": d.note}
        for d in descriptors
    ]
    knowledge = [{"source": f"kb:{h['id']}", "title": h["title"], "text": h["text"]} for h in knowledge_hits]
    if similar_cases:
        n_mal = sum(1 for c in similar_cases if c["label"] == "malignant")
        similar = (f"{n_mal} of the {len(similar_cases)} most similar training lesions were malignant "
                   f"(ids: {', '.join(c['id'] for c in similar_cases)}).")
    else:
        similar = "No similar-case index available."

    user = f"""Proposed BI-RADS category (fixed): {rules.category}
Likelihood of malignancy for this category: {rules.likelihood}
Rule score: {rules.score} (points: {json.dumps(rules.points)})
Image classifier probability of malignancy: {rules.malignant_probability if rules.malignant_probability is not None else 'n/a'}
Default recommendation for this category: {rules.recommendation}
Similar cases: {similar}

Computed findings:
{json.dumps(findings, indent=1)}

Knowledge chunks:
{json.dumps(knowledge, indent=1)}

Write the JSON report now. Include rationale items for at least shape, orientation, margin, echo pattern and
posterior features, citing their feature sources."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def retry_message(error: str) -> dict:
    return {
        "role": "user",
        "content": f"Your JSON failed validation: {error}\nReturn the corrected JSON object only.",
    }
