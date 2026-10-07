"""Readable Markdown rendering of a reviewed Stage 6 report (for the case bundle)."""

from __future__ import annotations

DISCLAIMER = "AI-generated draft for clinician review. Not a diagnosis."


def report_to_markdown(report: dict, review: dict | None, stale: bool) -> str:
    """``report`` is ``ReportResult.to_dict()``; ``review`` is the clinician decision (or None)."""
    status = "stale" if stale else (review["status"] if review else "pending_review")
    final_category = (review or {}).get("final_category")
    lines = [
        f"# Structured breast ultrasound report: {report['case_id']}",
        "",
        f"> {DISCLAIMER}",
        "",
        f"- **Review status:** {status}",
        f"- **AI-proposed BI-RADS category:** {report.get('category') or 'n/a'} "
        f"(likelihood of malignancy {report.get('likelihood') or 'n/a'})",
    ]
    if final_category:
        lines.append(f"- **Clinician final category:** {final_category}")
    lines += [
        f"- **Confidence:** {report.get('confidence') or 'n/a'}"
        + (f" ({'; '.join(report['confidence_reasons'])})" if report.get("confidence_reasons") else ""),
        f"- **Mask source:** {report.get('mask_source')} · **Box source:** {report.get('bbox_source')}",
        f"- **Model:** {report.get('model_id')} ({report.get('backend')}) · prompt {report.get('prompt_version')}"
        f" · knowledge base {report.get('kb_version')}",
        f"- **Generated:** {report.get('generated_at')}",
        "",
        "## BI-RADS lexicon descriptors",
        "",
        "| Category | Value | Assessable | Supports | Measurements |",
        "|---|---|---|---|---|",
    ]
    for d in report.get("descriptors", []):
        evidence = ", ".join(f"{k}={v}" for k, v in (d.get("evidence") or {}).items())
        lines.append(f"| {d['category']} | {d['value']} | {'yes' if d['assessable'] else 'no'} "
                     f"| {d['polarity']} | {evidence} |")

    if report.get("status") == "ok":
        lines += ["", "## Rationale", "", "| Descriptor | Value | Supports | Source |", "|---|---|---|---|"]
        lines += [f"| {r['descriptor']} | {r['value']} | {r['supports']} | {r['source']} |"
                  for r in report.get("rationale", [])]
        recommendation = (review or {}).get("recommendation") or report.get("recommendation")
        lines += ["", "## Recommendation", "", recommendation or ""]
        if (review or {}).get("notes"):
            lines += ["", "## Clinician notes", "", review["notes"]]
        lines += ["", "## Limitations", ""] + [f"- {item}" for item in report.get("limitations", [])]
        if review and review["status"] in ("accepted", "edited"):
            lines += ["", "## Patient summary", "", report.get("patient_summary") or ""]
    else:
        lines += ["", "## Draft not available", "", report.get("error") or ""]

    lines += ["", "## Evidence", ""]
    lines += [f"- `kb:{h['id']}`: {h['title']}" for h in report.get("knowledge_hits", [])]
    lines += [f"- similar case `{c['id']}`: {c['label']} (similarity {c['similarity']:.2f})"
              for c in report.get("similar_cases", [])]
    return "\n".join(lines) + "\n"
