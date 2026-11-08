"""Retrieval against the two supplied PDFs. Skips if a file is absent."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "dataset" / "BIRADS Reference Card 1.pdf").exists()
    or not (ROOT / "dataset" / "usg-16034 (1).pdf").exists(),
    reason="BI-RADS reference PDFs are not in this checkout",
)
def test_supplied_pdfs_are_retrievable():
    from bus_research.rag_eval import QUESTIONS, build_project_index, evaluate_retrieval

    index = build_project_index(ROOT)
    metrics = evaluate_retrieval(index, QUESTIONS, k=5)
    by_id = {row["id"]: row for row in metrics["questions"]}
    assert by_id["orientation"]["recall_at_k"] == 1.0
    assert by_id["category6"]["recall_at_k"] == 1.0
    assert by_id["likelihood"]["recall_at_k"] == 1.0
    assert metrics["recall_at_k"] >= 0.8
    assert metrics["mrr"] > 0.5
    cited = by_id["orientation"]["top_citations"]
    assert any(hit["source_id"] == "lee-2017-us-birads" and hit["page"] is not None for hit in cited)
