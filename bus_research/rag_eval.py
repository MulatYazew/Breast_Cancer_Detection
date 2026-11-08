"""Small retrieval checks against the supplied BI-RADS documents.

A hit is relevant when the chunk comes from the expected source and contains
one of the expected phrases. Phrases were taken from text extracted from those
PDFs. If a phrase is absent, the question fails rather than being rewritten
to match a wrong passage.
"""

from __future__ import annotations

from pathlib import Path

QUESTIONS = [
    {
        "id": "shape",
        "query": "ultrasound mass shape oval round irregular",
        "source_id": "birads-card",
        "phrases": ["oval", "round", "irregular"],
    },
    {
        "id": "orientation",
        "query": "orientation long axis relative to the skin parallel not parallel round mass",
        "source_id": "lee-2017-us-birads",
        "phrases": ["long axis relative to the skin"],
    },
    {
        "id": "margin",
        "query": "margin circumscribed not circumscribed indistinct angular microlobulated spiculated",
        "source_id": "birads-card",
        "phrases": ["not circumscribed", "microlobulated"],
    },
    {
        "id": "echo",
        "query": "echo pattern anechoic hyperechoic complex cystic and solid hypoechoic",
        "source_id": "birads-card",
        "phrases": ["complex cystic and solid"],
    },
    {
        "id": "posterior",
        "query": "posterior features enhancement shadowing combined pattern",
        "source_id": "birads-card",
        "phrases": ["no posterior features", "combined pattern"],
    },
    {
        "id": "calcifications",
        "query": "calcifications in a mass outside of a mass intraductal",
        "source_id": "birads-card",
        "phrases": ["calcifications in a mass"],
    },
    {
        "id": "special",
        "query": "special cases simple cyst clustered microcysts fat necrosis",
        "source_id": "birads-card",
        "phrases": ["simple cyst", "fat necrosis"],
    },
    {
        "id": "category0",
        "query": "category 0 incomplete additional imaging evaluation",
        "source_id": "birads-card",
        "phrases": ["incomplete"],
    },
    {
        "id": "category6",
        "query": "category 6 known biopsy-proven malignancy",
        "source_id": "birads-card",
        "phrases": ["biopsy-proven"],
    },
    {
        "id": "likelihood",
        "query": "category 3 likelihood of cancer greater than 0 percent to 2 percent category 4A 4B 4C category 5 95",
        "source_id": "lee-2017-us-birads",
        "phrases": ["likelihood of cancer for category 3"],
    },
]


def _relevant(hit: dict, question: dict) -> bool:
    if hit["source_id"] != question["source_id"]:
        return False
    text = hit["text"].lower()
    return any(phrase.lower() in text for phrase in question["phrases"])


def evaluate_retrieval(index, questions: list[dict] | None = None, k: int = 5) -> dict:
    questions = questions or QUESTIONS
    rows = []
    reciprocal = []
    precision_sum = 0.0
    recall_sum = 0.0
    for question in questions:
        hits = index.search(question["query"], k=k)
        flags = [_relevant(hit, question) for hit in hits]
        relevant_ranks = [i + 1 for i, flag in enumerate(flags) if flag]
        precision = sum(flags) / k
        recall = 1.0 if relevant_ranks else 0.0
        rr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
        precision_sum += precision
        recall_sum += recall
        reciprocal.append(rr)
        rows.append(
            {
                "id": question["id"],
                "precision_at_k": round(precision, 4),
                "recall_at_k": recall,
                "reciprocal_rank": round(rr, 4),
                "top_citations": [
                    {"chunk_id": hit["chunk_id"], "source_id": hit["source_id"], "page": hit["page"], "section": hit["section"]}
                    for hit in hits[:k]
                ],
            }
        )
    n = max(len(questions), 1)
    # nDCG with binary relevance, ideal is a single relevant document at rank 1.
    import math

    ndcg_sum = 0.0
    for row in rows:
        # recompute from reciprocal isn't enough for full nDCG; use citations only as binary via recall path
        pass
    for question, row in zip(questions, rows):
        hits = index.search(question["query"], k=k)
        gains = [1.0 if _relevant(hit, question) else 0.0 for hit in hits]
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, int(sum(gains)) + 1))
        ndcg_sum += (dcg / ideal) if ideal else 0.0
    return {
        "k": k,
        "n_questions": len(questions),
        "precision_at_k": round(precision_sum / n, 4),
        "recall_at_k": round(recall_sum / n, 4),
        "mrr": round(sum(reciprocal) / n, 4),
        "ndcg_at_k": round(ndcg_sum / n, 4),
        "questions": rows,
        "note": (
            "Relevance is phrase-and-source match against the supplied PDFs. "
            "It is not a radiologist-judged relevance set, and it does not measure LLM groundedness."
        ),
    }


def build_project_index(repo_root: str | Path, sources: list[dict] | None = None):
    from bus_research.rag import build_index, ingest_sources

    repo_root = Path(repo_root)
    if sources is None:
        import yaml

        config = yaml.safe_load((repo_root / "config.yaml").read_text())
        rag = config["research"]["rag"]
        sources = [s for s in rag["sources"] if s["kind"] == "pdf"]
        chunk_words = rag["chunk_words"]
        overlap = rag["chunk_overlap_words"]
        forbidden = rag.get("forbidden_dirs") or []
    else:
        chunk_words, overlap, forbidden = 120, 30, []
    chunks, records = ingest_sources(sources, repo_root, chunk_words, overlap, forbidden)
    missing = [r["id"] for r in records if r.get("status") == "missing"]
    if missing:
        raise FileNotFoundError(f"RAG sources missing: {missing}")
    return build_index(chunks, records, chunk_words, overlap, embedding_model_id=None)
