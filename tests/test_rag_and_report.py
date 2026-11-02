from __future__ import annotations

import json

import pytest

from bus_research import rag
from bus_research import report as rp

SOURCES = [
    {"id": "lex", "kind": "markdown", "path": "refs/lexicon.md", "title": "Lexicon", "citation": "Test lexicon",
     "version": "1", "evidence_tier": "primary_reference"},
    {"id": "notes", "kind": "text", "path": "refs/notes.txt", "title": "Notes", "citation": "Test notes",
     "version": None, "evidence_tier": "secondary_project_notes"},
]


@pytest.fixture
def ref_root(tmp_path):
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "lexicon.md").write_text(
        "# Lexicon\n\n## Margin\nA spiculated margin is formed by sharp lines radiating from the mass.\n\n"
        "## Orientation\nParallel orientation means the long axis is parallel to the skin.\n")
    (tmp_path / "refs" / "notes.txt").write_text(" ".join(f"word{i}" for i in range(300)))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "leak.txt").write_text("rec_abc malignant")
    return tmp_path


def test_ingestion_preserves_provenance_and_overlap(ref_root):
    chunks, sources = rag.ingest_sources(SOURCES, ref_root, chunk_words=100, overlap_words=20)
    margin = next(c for c in chunks if c.section == "Margin")
    assert margin.citation == "Test lexicon" and margin.evidence_tier == "primary_reference"
    assert margin.chunk_id == "lex:margin:0"
    notes = [c for c in chunks if c.source_id == "notes"]
    assert len(notes) == 4 and notes[1].text.split()[0] == "word80"            # 100-word windows, 20 overlap
    assert all(s["status"] == "ingested" and s["content_sha256"] for s in sources)


def test_forbidden_dirs_are_refused(ref_root):
    leak = [{"id": "leak", "kind": "text", "path": "results/leak.txt", "title": "x", "citation": "x",
             "evidence_tier": "secondary_project_notes"}]
    with pytest.raises(PermissionError):
        rag.ingest_sources(leak, ref_root, forbidden_dirs=["results"])


def test_case_data_guard(ref_root):
    chunks, _ = rag.ingest_sources(SOURCES, ref_root)
    rag.assert_no_case_data(chunks, ["rec_abc", "benign (12)"])
    with pytest.raises(AssertionError):
        rag.assert_no_case_data(chunks, ["spiculated"])


def test_retrieval_is_deterministic_and_round_trips(ref_root, tmp_path):
    chunks, sources = rag.ingest_sources(SOURCES, ref_root)
    index = rag.build_index(chunks, sources, 120, 30)
    hits = index.search("spiculated margin", k=3)
    assert hits[0]["chunk_id"] == "lex:margin:0" and hits[0]["citation"] == "Test lexicon"
    index.save(tmp_path / "idx")
    reloaded = rag.ReferenceIndex.load(tmp_path / "idx")
    assert reloaded.search("spiculated margin", k=3) == hits
    assert reloaded.version == index.version
    assert index.search("spiculated margin", k=3, tiers=("secondary_project_notes",)) == []


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
HITS = [{"chunk_id": "lex:margin:0", "title": "Lexicon", "citation": "Test lexicon", "evidence_tier":
         "primary_reference", "page": None, "section": "Margin", "score": 1.0,
         "text": "A spiculated margin is formed by sharp lines radiating from the mass."}]
PRED = {"available": True, "model_id": "m1", "prediction_type": "benign_vs_malignant",
        "class_probabilities": {"benign": 0.266, "malignant": 0.734}, "calibration_status": "calibrated",
        "ood": {"out_of_distribution": False}}


@pytest.fixture
def record(make_record):
    pytest.importorskip("radiomics")
    return make_record()


def _narrative(**overrides):
    base = {"summary": "Quantitative analysis of a predicted mask. The research model gives p(malignant) 0.734 "
                       "(calibrated). Descriptors were not assessed.",
            "supporting_evidence": [{"claim": "The lexicon defines spiculated margins.",
                                     "citations": ["lex:margin:0"]}],
            "uncertainty_and_limitations": ["Single still image."],
            "recommended_review": "A radiologist should review the outline and assess descriptors."}
    return json.dumps({**base, **overrides})


class FixedLLM:
    backend, model_id = "fixed", "fixed"

    def __init__(self, *responses):
        self.responses = list(responses)

    def generate(self, messages):
        return self.responses.pop(0)


def test_stub_report_is_schema_complete(record):
    out = rp.generate_report(record, PRED, HITS, rp.StubLLM())
    assert out["status"] == "ok", out["error"]
    assert all(k in out for k in rp.REPORT_SCHEMA_KEYS)
    assert out["requires_radiologist_review"] is True
    assert out["birads_assessment"]["status"] == "unavailable"
    assert out["classifier_output"]["p_malignant"] == 0.734
    assert all(d["status"] != "observed" for d in out["observed_features"])
    assert out["supporting_evidence"][0]["citations"] == ["lex:margin:0"]


@pytest.mark.parametrize("bad,reason", [
    (_narrative(summary="Findings are consistent with BI-RADS 4B given p(malignant) 0.734 here."), "BI-RADS"),
    (_narrative(recommended_review="Ultrasound-guided biopsy is recommended for this lesion now."), "directive"),
    (_narrative(supporting_evidence=[{"claim": "Lexicon definition here.", "citations": ["made:up:1"]}]),
     "unknown ids"),
    (_narrative(summary="The lesion measures 12.5 px across and the model gives p(malignant) 0.734 now."),
     "not in the facts"),
    (_narrative(summary="The mass has a spiculated margin and the model gives p(malignant) 0.734 now."),
     "not observed"),
    (_narrative(supporting_evidence=[{"claim": "The model gives p(malignant) 0.734 for this lesion.",
                                      "citations": ["lex:margin:0"]}]), "restates a computed value"),
    (_narrative(recommended_review="Given the model output, further evaluation may be warranted here."),
     "directive"),
])
def test_validator_rejects_unsupported_claims(record, bad, reason):
    out = rp.generate_report(record, PRED, HITS, FixedLLM(bad, bad, bad), max_retries=2)
    assert out["status"] == "invalid" and reason in out["error"]
    assert out["summary"] is None


def test_retry_recovers_after_invalid_draft(record):
    out = rp.generate_report(record, PRED, HITS, FixedLLM("not json", _narrative()), max_retries=1)
    assert out["status"] == "ok" and out["attempts"] == 2
    assert out["summary"].startswith("Quantitative analysis")


def test_percent_form_of_probability_is_accepted(record):
    ok = _narrative(summary="Quantitative analysis only; the research model output is 73.4% malignant (calibrated).")
    assert rp.generate_report(record, PRED, HITS, FixedLLM(ok))["status"] == "ok"


def test_abstains_without_calling_llm_on_failed_extraction(make_record, synthetic_case):
    pytest.importorskip("radiomics")
    import numpy as np

    failed = make_record(mask=np.zeros_like(synthetic_case[1]))

    class Boom:
        backend, model_id = "boom", "boom"

        def generate(self, messages):
            raise AssertionError("LLM must not be called")

    out = rp.generate_report(failed, PRED, HITS, Boom())
    assert out["status"] == "abstained" and "extraction failed" in out["summary"]


def test_provider_errors_and_missing_prediction(record):
    class Down:
        backend, model_id = "down", "down"

        def generate(self, messages):
            raise TimeoutError("provider timed out")

    assert rp.generate_report(record, PRED, HITS, Down())["status"] == "error"
    unavailable = {"available": False, "reason": "no model"}
    out = rp.generate_report(record, unavailable, HITS, rp.StubLLM())
    assert out["status"] == "ok" and out["classifier_output"] == {"available": False, "reason": "no model"}
    assert "No compatible" in out["summary"]


def test_external_provider_needs_explicit_authorization():
    cfg = {"research": {"reports": {"llm_backend": "hf_api", "allow_external_llm": False}}, "reporting": {}}
    with pytest.raises(PermissionError):
        rp.build_provider(cfg)
    assert isinstance(rp.build_provider({"research": {"reports": {"llm_backend": "stub"}}}), rp.StubLLM)
