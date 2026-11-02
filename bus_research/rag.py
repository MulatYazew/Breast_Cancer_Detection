"""Authorized-reference ingestion and retrieval with citations.

Sources are declared in ``config.yaml`` (``research.rag.sources``), each with a
title, citation, version and an ``evidence_tier``:

* ``primary_reference``        e.g. the ACR BI-RADS quick-reference card shipped with the dataset
* ``official_documentation``   e.g. PyRadiomics feature definitions (docstrings of the installed version)
* ``secondary_project_notes``  project-authored paraphrases; retrievable, but never primary evidence

Ingestion refuses anything under ``research.rag.forbidden_dirs`` (feature
records, splits, labels), so retrieval can never surface a case or its label.

Retrieval is deterministic BM25 by default; when ``embedding_model_id`` is set
and sentence-transformers is installed, BM25 and cosine scores are blended.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

INDEX_FORMAT = "bus-rag-1"
TOKEN = re.compile(r"[a-z0-9]+")
EVIDENCE_TIERS = ("primary_reference", "official_documentation", "secondary_project_notes")


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    citation: str
    version: str | None
    url: str | None
    evidence_tier: str
    page: int | None
    section: str | None
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words] if words else []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Readers: (page, section, text) units per source kind
# --------------------------------------------------------------------------- #
def _read_pdf(path: Path):
    from pypdf import PdfReader

    for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
        yield page_no, None, page.extract_text() or ""


def _read_markdown(path: Path):
    text = path.read_text()
    parts = re.split(r"^(#{1,3} .+)$", text, flags=re.MULTILINE)
    section = None
    for part in parts:
        if re.match(r"^#{1,3} ", part):
            section = part.lstrip("#").strip()
        elif part.strip():
            yield None, section, part


def _read_text(path: Path):
    yield None, None, path.read_text()


def _read_python_docstrings(modules: list[str]):
    """Feature definitions from the *installed* package's docstrings (what its official docs render)."""
    for module_name in modules:
        module = importlib.import_module(module_name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module_name:
                continue
            doc = inspect.getdoc(cls)
            if doc:
                yield None, cls.__name__, doc
            for name, fn in inspect.getmembers(cls, inspect.isfunction):
                if name.startswith("get") and name.endswith("FeatureValue") and inspect.getdoc(fn):
                    yield None, f"{cls.__name__}.{name[3:-12]}", inspect.getdoc(fn)


def _clean(text: str) -> str:
    text = re.sub(r":math:`([^`]*)`", r"\1", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def ingest_sources(sources: list[dict], repo_root: str | Path, chunk_words: int = 120, overlap_words: int = 30,
                   forbidden_dirs: list[str] | None = None) -> tuple[list[Chunk], list[dict]]:
    """Read every declared source into overlapping word-window chunks; returns (chunks, source records)."""
    repo_root = Path(repo_root).resolve()
    forbidden = [(repo_root / d).resolve() for d in (forbidden_dirs or [])]
    chunks: list[Chunk] = []
    source_records: list[dict] = []
    ids = [s["id"] for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source ids in research.rag.sources")
    for src in sources:
        if src.get("evidence_tier") not in EVIDENCE_TIERS:
            raise ValueError(f"source {src['id']}: evidence_tier must be one of {EVIDENCE_TIERS}")
        kind = src["kind"]
        record = {k: src.get(k) for k in ("id", "title", "citation", "version", "url", "evidence_tier", "kind",
                                          "license_note")}
        if kind == "python_docstrings":
            module = importlib.import_module(src["modules"][0].split(".")[0])
            record["version"] = src.get("version") or getattr(module, "__version__", None)
            record["content_sha256"] = None
            units = _read_python_docstrings(src["modules"])
        else:
            path = (repo_root / src["path"]).resolve()
            if any(path.is_relative_to(f) for f in forbidden):
                raise PermissionError(f"source {src['id']} is under a forbidden directory (case data/labels)")
            if not path.exists():
                record["status"] = "missing"
                source_records.append(record)
                continue
            record["path"] = src["path"]
            record["content_sha256"] = _sha256(path)
            units = {"pdf": _read_pdf, "markdown": _read_markdown, "text": _read_text}[kind](path)
        n_before = len(chunks)
        for page, section, text in units:
            words = _clean(text).split()
            for i, window in enumerate(_windows(words, chunk_words, overlap_words)):
                loc = f"p{page}" if page is not None else (
                    re.sub(r"[^a-z0-9]+", "-", (section or "").lower()).strip("-")[:40].strip("-") or "doc")
                chunks.append(Chunk(
                    chunk_id=f"{src['id']}:{loc}:{i}",
                    source_id=src["id"], title=src["title"], citation=src["citation"],
                    version=record["version"], url=src.get("url"), evidence_tier=src["evidence_tier"],
                    page=page, section=section, text=" ".join(window),
                ))
        record["status"] = "ingested"
        record["n_chunks"] = len(chunks) - n_before
        source_records.append(record)
    seen: Counter = Counter()
    for c in chunks:  # disambiguate repeated section slugs
        seen[c.chunk_id] += 1
        if seen[c.chunk_id] > 1:
            c.chunk_id = f"{c.chunk_id}.{seen[c.chunk_id] - 1}"
    return chunks, source_records


def assert_no_case_data(chunks: list[Chunk], forbidden_terms: list[str]) -> None:
    """Fail if any chunk mentions a case/record id (label-leakage guard for retrieval)."""
    terms = [t.lower() for t in forbidden_terms if t]
    for c in chunks:
        low = c.text.lower()
        for t in terms:
            if t in low:
                raise AssertionError(f"chunk {c.chunk_id} contains case identifier {t!r}")


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
class ReferenceIndex:
    def __init__(self, chunks: list[Chunk], manifest: dict, embeddings: np.ndarray | None = None,
                 embed_fn=None, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.manifest = manifest
        self.embeddings = embeddings
        self.embed_fn = embed_fn
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(f"{c.section or ''} {c.text}")) for c in chunks]
        self._len = np.array([sum(tf.values()) for tf in self._tf], dtype=np.float64)
        self._avg = float(self._len.mean()) if len(chunks) else 1.0
        df = Counter(t for tf in self._tf for t in tf)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @property
    def version(self) -> str:
        return self.manifest["index_version"]

    def bm25(self, query: str) -> np.ndarray:
        q = tokenize(query)
        scores = np.zeros(len(self.chunks))
        for i, tf in enumerate(self._tf):
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
            scores[i] = sum(self._idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return scores

    def search(self, query: str, k: int = 5, tiers: tuple[str, ...] | None = None) -> list[dict]:
        """Top-k chunks with scores and full citation metadata (deterministic tie-break by chunk id)."""
        lexical = self.bm25(query)
        score = lexical / lexical.max() if lexical.max() > 0 else lexical
        method = "bm25"
        if self.embeddings is not None and self.embed_fn is not None:
            q = np.asarray(self.embed_fn([query]))[0]
            score = 0.5 * score + 0.5 * (self.embeddings @ q)
            method = "bm25+embedding"
        order = sorted(range(len(self.chunks)), key=lambda i: (-score[i], self.chunks[i].chunk_id))
        hits = []
        for i in order:
            c = self.chunks[i]
            if tiers and c.evidence_tier not in tiers:
                continue
            if lexical[i] <= 0 and method == "bm25":
                break
            hits.append({**c.to_dict(), "score": round(float(score[i]), 4), "retrieval": method})
            if len(hits) >= k:
                break
        return hits

    def save(self, index_dir: str | Path) -> Path:
        from bus_research.feature_store import atomic_write_json

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.jsonl").write_text("\n".join(json.dumps(c.to_dict()) for c in self.chunks) + "\n")
        if self.embeddings is not None:
            np.save(index_dir / "embeddings.npy", self.embeddings.astype(np.float32))
        atomic_write_json(index_dir / "index_manifest.json", self.manifest)
        return index_dir

    @classmethod
    def load(cls, index_dir: str | Path, embed_fn=None) -> "ReferenceIndex":
        index_dir = Path(index_dir)
        manifest = json.loads((index_dir / "index_manifest.json").read_text())
        if manifest.get("format") != INDEX_FORMAT:
            raise ValueError(f"unsupported index format {manifest.get('format')!r}")
        chunks = [Chunk(**json.loads(line)) for line in (index_dir / "chunks.jsonl").read_text().splitlines() if line]
        emb_path = index_dir / "embeddings.npy"
        embeddings = np.load(emb_path) if emb_path.exists() and embed_fn is not None else None
        return cls(chunks, manifest, embeddings, embed_fn)


def build_index(chunks: list[Chunk], source_records: list[dict], chunk_words: int, overlap_words: int,
                embedding_model_id: str | None = None) -> ReferenceIndex:
    embed_fn, embeddings = None, None
    if embedding_model_id:
        from reporting.retrieval import sentence_embedder

        embed_fn = sentence_embedder(embedding_model_id)
        embeddings = np.asarray(embed_fn([f"{c.section or ''}. {c.text}" for c in chunks]), dtype=np.float32)
    digest = hashlib.sha256()
    for c in chunks:
        digest.update(c.chunk_id.encode() + c.text.encode())
    manifest = {
        "format": INDEX_FORMAT,
        "index_version": f"rag-{digest.hexdigest()[:12]}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_chunks": len(chunks),
        "chunking": {"unit": "words", "chunk_words": chunk_words, "overlap_words": overlap_words},
        "retrieval": "bm25" + ("+embedding" if embedding_model_id else ""),
        "embedding_model_id": embedding_model_id,
        "sources": source_records,
    }
    return ReferenceIndex(chunks, manifest, embeddings, embed_fn)


def build_query(record: dict, prediction: dict | None = None) -> str:
    """Retrieval query from what is actually known about the lesion (no labels, no ids)."""
    terms = ["breast ultrasound mass assessment category"]
    for key, d in (record.get("birads_descriptors") or {}).items():
        if d.get("status") == "observed":
            terms.append(f"{key.replace('_', ' ')} {d['value']}")
    feats = (record.get("radiomics") or {}).get("features") or {}
    if feats.get("shape2D"):
        terms.append("shape 2D sphericity elongation perimeter")
    if feats.get("glcm"):
        terms.append("gray level co-occurrence texture contrast")
    if feats.get("firstorder"):
        terms.append("first order intensity entropy")
    if prediction and prediction.get("class_probabilities"):
        terms.append("malignancy suspicious benign")
    return " ".join(terms)
