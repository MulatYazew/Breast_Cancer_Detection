"""Stage 6d: retrieval: BI-RADS knowledge chunks and similar training cases.

* ``KnowledgeIndex``: the markdown knowledge base under ``reporting/knowledge``
  split into ``## [id] Title`` chunks, embedded with a sentence-transformer and
  searched with FAISS (inner product on L2-normalized vectors = cosine). Small
  enough to build in memory on first use, so no prebuilt artifact is needed.
* ``CaseIndex``: ResNet50 penultimate-layer embeddings of every *training*
  lesion's masked ROI (ground-truth mask), with its true label and a
  thumbnail. Built offline by ``python -m reporting.build_index``; optional:
  if it isn't there, reports are generated without similar cases.

faiss / sentence-transformers are imported lazily (they're optional deps).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

CHUNK_HEADING = re.compile(r"^## \[(?P<id>[a-z0-9-]+)\] (?P<title>.+)$", re.MULTILINE)


@dataclass
class KnowledgeChunk:
    id: str
    title: str
    text: str
    source_file: str


def load_knowledge(knowledge_dir: str | Path) -> tuple[list[KnowledgeChunk], str]:
    """Parse every ``*.md`` file into chunks; return (chunks, kb_version hash)."""
    chunks: list[KnowledgeChunk] = []
    digest = hashlib.sha1()
    for path in sorted(Path(knowledge_dir).glob("*.md")):
        content = path.read_text()
        digest.update(path.name.encode() + content.encode())
        matches = list(CHUNK_HEADING.finditer(content))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = " ".join(content[match.end() : end].split())
            chunks.append(KnowledgeChunk(match["id"], match["title"].strip(), body, path.name))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate knowledge chunk ids in {knowledge_dir}")
    return chunks, f"kb-{digest.hexdigest()[:10]}"


class KnowledgeIndex:
    def __init__(self, chunks: list[KnowledgeChunk], embed_texts: Callable[[list[str]], np.ndarray]) -> None:
        import faiss

        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self._embed = embed_texts
        vectors = embed_texts([f"{c.title}. {c.text}" for c in chunks]).astype(np.float32)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int) -> list[tuple[KnowledgeChunk, float]]:
        vector = self._embed([query]).astype(np.float32)
        scores, idx = self.index.search(vector, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    def retrieve(self, query: str, k: int, pinned_ids: list[str]) -> list[dict]:
        """Pinned chunks (the computed descriptors' and category's definitions) + top-k semantic hits."""
        hits: dict[str, dict] = {}
        for chunk_id in pinned_ids:
            if chunk_id in self.by_id and chunk_id not in hits:
                c = self.by_id[chunk_id]
                hits[chunk_id] = {"id": c.id, "title": c.title, "text": c.text, "score": None, "pinned": True}
        for chunk, score in self.search(query, k):
            if chunk.id not in hits:
                hits[chunk.id] = {"id": chunk.id, "title": chunk.title, "text": chunk.text,
                                  "score": round(score, 4), "pinned": False}
        return list(hits.values())


def sentence_embedder(model_id: str) -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device="cpu")

    def embed(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))

    return embed


class CaseIndex:
    """FAISS index over training-lesion ROI embeddings (see ``reporting.build_index``)."""

    def __init__(self, index_dir: str | Path) -> None:
        import faiss

        self.index_dir = Path(index_dir)
        self.meta: list[dict] = json.loads((self.index_dir / "cases.json").read_text())
        vectors = np.load(self.index_dir / "case_embeddings.npy").astype(np.float32)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    @classmethod
    def load_if_present(cls, index_dir: str | Path) -> "CaseIndex | None":
        path = Path(index_dir)
        if (path / "cases.json").exists() and (path / "case_embeddings.npy").exists():
            return cls(path)
        return None

    def search(self, embedding: np.ndarray, k: int) -> list[dict]:
        vector = embedding.reshape(1, -1).astype(np.float32)
        scores, idx = self.index.search(vector, min(k, len(self.meta)))
        results = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            meta = self.meta[i]
            results.append({
                "id": meta["id"],
                "label": meta["label"],
                "similarity": round(float(s), 4),
                "thumbnail": str(self.index_dir / meta["thumbnail"]),
            })
        return results
