"""Stage 6: structured BI-RADS report drafting (features -> lexicon -> rules -> RAG -> LLM).

Importing this package is cheap and dependency-free: the optional reporting
dependencies (requirements-reporting.txt) are only imported inside
``reporting.service.build_report_service`` and the modules it pulls in, so the
app can check ``missing_dependencies`` first and keep running without them.
"""

from __future__ import annotations

import hashlib
import importlib.util

import numpy as np

_COMMON = {"faiss": "faiss-cpu", "sentence_transformers": "sentence-transformers", "pydantic": "pydantic"}
_BY_BACKEND = {
    "local": {"transformers": "transformers", "accelerate": "accelerate"},
    "hf_api": {"huggingface_hub": "huggingface_hub", "httpx": "httpx"},
}


def missing_dependencies(backend: str) -> list[str]:
    """pip package names required by ``backend`` that aren't importable."""
    required = {**_COMMON, **_BY_BACKEND.get(backend, {})}
    missing = []
    for module, package in required.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):  # e.g. sys.modules[module] = None
            found = False
        if not found:
            missing.append(package)
    return missing


def report_key(roi_mask: np.ndarray, padded_bbox) -> str:
    """Identity of the clinician-approved inputs: any change to the mask or box changes the key."""
    digest = hashlib.sha1()
    mask = (np.asarray(roi_mask) > 0).astype(np.uint8)
    digest.update(str(mask.shape).encode())
    digest.update(np.packbits(mask).tobytes())
    digest.update(str([int(v) for v in padded_bbox]).encode())
    return digest.hexdigest()[:16]
