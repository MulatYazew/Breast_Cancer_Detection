"""Inference-time runtime for the BI-RADS-aware research workflow (notebooks 01-06).

The research *workflows* (dataset audit, batch extraction, training, evaluation,
reference ingestion, report generation) live in ``notebooks/0[1-6]_*.ipynb``.
This package holds only the functions those notebooks *and* ``app.py`` both
need at inference time, so the app loads the notebooks' saved artifacts
instead of re-implementing them:

* ``birads``        assessment categories, US descriptor schema, label validation/gating
* ``radiomics_extract``  mask QC + PyRadiomics extraction on the native-resolution image
* ``feature_store`` versioned, schema-validated, atomically written lesion JSON records
* ``rag``           authorized-reference ingestion, BM25 (+ optional embedding) retrieval, citations
* ``classifier``    load/predict with the notebook-04 radiomics classifier artifact
* ``report``        evidence-grounded, schema-validated LLM report with abstention

Importing this package is cheap; PyRadiomics, SimpleITK, pypdf, joblib and the
LLM backends are imported lazily, so ``missing_dependencies()`` can be checked
first and the app keeps working without them.
"""

from __future__ import annotations

import importlib.util

DISCLAIMER = "Research model output — not a clinical diagnosis."

_REQUIRED = {"radiomics": "pyradiomics", "SimpleITK": "SimpleITK", "jsonschema": "jsonschema", "joblib": "joblib"}


def missing_dependencies() -> list[str]:
    """pip package names the research runtime needs that aren't importable."""
    missing = []
    for module, package in _REQUIRED.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(package)
    return missing
