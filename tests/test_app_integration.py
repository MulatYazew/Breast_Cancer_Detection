"""The research panel inside the real app module, driven by Streamlit's AppTest (no browser, no checkpoints)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

PANEL_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
import numpy as np
import app
import bus_research

app.RESEARCH_OUTPUT_DIR = __import__("pathlib").Path({out!r})
if {mode!r} == "missing_deps":
    bus_research.missing_dependencies = lambda: ["pyradiomics"]
if {mode!r} == "no_artifacts":
    app.get_research_classifier = lambda: None
    app.get_reference_index = lambda: None
rng = np.random.RandomState(0)
image = rng.randint(90, 160, (300, 400, 3)).astype(np.uint8)
mask = np.zeros((256, 256), np.uint8)
mask[60:200, 50:210] = 1
state = {{"image": image, "mask": mask, "padded_bbox": (100, 60, 300, 220), "mask_source": "model",
          "bbox_source": "model", "detection_confidence": 0.9}}
app.render_research_section("upload.png", state)
"""


@pytest.mark.parametrize("mode", ["normal", "no_artifacts", "missing_deps"])
def test_research_panel_degrades_gracefully(mode, monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_ROOT)   # config paths are repo-relative, as when launched with `streamlit run app.py`
    at = AppTest.from_string(PANEL_SCRIPT.format(root=str(REPO_ROOT), mode=mode, out=str(tmp_path)), default_timeout=120).run()
    assert not at.exception, at.exception
    infos = " ".join(i.value for i in at.info)
    if mode == "missing_deps":
        assert "isn't installed" in infos and "pyradiomics" in infos
        return
    assert "BI-RADS assessment unavailable: validated BI-RADS labels/model not configured." in infos
    assert any("Research model output" in c.value for c in at.caption)
    if mode == "no_artifacts":
        assert "Radiomics classifier unavailable" in infos
        assert any("Reference index not built" in c.value for c in at.caption)
    assert [b.label for b in at.button] == ["Generate research report"]
    assert len(list((tmp_path / "records").glob("rec_*.json"))) == 1   # validated record saved


def test_stage6_export_hides_rule_proposal():
    import app

    result = {"category": "4B", "likelihood": ">10% to ≤50%", "recommendation": "biopsy", "patient_summary": "x",
              "rule_score": 2.0, "rule_points": [], "status": "ok"}
    hidden = app._gated_report(result, show_category=False)
    assert hidden["category"] is None and hidden["recommendation"] is None and hidden["patient_summary"] is None
    assert "unavailable" in hidden["note"]
    assert app._gated_report(result, show_category=True) is result
