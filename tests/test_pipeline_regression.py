"""Regression: the original YOLO26n -> U-Net -> ResNet50 cascade still produces the outputs recorded
in tests/fixtures/pipeline_regression.json (captured before the research workflow was added)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((REPO_ROOT / "tests/fixtures/pipeline_regression.json").read_text())


@pytest.fixture(scope="module")
def pipeline():
    torch = pytest.importorskip("torch")
    from utils.config import load_config

    config = load_config()
    inf = config["inference"]
    for key in ("yolo_weights", "unet_weights", "resnet_weights"):
        if not (REPO_ROOT / inf[key]).exists():
            pytest.skip(f"checkpoint {inf[key]} not available")
        inf[key] = str(REPO_ROOT / inf[key])
    from inference.pipeline import CADPipeline

    return CADPipeline(config, device=torch.device("cpu"))


@pytest.mark.parametrize("name", sorted(FIXTURE))
def test_cascade_outputs_unchanged(pipeline, name):
    folder = "malignant" if name.startswith("malignant") else "benign"
    path = REPO_ROOT / "dataset/BUSI_Jpeg" / folder / name
    if not path.exists():
        pytest.skip("BUSI images not available")
    expected = FIXTURE[name]
    result = pipeline.run(path)
    assert list(result.bbox) == expected["bbox"]
    assert result.predicted_class == expected["predicted_class"]
    assert result.class_probs["malignant"] == pytest.approx(expected["p_malignant"], abs=1e-3)
    assert int(result.mask.sum()) == pytest.approx(expected["mask_area"], rel=0.01)
