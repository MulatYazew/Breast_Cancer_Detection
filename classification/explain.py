"""Grad-CAM overlays for the small classifiers.

A heatmap shows where the gradient was large for the predicted class. It does
not prove that the model found a tumour, and it is not a segmentation.
"""

from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

LIMITATION = (
    "Grad-CAM highlights regions that influenced this class score. "
    "It is not evidence that a lesion was localized correctly, and it is not a diagnosis."
)


def last_conv(model: nn.Module) -> nn.Conv2d:
    found = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            found = module
    if found is None:
        raise ValueError("no Conv2d layer")
    return found


def overlay_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    image_uint8: np.ndarray,
    target_class: int | None = None,
) -> np.ndarray:
    """Return an RGB uint8 overlay. ``image_tensor`` is 1x3xHxW, already normalized."""
    model.eval()
    cam = GradCAM(model=model, target_layers=[last_conv(model)])
    targets = None if target_class is None else [ClassifierOutputTarget(int(target_class))]
    grayscale = cam(input_tensor=image_tensor, targets=targets)[0]
    background = image_uint8.astype(np.float32) / 255.0
    if background.shape[:2] != grayscale.shape:
        raise ValueError("preview and CAM spatial sizes differ")
    return show_cam_on_image(background, grayscale, use_rgb=True)


def save_case_overlays(repo_root, results: dict, limit: int = 2) -> list[str]:
    """Reload only the selected model and write a few Grad-CAM overlays."""
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt

    from classification.data import letterbox
    from classification.models import build_image_classifier
    from preprocessing.normalization import imagenet_normalize, load_image_rgb, to_unit_range
    from utils.seed import get_device

    repo_root = Path(repo_root)
    pointer = json.loads((repo_root / "checkpoints" / "classification" / "selected" / "selected.json").read_text())
    selected = next(item for item in results["models"] if item["backbone"] == pointer["backbone"])
    stored = selected["test_probabilities"]
    threshold = pointer["threshold"]
    split_rows = {}
    for name in ("train", "val", "test"):
        path = repo_root / "dataset" / "splits_classification" / f"{name}.json"
        if path.exists():
            for row in json.loads(path.read_text()):
                split_rows[row["image_id"]] = row

    chosen = []
    for kind, want_wrong in (("correct", False), ("incorrect", True)):
        found = 0
        for image_id, label, probability in zip(stored["image_ids"], stored["labels"], stored["probabilities"]):
            predicted = int(probability >= threshold)
            wrong = predicted != int(label)
            if wrong != want_wrong:
                continue
            if image_id not in split_rows:
                continue
            chosen.append((kind, image_id, int(label), float(probability), split_rows[image_id]["image_path"]))
            found += 1
            if found >= limit:
                break

    device = get_device(None)
    model = build_image_classifier(pointer["backbone"], num_classes=2, pretrained=False).to(device)
    state = __import__("torch").load(pointer["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    out_dir = repo_root / "reports" / "figures" / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for kind, image_id, label, probability, image_path in chosen:
        preview = letterbox(load_image_rgb(image_path), pointer["image_size"])
        tensor = __import__("torch").from_numpy(
            imagenet_normalize(to_unit_range(preview)).transpose(2, 0, 1)
        ).float().unsqueeze(0).to(device)
        overlay = overlay_gradcam(model, tensor, preview, target_class=int(probability >= threshold))
        safe_name = image_id.replace(" ", "_").replace("(", "").replace(")", "")
        destination = out_dir / f"gradcam_{kind}_{safe_name}.png"
        plt.imsave(destination, overlay)
        written.append(str(destination))
        del tensor
    del model
    import gc
    gc.collect()
    if __import__("torch").backends.mps.is_available():
        __import__("torch").mps.empty_cache()
    return written
