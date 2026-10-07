"""Build the similar-case index for Stage 6 from the training split.

    python -m reporting.build_index

For every training lesion: crop the ROI with its ground-truth mask exactly as
Stage 4 training does, embed the masked ROI with the trained ResNet50's
penultimate (2048-d, globally pooled) layer, and save the L2-normalized
vector + true label + a small thumbnail under ``reporting.index_dir``. Only
the training split is indexed, so a validation/test case can never retrieve
itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from inference.pipeline import to_chw_tensor
from preprocessing.normalization import to_unit_range


def resnet_case_embedder(resnet_model: nn.Module, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    """masked ROI (HxWx3 uint8) -> L2-normalized 2048-d ResNet50 feature vector."""
    backbone = nn.Sequential(*list(resnet_model.children())[:-1]).eval()

    def embed(masked_roi: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            vector = backbone(to_chw_tensor(to_unit_range(masked_roi), device)).flatten().cpu().numpy()
        return vector / (np.linalg.norm(vector) + 1e-8)

    return embed


def build_case_index(config: dict) -> Path:
    """Embed every training lesion and write the index under ``reporting.index_dir``."""
    from PIL import Image

    from dataset.lesion_manifest import load_lesion_manifest
    from preprocessing.mask_to_yolo import component_mask
    from preprocessing.normalization import load_image_rgb
    from preprocessing.roi_utils import apply_mask, crop_roi_pair
    from resnet.model import build_resnet50
    from utils.seed import get_device

    device = get_device(config.get("device"))
    model = build_resnet50(pretrained=False, num_classes=config["resnet"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(config["inference"]["resnet_weights"], map_location=device)["model_state"])
    embed = resnet_case_embedder(model.eval(), device)

    index_dir = Path(config["reporting"]["index_dir"])
    thumbs_dir = index_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    roi_size = tuple(config["preprocessing"]["roi_size"])
    rows = load_lesion_manifest(Path(config["paths"]["splits_dir"]) / "train_lesions.json")

    vectors, meta = [], []
    for row in rows:
        case_id = f"{row['image_id']}#{row['component_label']}"
        image = load_image_rgb(row["image_path"])
        mask = component_mask(row["source_mask_path"], row["component_label"])
        roi, roi_mask, _ = crop_roi_pair(image, mask, tuple(row["bbox"]), pad_ratio=0.0, target_size=roi_size)
        vectors.append(embed(apply_mask(roi, roi_mask)))
        thumb_name = f"{len(meta):04d}.png"
        Image.fromarray(roi.astype(np.uint8)).resize((112, 112)).save(thumbs_dir / thumb_name)
        meta.append({"id": case_id, "label": row["label"], "thumbnail": f"thumbs/{thumb_name}"})

    np.save(index_dir / "case_embeddings.npy", np.stack(vectors).astype(np.float32))
    (index_dir / "cases.json").write_text(json.dumps(meta, indent=1))
    print(f"indexed {len(meta)} training lesions -> {index_dir}")
    return index_dir


def main() -> None:
    from utils.config import load_config

    build_case_index(load_config())


if __name__ == "__main__":
    main()
