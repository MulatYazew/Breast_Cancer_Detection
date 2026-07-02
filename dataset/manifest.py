"""Build a manifest of the raw BUSI_Jpeg dataset.

BUSI stores each class in an ``images/`` + ``images_mask/`` directory pair.
Images with more than one lesion have multiple mask files sharing the image's
stem, e.g. ``benign (100).png`` is paired with both ``benign (100)_mask.png``
and ``benign (100)_mask_1.png``. This module discovers those groupings so the
rest of the pipeline can treat "one image -> N mask files" uniformly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

MASK_SUFFIX_RE = re.compile(r"_mask(_\d+)?$")


def file_hash(path: str | Path, chunk_size: int = 65536) -> str:
    """SHA-256 hash of a file's bytes, used to detect byte-identical images.

    Shared by ``preprocessing.validation`` (duplicate reporting) and
    ``dataset.splits`` (duplicate-aware split grouping) so both agree on what
    counts as "the same image".
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class ManifestRecord:
    """One raw image and all of its associated lesion mask files."""

    image_id: str          # e.g. "benign (100)"
    label: str              # "benign" | "malignant"
    image_path: Path
    mask_paths: list[Path] = field(default_factory=list)


def _stem_to_image_id(mask_stem: str) -> str:
    """Strip a mask filename's ``_mask`` / ``_mask_N`` suffix to recover the image id."""
    return MASK_SUFFIX_RE.sub("", mask_stem)


def build_manifest(
    benign_dir: str | Path,
    benign_mask_dir: str | Path,
    malignant_dir: str | Path,
    malignant_mask_dir: str | Path,
    image_ext: str = ".png",
) -> list[ManifestRecord]:
    """Scan the raw BUSI directories and pair every image with its mask file(s).

    Returns records sorted by (label, image_id) for deterministic downstream
    ordering (important for reproducible splits).
    """
    records: list[ManifestRecord] = []
    for label, image_dir, mask_dir in (
        ("benign", Path(benign_dir), Path(benign_mask_dir)),
        ("malignant", Path(malignant_dir), Path(malignant_mask_dir)),
    ):
        masks_by_image_id: dict[str, list[Path]] = {}
        for mask_path in sorted(mask_dir.glob(f"*{image_ext}")):
            image_id = _stem_to_image_id(mask_path.stem)
            masks_by_image_id.setdefault(image_id, []).append(mask_path)

        for image_path in sorted(image_dir.glob(f"*{image_ext}")):
            image_id = image_path.stem
            records.append(
                ManifestRecord(
                    image_id=image_id,
                    label=label,
                    image_path=image_path,
                    mask_paths=sorted(masks_by_image_id.get(image_id, [])),
                )
            )

    records.sort(key=lambda r: (r.label, r.image_id))
    return records


def manifest_to_rows(records: list[ManifestRecord]) -> list[dict]:
    """Flatten records into plain dicts (JSON/CSV-serializable, one row per image)."""
    return [
        {
            "image_id": r.image_id,
            "label": r.label,
            "image_path": str(r.image_path),
            "mask_paths": [str(p) for p in r.mask_paths],
            "num_lesions": len(r.mask_paths),
        }
        for r in records
    ]
