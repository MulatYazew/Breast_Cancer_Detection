"""Programmatic audit of the BUSI image tree, masks, YOLO labels, and references.

The module discovers formats, sizes, duplicates, split leakage, and whether
BI-RADS descriptor or assessment labels exist. It does not invent labels.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from dataset.manifest import build_manifest

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
LABELISH_SUFFIXES = {".csv", ".json", ".txt", ".tsv", ".xlsx"}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _dhash(gray_9x8: np.ndarray) -> int:
    bits = (gray_9x8[:, 1:] > gray_9x8[:, :-1]).astype(np.uint8).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _probe_image(path: Path) -> dict:
    """Read one image. Failures are returned, not raised."""
    try:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            width, height = image.size
            bands = list(image.getbands())
            gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
            small = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
        return {
            "ok": True,
            "mode": mode,
            "width": width,
            "height": height,
            "bands": bands,
            "n_channels": len(bands),
            "dhash": _dhash(gray),
            "small": small,
            "error": None,
        }
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _load_split_ids(splits_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not splits_dir.exists():
        return mapping
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        for row in rows:
            mapping[row["image_id"]] = name
    return mapping


def _audit_yolo(yolo_dir: Path) -> dict:
    report: dict = {"present": yolo_dir.exists(), "splits": {}}
    if not yolo_dir.exists():
        return report
    for split in ("train", "val", "test"):
        image_dir = yolo_dir / "images" / split
        label_dir = yolo_dir / "labels" / split
        images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if image_dir.exists() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        invalid = []
        class_ids: Counter = Counter()
        n_boxes = 0
        empty = []
        for label_path in labels:
            text = label_path.read_text().strip()
            if not text:
                empty.append(label_path.stem)
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    invalid.append({"file": label_path.name, "line": line_no, "reason": "expected 5 fields"})
                    continue
                try:
                    cls = int(parts[0])
                    xc, yc, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    invalid.append({"file": label_path.name, "line": line_no, "reason": "non-numeric"})
                    continue
                class_ids[str(cls)] += 1
                n_boxes += 1
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    invalid.append({"file": label_path.name, "line": line_no, "reason": "box outside normalized range"})
                if xc - w / 2 < -1e-3 or yc - h / 2 < -1e-3 or xc + w / 2 > 1 + 1e-3 or yc + h / 2 > 1 + 1e-3:
                    invalid.append({"file": label_path.name, "line": line_no, "reason": "box extends outside the image"})
        report["splits"][split] = {
            "n_images": len(images),
            "n_label_files": len(labels),
            "images_missing_labels": sorted(image_stems - label_stems),
            "labels_missing_images": sorted(label_stems - image_stems),
            "empty_label_files": empty,
            "n_boxes": n_boxes,
            "class_id_counts": dict(class_ids),
            "n_invalid_boxes": len(invalid),
            "invalid_box_examples": invalid[:20],
        }
    return report


def _search_birads_labels(dataset_root: Path) -> dict:
    """Look for annotation tables. Reference PDFs are not labels."""
    hits = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in LABELISH_SUFFIXES:
            continue
        if "yolo_dataset" in path.parts or path.name.endswith(".cache"):
            continue
        try:
            sample = path.read_text(errors="ignore")[:4000].lower()
        except OSError:
            continue
        if "birads" in sample or "bi-rads" in sample or "descriptor" in sample:
            hits.append(str(path.relative_to(dataset_root.parent)) if dataset_root.parent in path.parents else str(path))
    pdfs = sorted(str(p.relative_to(dataset_root.parent)) for p in dataset_root.parent.glob("dataset/*.pdf"))
    pdfs += sorted(
        str(p.relative_to(dataset_root.parent))
        for p in dataset_root.rglob("*.pdf")
        if "dataset" in p.parts
    )
    # de-duplicate while keeping order
    seen = set()
    reference_pdfs = []
    for item in pdfs:
        if item not in seen:
            seen.add(item)
            reference_pdfs.append(item)
    return {
        "descriptor_or_assessment_tables": hits,
        "reference_pdfs": reference_pdfs,
        "radiologist_descriptor_labels_available": False,
        "radiologist_assessment_labels_available": False,
        "reason": (
            "No CSV/JSON/TXT annotation table of BI-RADS descriptors or assessment "
            "categories was found. The PDFs are terminology references, not image labels. "
            "Folder names supply benign/malignant only."
        ),
    }


def run_audit(repo_root: str | Path, near_hamming_max: int = 5) -> dict:
    """Audit the dataset described by ``config.yaml`` paths under ``repo_root``."""
    repo_root = Path(repo_root).resolve()
    import yaml

    config = yaml.safe_load((repo_root / "config.yaml").read_text())
    paths = config["paths"]
    benign = repo_root / paths["raw_benign"]
    benign_mask = repo_root / paths["raw_benign_mask"]
    malignant = repo_root / paths["raw_malignant"]
    malignant_mask = repo_root / paths["raw_malignant_mask"]

    records = build_manifest(benign, benign_mask, malignant, malignant_mask)
    split_of = _load_split_ids(repo_root / paths["splits_dir"])

    images = []
    unreadable = []
    hashes: dict[str, list[str]] = defaultdict(list)
    probes: dict[str, dict] = {}
    modes: Counter = Counter()
    suffixes: Counter = Counter()
    widths, heights = [], []

    for record in records:
        suffixes[record.image_path.suffix.lower()] += 1
        digest = _sha256(record.image_path)
        hashes[digest].append(record.image_id)
        probe = _probe_image(record.image_path)
        if not probe["ok"]:
            unreadable.append({"image_id": record.image_id, "error": probe["error"]})
            continue
        modes[probe["mode"]] += 1
        widths.append(probe["width"])
        heights.append(probe["height"])
        probes[record.image_id] = probe
        images.append(record)

    exact_groups = [sorted(ids) for ids in hashes.values() if len(ids) > 1]

    near_pairs = []
    ids = [r.image_id for r in images if r.image_id in probes]
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            distance = (probes[left]["dhash"] ^ probes[right]["dhash"]).bit_count()
            if distance > near_hamming_max or distance == 0:
                continue
            a = probes[left]["small"].ravel()
            b = probes[right]["small"].ravel()
            corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
            label_l = next(r.label for r in images if r.image_id == left)
            label_r = next(r.label for r in images if r.image_id == right)
            split_l = split_of.get(left)
            split_r = split_of.get(right)
            near_pairs.append(
                {
                    "image_a": left,
                    "image_b": right,
                    "dhash_hamming": distance,
                    "pixel_corr_32": round(corr, 4),
                    "split_a": split_l,
                    "split_b": split_r,
                    "cross_split": bool(split_l and split_r and split_l != split_r),
                    "cross_label": label_l != label_r,
                }
            )
    near_pairs.sort(key=lambda row: (row["dhash_hamming"], -row["pixel_corr_32"], row["image_a"], row["image_b"]))

    mask_counts = Counter(len(r.mask_paths) for r in records)
    missing_masks = [r.image_id for r in records if not r.mask_paths]
    class_counts = Counter(r.label for r in records)
    lesions = Counter()
    for record in records:
        lesions[record.label] += len(record.mask_paths)

    split_counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        split_counts[split_of.get(record.image_id, "unassigned")][record.label] += 1

    yolo = _audit_yolo(repo_root / paths["yolo_dataset_dir"])
    references = _search_birads_labels(repo_root / "dataset")

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BUSI JPEG copy configured in config.yaml",
        "label_source": {
            "classification": "directory name (benign or malignant)",
            "histopathology_table_in_repository": False,
            "patient_identifiers_available": False,
            "examination_identifiers_available": False,
            "note": (
                "Filenames such as 'benign (100)' are dataset indices, not patient or "
                "accession numbers. Images from one patient cannot be grouped, so the "
                "existing split is image-level. Byte-identical files are kept together; "
                "near-duplicates are not."
            ),
        },
        "n_images": len(records),
        "class_counts": dict(class_counts),
        "class_balance": {
            "benign_fraction": round(class_counts["benign"] / max(len(records), 1), 4),
            "malignant_fraction": round(class_counts["malignant"] / max(len(records), 1), 4),
            "malignant_to_benign_ratio": round(class_counts["malignant"] / max(class_counts["benign"], 1), 4),
        },
        "formats": dict(suffixes),
        "modes": dict(modes),
        "channels_observed": sorted({probes[i]["n_channels"] for i in probes}),
        "unreadable_or_corrupted": unreadable,
        "image_size_px": {
            "width_min": int(min(widths)) if widths else None,
            "width_max": int(max(widths)) if widths else None,
            "height_min": int(min(heights)) if heights else None,
            "height_max": int(max(heights)) if heights else None,
            "n_unique_sizes": len({(probes[i]["width"], probes[i]["height"]) for i in probes}),
        },
        "physical_spacing": None,
        "bit_depth_note": "PNG mode is recorded per image. This copy is 8-bit (mode L or RGB), not DICOM.",
        "masks": {
            "images_without_mask": missing_masks,
            "mask_files_per_image": {str(k): v for k, v in sorted(mask_counts.items())},
            "lesion_masks_by_class": dict(lesions),
            "total_lesion_masks": int(sum(lesions.values())),
        },
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_rule": f"dHash Hamming distance 1..{near_hamming_max} (distance 0 is an exact content duplicate)",
        "n_near_duplicate_pairs": len(near_pairs),
        "n_near_duplicate_cross_split": sum(1 for row in near_pairs if row["cross_split"]),
        "n_near_duplicate_cross_label": sum(1 for row in near_pairs if row["cross_label"]),
        "near_duplicate_pairs": near_pairs,
        "splits": {name: dict(counts) for name, counts in split_counts.items()},
        "split_policy": (
            "Existing files in dataset/splits are stratified by folder label and group "
            "byte-identical images. They are not patient-level. Near-duplicates already "
            "cross the frozen split, so published checkpoint metrics are not a clean "
            "patient-level estimate. The split was not rebuilt, because rebuilding it "
            "would make the saved YOLO, U-Net, and ResNet50 checkpoints incomparable."
        ),
        "yolo": yolo,
        "birads": references,
        "normal_class_present": False,
    }


def render_markdown(audit: dict) -> str:
    """Human-readable summary. Pair-level rows stay in the JSON report."""
    size = audit["image_size_px"]
    lines = [
        "# Dataset audit",
        "",
        f"Generated: {audit['created_at_utc']}",
        "",
        "Research use only. This report describes files on disk. It does not add labels.",
        "",
        "## What the files contain",
        "",
        f"- Images: **{audit['n_images']}** "
        f"(benign {audit['class_counts'].get('benign', 0)}, malignant {audit['class_counts'].get('malignant', 0)}).",
        f"- Formats: {audit['formats']}. Pixel modes: {audit['modes']}. Channels: {audit['channels_observed']}.",
        f"- Size range: width {size['width_min']}–{size['width_max']}, height {size['height_min']}–{size['height_max']} "
        f"({size['n_unique_sizes']} unique sizes).",
        f"- Unreadable images: {len(audit['unreadable_or_corrupted'])}.",
        f"- Lesion masks: {audit['masks']['total_lesion_masks']}. Images without a mask: {audit['masks']['images_without_mask'] or 'none'}.",
        f"- Masks per image: {audit['masks']['mask_files_per_image']}.",
        "- No pixel spacing, patient identifier, examination identifier, or `normal` class is stored.",
        "",
        "## Labels that are not in this copy",
        "",
        audit["birads"]["reason"],
        "",
        f"Reference PDFs found: {', '.join(audit['birads']['reference_pdfs']) or 'none'}.",
        "",
        "Classification labels are the folder names `benign` and `malignant`. "
        "No histopathology table is in the repository. Those folder labels must not be converted into BI-RADS categories or descriptors.",
        "",
        "## Duplicates and split leakage",
        "",
        f"- Exact duplicate groups: {audit['exact_duplicate_groups'] or 'none'}.",
        f"- Near-duplicate pairs (dHash Hamming 1–5): **{audit['n_near_duplicate_pairs']}**.",
        f"- Of those, cross-split: **{audit['n_near_duplicate_cross_split']}**; cross-label: **{audit['n_near_duplicate_cross_label']}**.",
        "",
        audit["split_policy"],
        "",
        "## YOLO label files",
        "",
    ]
    for split, info in audit.get("yolo", {}).get("splits", {}).items():
        lines.append(
            f"- `{split}`: {info['n_images']} images, {info['n_boxes']} boxes, "
            f"class ids {info['class_id_counts'] or '{}'}, invalid boxes {info['n_invalid_boxes']}, "
            f"images missing labels {len(info['images_missing_labels'])}."
        )
    lines += [
        "",
        "## Consequence for later stages",
        "",
        "- Detection, segmentation, and benign/malignant classification can be trained from these files.",
        "- A multi-task BI-RADS descriptor model cannot be trained. There is no expert descriptor annotation.",
        "- A BI-RADS assessment model cannot be trained. Categories 0–6 are not image labels here.",
        "- Patient-level metrics cannot be computed until a patient identifier exists.",
        "",
    ]
    return "\n".join(lines)


def write_reports(audit: dict, repo_root: str | Path) -> tuple[Path, Path]:
    repo_root = Path(repo_root)
    out_dir = repo_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(audit))
    json_path = out_dir / "dataset_audit.json"
    md_path = out_dir / "dataset_audit.md"
    json_path.write_text(json.dumps(serializable, indent=2))
    md_path.write_text(render_markdown(serializable))
    _write_class_plot(serializable, repo_root / "reports" / "figures" / "class_distribution.png")
    return json_path, md_path


def _write_class_plot(audit: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = audit["class_counts"]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#4C78A8", "#E45756"])
    ax.set_ylabel("Images")
    ax.set_title("BUSI folder labels in this copy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    audit = run_audit(repo_root)
    json_path, md_path = write_reports(audit, repo_root)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(
        f"images={audit['n_images']} near_pairs={audit['n_near_duplicate_pairs']} "
        f"cross_split={audit['n_near_duplicate_cross_split']}"
    )


if __name__ == "__main__":
    main()
