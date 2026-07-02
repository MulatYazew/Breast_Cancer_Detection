"""Dataset and mask validation: pairing, corruption, duplicates, dimension checks.

Stops on critical errors (unreadable images, missing pairs); collects
non-critical issues (mismatched dimensions, degenerate masks) into a report
so they can be reviewed rather than silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from dataset.manifest import ManifestRecord, file_hash
from preprocessing.mask_to_yolo import load_binary_mask


class CriticalValidationError(RuntimeError):
    """Raised when a critical, unrecoverable data issue is found."""


@dataclass
class ValidationIssue:
    severity: str   # "critical" | "warning"
    image_id: str
    kind: str
    detail: str


@dataclass
class ValidationReport:
    num_images: int = 0
    num_masks: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict:
        return {
            "num_images": self.num_images,
            "num_masks": self.num_masks,
            "num_critical": len(self.critical_issues),
            "num_warnings": len(self.warnings),
            "issues": [vars(i) for i in self.issues],
        }


def _is_readable_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - want to report any decoder error
        return False, str(exc)


def validate_dataset(
    records: list[ManifestRecord],
    stop_on_critical: bool = True,
) -> ValidationReport:
    """Validate image/mask pairing, corruption, duplicates, dims, and naming.

    Args:
        records: Output of ``dataset.manifest.build_manifest``.
        stop_on_critical: If True, raise ``CriticalValidationError`` as soon as
            a critical issue is found (missing pair, corrupted file, empty
            mask). If False, collect all issues before returning.

    Returns:
        A ``ValidationReport`` summarizing everything found.
    """
    report = ValidationReport(num_images=len(records))
    seen_hashes: dict[str, tuple[str, str]] = {}

    def _flag(severity: str, image_id: str, kind: str, detail: str) -> None:
        issue = ValidationIssue(severity, image_id, kind, detail)
        report.issues.append(issue)
        if stop_on_critical and severity == "critical":
            raise CriticalValidationError(f"[{kind}] {image_id}: {detail}")

    for record in records:
        report.num_masks += len(record.mask_paths)

        # 1. Naming convention: mask files must resolve back to this image id.
        expected_prefix = record.image_id
        for mask_path in record.mask_paths:
            if not mask_path.stem.startswith(expected_prefix):
                _flag(
                    "critical",
                    record.image_id,
                    "naming_mismatch",
                    f"mask {mask_path.name} does not match image id {expected_prefix}",
                )

        # 2. Missing pair.
        if not record.mask_paths:
            _flag("critical", record.image_id, "missing_mask", "no mask file found for image")
            continue
        if not record.image_path.exists():
            _flag("critical", record.image_id, "missing_image", "image file does not exist")
            continue

        # 3. Corrupted files.
        ok, err = _is_readable_image(record.image_path)
        if not ok:
            _flag("critical", record.image_id, "corrupted_image", err)
            continue
        for mask_path in record.mask_paths:
            ok, err = _is_readable_image(mask_path)
            if not ok:
                _flag("critical", record.image_id, "corrupted_mask", f"{mask_path.name}: {err}")

        # 4. Duplicate detection (exact byte-identical files). Cross-class
        # duplicates are flagged distinctly since they carry a leakage risk
        # that dataset.splits handles by keeping duplicate groups together.
        img_hash = file_hash(record.image_path)
        if img_hash in seen_hashes:
            other_id, other_label = seen_hashes[img_hash]
            kind = "duplicate_image_cross_class" if other_label != record.label else "duplicate_image"
            _flag("warning", record.image_id, kind, f"identical to {other_id} ({other_label})")
        else:
            seen_hashes[img_hash] = (record.image_id, record.label)

        # 5. Dimension mismatch between image and each mask.
        with Image.open(record.image_path) as img:
            img_size = img.size  # (W, H)
        for mask_path in record.mask_paths:
            with Image.open(mask_path) as m:
                mask_size = m.size
            if mask_size != img_size:
                _flag(
                    "critical",
                    record.image_id,
                    "dimension_mismatch",
                    f"image {img_size} vs mask {mask_path.name} {mask_size}",
                )

        # 6. Mask validity: binary after threshold, non-empty.
        for mask_path in record.mask_paths:
            try:
                binary = load_binary_mask(mask_path)
            except Exception as exc:  # noqa: BLE001
                _flag("critical", record.image_id, "unreadable_mask", str(exc))
                continue
            if binary.sum() == 0:
                _flag(
                    "critical",
                    record.image_id,
                    "empty_mask",
                    f"{mask_path.name} has no foreground pixels",
                )

    return report


def report_summary(report: ValidationReport) -> str:
    """Human-readable one-paragraph summary for logging/CLI output."""
    return (
        f"Validated {report.num_images} images / {report.num_masks} mask files: "
        f"{len(report.critical_issues)} critical issue(s), "
        f"{len(report.warnings)} warning(s)."
    )
