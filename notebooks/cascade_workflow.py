"""Helpers for the single end-to-end ultrasound notebook.

Vision models are loaded one at a time. The language-model call is separate
and is refused for MedGemma 27B and for the 8.60 GB MedGemma 1.5 4B checkpoint.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import numpy as np

from evaluation.metrics import bbox_iou, segmentation_metrics
from preprocessing.mask_to_yolo import load_binary_mask, mask_components
from preprocessing.normalization import load_image_rgb
from preprocessing.roi_utils import apply_mask, crop_roi_pair
from utils.hardware import assert_model_id_allowed, current_rss_bytes

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def rss_gib() -> float:
    return current_rss_bytes() / 1024**3


def check_image_quality(image: np.ndarray) -> dict:
    comments = []
    if image.ndim != 3 or image.shape[2] != 3:
        comments.append("The file did not load as an RGB image.")
    height, width = image.shape[:2]
    if min(height, width) < 64:
        comments.append("The image is smaller than 64 pixels on one side.")
    if float(image.std()) < 5:
        comments.append("The pixel variation is very low, so the frame may be blank.")
    return {
        "status": "pass" if not comments else "needs_review",
        "height": int(height),
        "width": int(width),
        "comments": comments,
    }


def read_split_rows(repo_root: Path, split_name: str = "test") -> list[dict]:
    path = repo_root / "dataset" / "splits" / f"{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file is missing: {path}")
    rows = json.loads(path.read_text())
    for row in rows:
        row["image_path"] = str(repo_root / row["image_path"])
        row["mask_paths"] = [str(repo_root / mask) for mask in row["mask_paths"]]
    return rows


def union_mask(mask_paths: list[str], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=np.uint8)
    for mask_path in mask_paths:
        mask = load_binary_mask(mask_path)
        if mask.shape != shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image {shape}: {mask_path}")
        union = np.maximum(union, mask)
    return union


def ground_truth_boxes(mask: np.ndarray, min_area_px: int) -> list[tuple[int, int, int, int]]:
    components = mask_components(mask, min_area_px=min_area_px, pad_ratio=0.0)
    return [component.raw_bbox for component in components]


def checkpoint_status(path: Path) -> dict:
    if path.exists():
        return {"available": True, "path": str(path), "bytes": path.stat().st_size}
    return {"available": False, "path": str(path), "bytes": 0, "reason": "checkpoint file is missing"}


def load_yolo(weights: Path):
    from yolo.model import load_trained_yolo

    if not weights.exists():
        raise FileNotFoundError(weights)
    return load_trained_yolo(weights)


def detect_image(model, image: np.ndarray, imgsz: int, conf: float, iou: float, device: str):
    from yolo.predict import detect_tumor

    return detect_tumor(model, image, imgsz=imgsz, conf_threshold=conf, iou_threshold=iou, device=device)


def load_unet(config: dict, weights: Path, device):
    import torch
    from unet.model import build_unet

    if not weights.exists():
        raise FileNotFoundError(weights)
    unet_cfg = config["unet"]
    model = build_unet(
        architecture=unet_cfg["architecture"],
        encoder_name=unet_cfg["encoder"],
        encoder_weights=None,
        in_channels=unet_cfg["in_channels"],
        classes=unet_cfg["classes"],
    ).to(device)
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def segment_roi(model, roi: np.ndarray, device) -> np.ndarray:
    import torch

    from inference.pipeline import to_chw_tensor
    from preprocessing.normalization import to_unit_range

    tensor = to_chw_tensor(to_unit_range(roi), device)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return (probability > 0.5).astype(np.uint8)


def load_resnet(config: dict, weights: Path, device):
    import torch
    from resnet.model import build_resnet50

    if not weights.exists():
        raise FileNotFoundError(weights)
    model = build_resnet50(pretrained=False, num_classes=config["resnet"]["num_classes"]).to(device)
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def classify_roi(model, masked_roi: np.ndarray, device) -> dict:
    import torch

    from dataset.busi_dataset import CLASS_NAMES
    from inference.pipeline import to_chw_tensor
    from preprocessing.normalization import to_unit_range

    tensor = to_chw_tensor(to_unit_range(masked_roi), device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    index = int(np.argmax(probabilities))
    return {
        "predicted_class": CLASS_NAMES[index],
        "confidence": float(probabilities[index]),
        "probabilities": {name: float(value) for name, value in zip(CLASS_NAMES, probabilities)},
    }


def evaluate_detection(model, rows: list[dict], conf: float, iou_nms: float, imgsz: int, device: str,
                       match_iou: float = 0.5) -> dict:
    true_positive = false_positive = false_negative = 0
    false_positive_ids = []
    false_negative_ids = []
    for row in rows:
        image = load_image_rgb(row["image_path"])
        mask = union_mask(row["mask_paths"], image.shape[:2])
        truth = ground_truth_boxes(mask, min_area_px=20)
        detections = detect_image(model, image, imgsz, conf, iou_nms, device)
        prediction = detections[0][0] if detections else None
        if prediction is None:
            false_negative += len(truth)
            if truth:
                false_negative_ids.append(row["image_id"])
            continue
        best = max((bbox_iou(prediction, box) for box in truth), default=0.0)
        if best >= match_iou:
            true_positive += 1
            if len(truth) > 1:
                false_negative += len(truth) - 1
        else:
            false_positive += 1
            false_positive_ids.append(row["image_id"])
            false_negative += len(truth)
            if truth:
                false_negative_ids.append(row["image_id"])
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    return {
        "images": len(rows),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "match_iou": match_iou,
        "false_positive_ids": false_positive_ids[:12],
        "false_negative_ids": false_negative_ids[:12],
        "note": "A detection matches when its box has IoU of at least 0.5 with a ground-truth mask box. This is a research count on the listed images.",
    }


def evaluate_segmentation(model, rows: list[dict], device, roi_size: tuple[int, int]) -> dict:
    scores = []
    for row in rows:
        image = load_image_rgb(row["image_path"])
        mask = union_mask(row["mask_paths"], image.shape[:2])
        boxes = ground_truth_boxes(mask, min_area_px=20)
        if not boxes:
            continue
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        roi, roi_mask, _ = crop_roi_pair(image, mask, (x0, y0, x1, y1), pad_ratio=0.08, target_size=roi_size)
        prediction = segment_roi(model, roi, device)
        scores.append(segmentation_metrics(prediction, roi_mask))
    if not scores:
        return {"images": 0, "status": "unavailable", "reason": "No ground-truth mask boxes were available."}
    keys = scores[0].keys()
    return {
        "images": len(scores),
        "status": "executed",
        "mean": {key: float(np.mean([item[key] for item in scores])) for key in keys},
        "note": "Metrics use the ground-truth lesion ROI, so they measure the segmenter rather than the detector.",
    }


def evaluate_classification(model, rows: list[dict], device, roi_size: tuple[int, int]) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
    )

    labels = []
    probabilities = []
    predictions = []
    for row in rows:
        image = load_image_rgb(row["image_path"])
        mask = union_mask(row["mask_paths"], image.shape[:2])
        boxes = ground_truth_boxes(mask, min_area_px=20)
        if not boxes:
            continue
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        roi, roi_mask, _ = crop_roi_pair(image, mask, (x0, y0, x1, y1), pad_ratio=0.08, target_size=roi_size)
        result = classify_roi(model, apply_mask(roi, roi_mask), device)
        labels.append(1 if row["label"] == "malignant" else 0)
        probabilities.append(result["probabilities"]["malignant"])
        predictions.append(1 if result["predicted_class"] == "malignant" else 0)
    if not labels:
        return {"images": 0, "status": "unavailable", "reason": "No labeled ROIs were available."}
    y_true = np.array(labels)
    y_prob = np.array(probabilities)
    y_pred = np.array(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    roc = roc_curve(y_true, y_prob) if len(set(labels)) > 1 else None
    pr = precision_recall_curve(y_true, y_prob) if len(set(labels)) > 1 else None
    return {
        "images": len(labels),
        "status": "executed",
        "sensitivity_malignant": float(recall),
        "precision_malignant": float(precision),
        "f1_malignant": float(f1),
        "specificity_benign": float(matrix[0][0] / max(1, matrix[0][0] + matrix[0][1])),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if roc else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if pr else None,
        "confusion_matrix": matrix,
        "roc_curve": None if roc is None else {"fpr": roc[0].tolist(), "tpr": roc[1].tolist()},
        "pr_curve": None if pr is None else {"precision": pr[0].tolist(), "recall": pr[1].tolist()},
        "note": "The positive class is the malignant folder label. The input is the ground-truth masked ROI. This is not a clinical sensitivity.",
    }


def build_or_load_faiss(repo_root: Path, sources: list[dict], embed_model_id: str, index_dir: Path,
                        chunk_words: int, overlap_words: int, rebuild: bool) -> dict:
    import faiss

    from bus_research.rag import ingest_sources
    from reporting.retrieval import sentence_embedder

    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists() and not rebuild:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("embedding_model_id") == embed_model_id and (index_dir / "vectors.faiss").exists():
            return {
                "status": "loaded",
                "index_dir": str(index_dir),
                "n_chunks": manifest["n_chunks"],
                "embedding_model_id": embed_model_id,
                "manifest": manifest,
            }
    chunks, records = ingest_sources(
        sources, repo_root, chunk_words=chunk_words, overlap_words=overlap_words, forbidden_dirs=["results", "dataset/splits"]
    )
    if not chunks:
        raise RuntimeError("No BI-RADS chunks were ingested. Check that the PDF paths exist.")
    embed = sentence_embedder(embed_model_id)
    vectors = np.asarray(embed([f"{chunk.section or ''}. {chunk.text}" for chunk in chunks]), dtype=np.float32)
    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(index_dir / "vectors.faiss"))
    (index_dir / "chunks.jsonl").write_text("\n".join(json.dumps(chunk.to_dict()) for chunk in chunks) + "\n")
    manifest = {
        "embedding_model_id": embed_model_id,
        "n_chunks": len(chunks),
        "sources": records,
        "index": "faiss.IndexFlatIP",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    del embed
    release_memory()
    return {"status": "built", "index_dir": str(index_dir), "n_chunks": len(chunks), "embedding_model_id": embed_model_id, "manifest": manifest}


def search_faiss(index_dir: Path, query: str, k: int = 5, embed=None) -> list[dict]:
    import faiss

    from bus_research.rag import Chunk
    from reporting.retrieval import sentence_embedder

    manifest = json.loads((index_dir / "manifest.json").read_text())
    chunks = [Chunk(**json.loads(line)) for line in (index_dir / "chunks.jsonl").read_text().splitlines() if line]
    faiss_index = faiss.read_index(str(index_dir / "vectors.faiss"))
    owns_embedder = embed is None
    if owns_embedder:
        embed = sentence_embedder(manifest["embedding_model_id"])
    vector = np.asarray(embed([query]), dtype=np.float32)
    scores, indices = faiss_index.search(vector, min(k, len(chunks)))
    hits = []
    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue
        chunk = chunks[int(index)]
        hits.append({**chunk.to_dict(), "score": round(float(score), 4), "retrieval": "faiss-cosine"})
    if owns_embedder:
        del embed
        release_memory()
    return hits


def retrieval_recall(index_dir: Path, questions: list[dict], k: int = 5) -> dict:
    from reporting.retrieval import sentence_embedder

    manifest = json.loads((index_dir / "manifest.json").read_text())
    embed = sentence_embedder(manifest["embedding_model_id"])
    hits_at_k = 0
    details = []
    for question in questions:
        hits = search_faiss(index_dir, question["query"], k=k, embed=embed)
        relevant = False
        for hit in hits:
            text = hit["text"].lower()
            if hit["source_id"] == question["source_id"] and any(phrase.lower() in text for phrase in question["phrases"]):
                relevant = True
                break
        hits_at_k += int(relevant)
        details.append({"id": question["id"], "hit": relevant, "top_source": hits[0]["source_id"] if hits else None, "top_page": hits[0]["page"] if hits else None})
    del embed
    release_memory()
    return {"questions": len(questions), "recall_at_5": hits_at_k / len(questions) if questions else 0.0, "details": details}


def medgemma_decision() -> dict:
    """Record why MedGemma is not loaded. This does not download either checkpoint."""
    decisions = []
    for model_id in ("google/medgemma-27b-text-it", "google/medgemma-1.5-4b-it"):
        try:
            assert_model_id_allowed(model_id)
            decisions.append({"model_id": model_id, "allowed": True})
        except RuntimeError as exc:
            decisions.append({"model_id": model_id, "allowed": False, "reason": str(exc)})
    return {
        "medgemma_27b": "removed; not a candidate",
        "medgemma_4b_published_gb": 8.60,
        "medgemma_4b_downloaded": False,
        "decisions": decisions,
        "local_generator": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "local_generator_note": "A general 1.1B chat model already stored on this machine. It is not a medical foundation model and it does not assign a diagnosis.",
    }


def generate_with_local_llm(model_id: str, messages: list[dict], max_new_tokens: int = 280) -> dict:
    """Load one small local chat model, generate once, then unload it."""
    assert_model_id_allowed(model_id)
    before = current_rss_bytes()
    from reporting.llm import LocalLLM

    llm = LocalLLM(model_id, max_new_tokens=max_new_tokens, temperature=0.0)
    loaded = current_rss_bytes()
    parameter_count = sum(parameter.numel() for parameter in llm.model.parameters())
    try:
        text = llm.generate(messages)
        generated_rss = current_rss_bytes()
    finally:
        del llm
        release_memory()
    after = current_rss_bytes()
    return {
        "model_id": model_id,
        "text": text,
        "parameter_count": parameter_count,
        "rss_before_bytes": before,
        "rss_loaded_bytes": max(loaded, generated_rss),
        "rss_after_bytes": after,
        "executed": True,
    }


def parse_llm_json(text: str) -> dict | None:
    match = JSON_OBJECT.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "findings" not in payload and "impression" not in payload:
        return None
    return payload


def assemble_report(study_id: str, quality: dict, detection: dict, segmentation: dict, classification: dict,
                    geometry: dict, passages: list[dict], llm_payload: dict | None, llm_executed: bool,
                    workflow_log: list[dict], physician_approved: bool) -> dict:
    from bus_research.structured_report import (
        AssessmentBlock,
        ClassificationBlock,
        DescriptorBlock,
        DetectionBlock,
        ImageQualityBlock,
        ReportText,
        SegmentationBlock,
        StructuredStudyReport,
        WorkflowBlock,
        unsupported_claim_flags,
    )

    findings = ""
    impression = ""
    limitations = [
        "Research output only. Folder labels are not a pathology report stored in this repository.",
        "No BI-RADS descriptor model was trained, so descriptor fields stay empty.",
        "No BI-RADS assessment is assigned by a model.",
    ]
    if not llm_executed:
        limitations.append("Report generation by an LLM was not executed.")
    elif not llm_payload:
        limitations.append("The local LLM ran, but its output was not a report JSON object with findings or impression, so it was not used as the report.")
    else:
        findings = str(llm_payload.get("findings") or "")
        impression = str(llm_payload.get("impression") or "")
        extra = llm_payload.get("limitations") or []
        if isinstance(extra, str):
            extra = [extra]
        limitations.extend(str(item) for item in extra)
    citations = [
        {
            "source_id": hit["source_id"],
            "title": hit["title"],
            "citation": hit["citation"],
            "page": hit["page"],
            "section": hit["section"],
            "score": hit["score"],
        }
        for hit in passages
    ]
    report = StructuredStudyReport(
        study_id=study_id,
        image_quality=ImageQualityBlock(status=quality["status"], comments=quality["comments"], physician_review_status="pending"),
        detection=DetectionBlock(
            lesion_detected=detection.get("lesion_detected"),
            bounding_box=detection.get("bounding_box"),
            confidence=detection.get("confidence"),
            physician_review_status="pending",
        ),
        segmentation=SegmentationBlock(
            mask_reference=segmentation.get("mask_reference"),
            confidence=None,
            physician_review_status="pending",
        ),
        classification=ClassificationBlock(
            predicted_class=classification.get("predicted_class"),
            probability=classification.get("probability"),
            physician_review_status="pending",
        ),
        birads_descriptors=DescriptorBlock(),
        birads_assessment=AssessmentBlock(),
        auxiliary_geometry=geometry,
        report=ReportText(findings=findings, impression=impression, limitations=limitations, source_references=citations),
        workflow=WorkflowBlock(physician_approved=physician_approved, audit_log=workflow_log),
    )
    validated = StructuredStudyReport.model_validate(report.model_dump())
    flags = unsupported_claim_flags(validated)
    prose_flags = [flag for flag in flags if flag.startswith("Prose") or flag.startswith("model_suggestion")]
    if prose_flags:
        validated.report.findings = ""
        validated.report.impression = ""
        validated.report.limitations.append("LLM prose was rejected: " + " ".join(prose_flags))
        validated = StructuredStudyReport.model_validate(validated.model_dump())
    return {
        "report": validated.model_dump(),
        "claim_flags": flags,
        "prose_rejected": prose_flags,
        "schema_valid": True,
        "llm_text_used": bool(findings) and not prose_flags,
    }
