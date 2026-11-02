"""Stage 5: clinician-in-the-loop Streamlit demo (+ Stage 6 structured report).

Loads the three trained checkpoints once (``st.cache_resource``) and walks a
physician through detect -> review/correct -> segment -> review/correct ->
classify for one or many uploaded ultrasound images, logging every
accept/reject/manual-correction decision for later audit. Once the box and
mask are approved, Stage 6 drafts a BI-RADS report on demand for the
clinician to accept, edit or reject.

All cascade logic (detect/crop/segment/mask/classify) is imported from
``inference.pipeline`` and all reporting logic from ``reporting.service`` —
nothing here duplicates either.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# app.py now lives at the repo root, so this is normally already on sys.path —
# but not every launcher guarantees that (Hugging Face Spaces' runner among
# them), and the repo-root absolute imports below (inference.*,
# preprocessing.*, resnet.*, utils.*) would otherwise fail with
# ModuleNotFoundError. Must run before any of those imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import streamlit as st

# --- streamlit-drawable-canvas compatibility shim ---------------------------
# streamlit-drawable-canvas==0.9.3 calls streamlit.elements.image.image_to_url,
# which modern Streamlit (>=1.4x) no longer exposes at that path. The function
# itself still exists — it moved to streamlit.elements.lib.image_utils and its
# second positional parameter changed from a raw `width: int` to a
# `LayoutConfig` dataclass. A shim that instead fabricates its own
# `data:image/...;base64,...` URL looks plausible but silently breaks: the
# canvas frontend does `e.src = origin + backgroundImageURL` (expecting a
# relative `/media/...` path), and concatenating an origin onto a `data:` URI
# produces an unparseable string, so the background image never loads (no
# exception — just a blank canvas). Only caught by actually driving the
# canvas in a browser. See project memory "gotcha-streamlit-drawable-canvas-compat".
import streamlit.elements.image as _st_image_module

if not hasattr(_st_image_module, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url as _real_image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

    def _image_to_url_shim(image, width, *args, **kwargs) -> str:
        return _real_image_to_url(image, _LayoutConfig(width=width), *args, **kwargs)

    _st_image_module.image_to_url = _image_to_url_shim

from streamlit_drawable_canvas import st_canvas  # noqa: E402

import reporting  # noqa: E402  (dependency-free; heavy reporting deps load lazily)
from inference.pipeline import CADPipeline, CaseResult, apply_mask, to_chw_tensor  # noqa: E402
from preprocessing.roi_utils import crop_roi  # noqa: E402
from resnet.gradcam import build_gradcam, gradcam_overlay  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.seed import get_device  # noqa: E402

logger = get_logger("app")

AUDIT_LOG_PATH = Path("results/inference/audit_log.jsonl")
CANVAS_MAX_WIDTH = 520
ASSETS_DIR = Path(__file__).parent / "assets"
THEME_CSS_PATH = ASSETS_DIR / "theme.css"

REPORT_DISCLAIMER = "AI-generated draft for clinician review. Not a diagnosis."
BIRADS_UNAVAILABLE = "BI-RADS assessment unavailable: validated BI-RADS labels/model not configured."
RESEARCH_OUTPUT_DIR = Path("results/radiomics/app")
# Clinician choices; the model can never propose "6" (or "1") — see reporting.scoring.
CLINICIAN_CATEGORIES = ["0", "1", "2", "3", "4A", "4B", "4C", "5", "6"]
CATEGORY_COLORS = {
    "0": "#6b7280", "1": "#16a34a", "2": "#16a34a", "3": "#0d9488", "4A": "#d97706",
    "4B": "#ea580c", "4C": "#dc2626", "5": "#b91c1c", "6": "#7c3aed",
}
REVIEW_DECISIONS = {"accepted": "accept", "edited": "edit_accept", "rejected": "reject"}


# --------------------------------------------------------------------------- #
# Theme: glassmorphism healthcare UI over the clinician background photo.
# Palette, blur, and per-widget glass styling all live in assets/theme.css —
# this just resolves which photo to use and substitutes it in.
# --------------------------------------------------------------------------- #
def _resolve_background_image() -> Path:
    """Prefer a real user-supplied photo (app/assets/background.*) over the placeholder."""
    for name in ("background.jpg", "background.jpeg", "background.png"):
        candidate = ASSETS_DIR / name
        if candidate.exists():
            return candidate
    return ASSETS_DIR / "background_placeholder.svg"


def apply_theme() -> None:
    """Inject the glassmorphism theme: background photo + palette + card styling."""
    import base64

    image_path = _resolve_background_image()
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    mime = "svg+xml" if image_path.suffix == ".svg" else image_path.suffix.lstrip(".")
    data_uri = f"data:image/{mime};base64,{encoded}"

    css = THEME_CSS_PATH.read_text().replace("__BACKGROUND_IMAGE_DATA_URI__", data_uri)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Model loading (cached so switching cases doesn't reload weights)
# --------------------------------------------------------------------------- #
def _apply_checkpoint_env_overrides(config: dict) -> None:
    """Let the deploy environment (e.g. HF Spaces "Variables and secrets") point
    at checkpoint files without editing config.yaml. Falls back to the
    config.yaml paths untouched when a var isn't set — today that's an empty
    checkpoints/ tree, which load_pipeline()'s FileNotFoundError handling below
    turns into the maintenance-mode UI rather than a crash."""
    env_by_key = {
        "yolo_weights": "YOLO_WEIGHTS_PATH",
        "unet_weights": "UNET_WEIGHTS_PATH",
        "resnet_weights": "RESNET_WEIGHTS_PATH",
    }
    for config_key, env_var in env_by_key.items():
        value = os.getenv(env_var)
        if value:
            config["inference"][config_key] = value


@st.cache_resource(show_spinner="Loading YOLO26n, U-Net, and ResNet50 checkpoints...")
def load_pipeline() -> CADPipeline:
    config = load_config()
    _apply_checkpoint_env_overrides(config)
    device = get_device(config.get("device"))
    return CADPipeline(config, device=device)


def log_decision(case_id: str, stage: str, decision: str, detail: dict | None = None) -> None:
    """Append one audit-log line: every accept/reject/manual correction."""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "stage": stage,
        "decision": decision,
        "detail": detail or {},
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# Stage 6 reporting service (lazy: loaded on the first "Generate" click only)
# --------------------------------------------------------------------------- #
def get_reporting_config() -> dict:
    return load_config().get("reporting") or {"enabled": False, "backend": "off"}


def _hf_token() -> str | None:
    try:
        token = st.secrets.get("HF_TOKEN")
    except Exception:  # noqa: BLE001 - no secrets.toml at all raises
        token = None
    return token or os.getenv("HF_TOKEN")


@st.cache_resource(show_spinner="Loading reporting models (first report only)...")
def get_report_service(_pipeline: CADPipeline):
    """Embedding model + FAISS indexes + LLM, loaded once per server process."""
    from reporting.build_index import resnet_case_embedder
    from reporting.service import build_report_service

    case_embedder = resnet_case_embedder(_pipeline.resnet_model, _pipeline.device)
    return build_report_service(load_config(), hf_token=_hf_token(), case_embedder=case_embedder)


# --------------------------------------------------------------------------- #
# Canvas <-> pixel-coordinate helpers
# --------------------------------------------------------------------------- #
def scale_for_canvas(image: np.ndarray, max_width: int = CANVAS_MAX_WIDTH) -> tuple[np.ndarray, float]:
    """Downscale ``image`` to fit the canvas width; return (resized, scale_factor)."""
    import cv2

    h, w = image.shape[:2]
    scale = min(1.0, max_width / w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image
    return resized, scale


def canvas_rect_to_bbox(objects: list[dict], scale: float) -> tuple[int, int, int, int] | None:
    """Convert the last drawn rectangle on the canvas back to original-image pixel bbox."""
    rects = [o for o in objects if o.get("type") == "rect"]
    if not rects:
        return None
    rect = rects[-1]
    x = rect["left"] / scale
    y = rect["top"] / scale
    w = rect["width"] * rect.get("scaleX", 1.0) / scale
    h = rect["height"] * rect.get("scaleY", 1.0) / scale
    return int(x), int(y), int(x + w), int(y + h)


def canvas_freedraw_to_mask(image_data: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Convert the canvas's painted RGBA layer into a binary mask at ``target_shape``."""
    import cv2

    alpha = image_data[..., 3]
    mask = (alpha > 0).astype(np.uint8)
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


# --------------------------------------------------------------------------- #
# Per-case state
# --------------------------------------------------------------------------- #
def init_case_state(case_id: str, image: np.ndarray) -> dict:
    if "case_states" not in st.session_state:
        st.session_state.case_states = {}
    if case_id not in st.session_state.case_states:
        st.session_state.case_states[case_id] = {"image": image, "ui_stage": "detect"}
    return st.session_state.case_states[case_id]


def render_case(pipeline: CADPipeline, case_id: str, image: np.ndarray, output_dir: Path) -> None:
    state = init_case_state(case_id, image)
    st.subheader(f"Case: {case_id}")

    # --- Stage 2: Detection --------------------------------------------- #
    if state["ui_stage"] == "detect":
        with st.spinner("Running YOLO26n detection..."):
            detection = pipeline.detect(image)
        if detection is None:
            st.warning("YOLO26n found no tumor candidate. Please draw the ROI manually.")
            state["detected_bbox"] = None
            state["detection_confidence"] = None
        else:
            state["detected_bbox"], state["detection_confidence"] = detection
        state["ui_stage"] = "review_detection"
        st.rerun()

    if state["ui_stage"] == "review_detection":
        st.markdown("### Step 1 — Detection review")
        fig_col, action_col = st.columns([2, 1])
        with fig_col:
            display_img = image.copy()
            if state["detected_bbox"] is not None:
                import cv2

                x1, y1, x2, y2 = state["detected_bbox"]
                display_img = cv2.rectangle(display_img.copy(), (x1, y1), (x2, y2), (0, 255, 0), 3)
            st.image(display_img, caption="YOLO26n detection", use_container_width=True)
            if state["detection_confidence"] is not None:
                st.caption(f"Confidence: {state['detection_confidence']:.1%}")

        with action_col:
            if state["detected_bbox"] is not None and st.button("Accept detection", key=f"accept_det_{case_id}"):
                state["bbox"] = state["detected_bbox"]
                state["bbox_source"] = "model"
                log_decision(case_id, "detection", "accept", {"bbox": list(state["bbox"])})
                state["ui_stage"] = "segment"
                st.rerun()
            if st.button("Reject / draw manually", key=f"reject_det_{case_id}"):
                state["ui_stage"] = "manual_bbox"
                st.rerun()

    if state["ui_stage"] == "manual_bbox":
        st.markdown("### Step 1 (manual) — Draw the tumor bounding box")
        from PIL import Image

        resized, scale = scale_for_canvas(image)
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=3,
            stroke_color="#ff0000",
            background_image=Image.fromarray(resized.astype(np.uint8)),
            update_streamlit=True,
            height=resized.shape[0],
            width=resized.shape[1],
            drawing_mode="rect",
            key=f"canvas_bbox_{case_id}",
        )

        if st.button("Confirm manual box", key=f"confirm_bbox_{case_id}"):
            objects = (canvas_result.json_data or {}).get("objects", []) if canvas_result else []
            bbox = canvas_rect_to_bbox(objects, scale)
            if bbox is None:
                st.error("Please draw a rectangle before confirming.")
            else:
                state["bbox"] = bbox
                state["bbox_source"] = "manual"
                log_decision(case_id, "detection", "manual_correction", {"bbox": list(bbox)})
                state["ui_stage"] = "segment"
                st.rerun()

    # --- Stage 3: Segmentation ------------------------------------------ #
    if state["ui_stage"] == "segment":
        with st.spinner("Cropping ROI and running U-Net segmentation..."):
            pad_ratio = 0.0 if state["bbox_source"] == "manual" else pipeline.bbox_pad_ratio
            roi_crop, padded_bbox = crop_roi(image, state["bbox"], pad_ratio=pad_ratio, target_size=pipeline.roi_size)
            mask = pipeline.segment(roi_crop)
        state["roi_crop"] = roi_crop
        state["padded_bbox"] = padded_bbox
        state["predicted_mask"] = mask
        state["ui_stage"] = "review_segmentation"
        st.rerun()

    if state["ui_stage"] == "review_segmentation":
        st.markdown("### Step 2 — Segmentation review")
        fig_col, action_col = st.columns([2, 1])
        with fig_col:
            overlay = state["roi_crop"].copy()
            overlay[state["predicted_mask"] > 0] = (
                0.6 * overlay[state["predicted_mask"] > 0] + 0.4 * np.array([255, 60, 60])
            ).astype(np.uint8)
            st.image(overlay, caption="U-Net predicted mask", use_container_width=True)

        with action_col:
            if st.button("Accept segmentation", key=f"accept_seg_{case_id}"):
                state["mask"] = state["predicted_mask"]
                state["mask_source"] = "model"
                log_decision(case_id, "segmentation", "accept")
                state["ui_stage"] = "classify"
                st.rerun()
            if st.button("Reject / correct manually", key=f"reject_seg_{case_id}"):
                state["ui_stage"] = "manual_mask"
                st.rerun()

    if state["ui_stage"] == "manual_mask":
        st.markdown("### Step 2 (manual) — Paint over the tumor region")
        from PIL import Image

        roi_resized, scale = scale_for_canvas(state["roi_crop"], max_width=CANVAS_MAX_WIDTH)
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.4)",
            stroke_width=8,
            stroke_color="#ff0000",
            background_image=Image.fromarray(roi_resized.astype(np.uint8)),
            update_streamlit=True,
            height=roi_resized.shape[0],
            width=roi_resized.shape[1],
            drawing_mode="freedraw",
            key=f"canvas_mask_{case_id}",
        )
        st.caption("Paint directly over the tumor region on the canvas above.")

        if st.button("Confirm manual mask", key=f"confirm_mask_{case_id}"):
            if canvas_result is None or canvas_result.image_data is None:
                st.error("Please paint the tumor region before confirming.")
            else:
                mask = canvas_freedraw_to_mask(canvas_result.image_data, state["roi_crop"].shape[:2])
                state["mask"] = mask
                state["mask_source"] = "manual"
                log_decision(case_id, "segmentation", "manual_correction")
                state["ui_stage"] = "classify"
                st.rerun()

    # --- Stage 4: Classification ----------------------------------------- #
    if state["ui_stage"] == "classify":
        with st.spinner("Running ResNet50 classification..."):
            masked_roi = apply_mask(state["roi_crop"], state["mask"])
            predicted_class, confidence, class_probs = pipeline.classify(masked_roi)
        state["masked_roi"] = masked_roi
        state["predicted_class"] = predicted_class
        state["confidence"] = confidence
        state["class_probs"] = class_probs
        log_decision(
            case_id, "classification", "model",
            {"predicted_class": predicted_class, "confidence": confidence},
        )
        state["ui_stage"] = "report"
        st.rerun()

    if state["ui_stage"] == "report":
        st.markdown("### Step 3 — Classification result")
        col1, col2, col3 = st.columns(3)
        col1.image(state["roi_crop"], caption="ROI crop")
        col2.image(state["masked_roi"], caption="Masked ROI (Stage 4 input)")

        with col3:
            try:
                cam = build_gradcam(pipeline.resnet_model)
                from preprocessing.normalization import to_unit_range

                tensor = to_chw_tensor(to_unit_range(state["masked_roi"]), pipeline.device)
                overlay = gradcam_overlay(
                    cam, tensor, to_unit_range(state["masked_roi"]).astype(np.float32)
                )
                st.image(overlay, caption="Grad-CAM")
            except Exception as exc:  # noqa: BLE001 - Grad-CAM is best-effort
                st.caption(f"Grad-CAM unavailable: {exc}")

        st.metric(
            "Prediction",
            state["predicted_class"].upper(),
            f"{state['confidence']:.1%} confidence",
        )
        st.json(state["class_probs"])

        st.markdown("#### Auditability")
        st.write(
            {
                "detection": state["bbox_source"],
                "segmentation": state["mask_source"],
                "classification": "model",
            }
        )
        revise_det, revise_seg = st.columns(2)
        if revise_det.button("Revise detection box", key=f"revise_det_{case_id}"):
            log_decision(case_id, "detection", "revise")
            state["ui_stage"] = "review_detection"
            st.rerun()
        if revise_seg.button("Revise segmentation mask", key=f"revise_seg_{case_id}"):
            log_decision(case_id, "segmentation", "revise")
            state["ui_stage"] = "review_segmentation"
            st.rerun()

        render_report_section(pipeline, case_id, state)
        render_research_section(case_id, state)

        bundle = build_case_bundle(case_id, state)
        st.download_button(
            "Download case report (JSON + images, .zip)",
            data=bundle,
            file_name=f"{case_id}_report.zip",
            mime="application/zip",
            key=f"download_{case_id}",
        )


# --------------------------------------------------------------------------- #
# Stage 6: structured report panel
# --------------------------------------------------------------------------- #
def _report_entry(case_id: str) -> dict | None:
    return st.session_state.get("reports", {}).get(case_id)


def report_status(case_id: str, state: dict) -> str:
    """not_generated | stale | failed | pending_review | accepted | edited | rejected."""
    entry = _report_entry(case_id)
    if entry is None:
        return "not_generated"
    if state.get("mask") is None or entry["key"] != reporting.report_key(state["mask"], state["padded_bbox"]):
        return "stale"
    if entry["result"]["status"] != "ok":
        return "failed"
    return entry["review"]["status"] if entry["review"] else "pending_review"


def _show_category_key(case_id: str) -> str:
    return f"show_rule_category_{case_id}"


def _category_shown(case_id: str) -> bool:
    return bool(st.session_state.get(_show_category_key(case_id), get_reporting_config().get("show_rule_category", False)))


def _gated_report(result: dict, show_category: bool) -> dict:
    """Copy of a Stage 6 result for export, without the rule proposal when it is hidden."""
    if show_category:
        return result
    hidden = dict(result)
    for key in ("category", "likelihood", "recommendation", "patient_summary", "rule_score", "rule_points"):
        hidden[key] = None
    hidden["note"] = BIRADS_UNAVAILABLE + " Experimental rule-based proposal hidden."
    return hidden


def _case_result_from_state(state: dict) -> CaseResult:
    """The case as approved by the clinician; ``bbox`` is the padded box the ROI was cropped from."""
    return CaseResult(
        image=state["image"],
        bbox=state["padded_bbox"],
        bbox_source=state["bbox_source"],
        detection_confidence=state.get("detection_confidence"),
        roi_crop=state["roi_crop"],
        mask=state["mask"],
        mask_source=state["mask_source"],
        masked_roi=state["masked_roi"],
        predicted_class=state["predicted_class"],
        confidence=state["confidence"],
        class_probs=state["class_probs"],
    )


def _generate_report(pipeline: CADPipeline, case_id: str, state: dict, key: str) -> None:
    with st.status("Preparing structured report...", expanded=True) as status:
        try:
            service = get_report_service(pipeline)
        except Exception as exc:  # noqa: BLE001 - model download/auth/OOM: show, don't crash
            logger.exception("Reporting service failed to load")
            status.update(label="Reporting models could not be loaded", state="error")
            st.error(f"The reporting models could not be loaded: {exc}")
            return

        def progress(step: str) -> None:
            status.update(label=f"{step}...")
            st.write(f"• {step}")

        result = service.generate(_case_result_from_state(state), state["mask"], case_id=case_id, progress=progress)
        status.update(
            label="Draft report ready" if result.status == "ok" else "Draft report failed",
            state="complete" if result.status == "ok" else "error",
        )

    st.session_state.setdefault("reports", {})[case_id] = {
        "key": key,
        "result": result.to_dict(),
        "review": None,
        "generated_monotonic": time.monotonic(),
    }
    log_decision(case_id, "report", "generated", {
        "status": result.status,
        "ai_category": result.category,
        "error_kind": result.error_kind,
        "model_id": result.model_id,
        "backend": result.backend,
        "prompt_version": result.prompt_version,
        "kb_version": result.kb_version,
        "retrieved_ids": result.retrieved_ids,
        "similar_case_ids": result.similar_case_ids,
        "mask_source": result.mask_source,
        "report_key": key,
        "attempts": result.attempts,
        "timing": result.timings,
    })


def _record_review(case_id: str, entry: dict, status: str, final_category: str | None,
                   recommendation: str | None, notes: str) -> None:
    result = entry["result"]
    entry["review"] = {
        "status": status,
        "final_category": final_category,
        "recommendation": recommendation,
        "notes": notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    log_decision(case_id, stage="report", decision=REVIEW_DECISIONS[status], detail={
        "ai_category": result["category"],
        "final_category": final_category,
        "changed": final_category is not None and final_category != result["category"],
        "recommendation_changed": recommendation is not None and recommendation != result["recommendation"],
        "notes": notes,
        "model_id": result["model_id"],
        "backend": result["backend"],
        "prompt_version": result["prompt_version"],
        "kb_version": result["kb_version"],
        "retrieved_ids": [h["id"] for h in result["knowledge_hits"]],
        "similar_case_ids": [c["id"] for c in result["similar_cases"]],
        "mask_source": result["mask_source"],
        "bbox_source": result["bbox_source"],
        "report_key": entry["key"],
        "timing": {
            "generation_s": result["timings"],
            "review_latency_s": round(time.monotonic() - entry["generated_monotonic"], 1),
        },
    })


def _category_badge(category: str, caption: str) -> None:
    color = CATEGORY_COLORS.get(category, "#6b7280")
    st.markdown(
        f'<span style="background:{color};color:#fff;padding:0.35rem 0.9rem;border-radius:999px;'
        f'font-weight:700;font-size:1.15rem;">BI-RADS {category}</span>'
        f'&nbsp;&nbsp;<span style="opacity:0.85">{caption}</span>',
        unsafe_allow_html=True,
    )


def _render_features_tab(result: dict) -> None:
    st.caption(
        "Heuristic lexicon estimates computed from the mask outline and pixel statistics — unvalidated, "
        "not radiologist-assessed BI-RADS descriptors."
    )
    rows = [
        {
            "Category": d["category"],
            "Value": d["value"],
            "Assessable": "yes" if d["assessable"] else "not assessable",
            "Supports": d["polarity"],
            "Supporting numbers": ", ".join(f"{k}={v}" for k, v in d["evidence"].items()),
            "Note": d["note"],
        }
        for d in result["descriptors"]
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    with st.expander("Raw radiomic features"):
        st.json(result["features"])


def _render_evidence_tab(result: dict) -> None:
    st.markdown("**Retrieved BI-RADS knowledge**")
    for hit in result["knowledge_hits"]:
        how = "pinned (descriptor definition)" if hit["pinned"] else f"similarity {hit['score']:.2f}"
        with st.expander(f"kb:{hit['id']} · {hit['title']} · {how}"):
            st.write(hit["text"])
    st.markdown("**Most similar training cases** (ResNet50 embedding, ground-truth labels)")
    if not result["similar_cases"]:
        st.caption("Similar-case index not available. Build it with `python -m reporting.build_index`.")
        return
    for col, case in zip(st.columns(len(result["similar_cases"])), result["similar_cases"]):
        if Path(case["thumbnail"]).exists():
            col.image(case["thumbnail"], use_container_width=True)
        col.caption(f"{case['id']}  \n**{case['label']}** · sim {case['similarity']:.2f}")


def _render_draft_tab(result: dict, review: dict | None, show_category: bool) -> None:
    if result["status"] != "ok":
        st.error(result["error"] or "No draft available.")
        st.caption("The model's text is never shown unless it passed validation. See the Features tab.")
        return
    if not show_category:
        # The category, its likelihood band and the recommendation derived from it come from
        # unvalidated rules; BUSI has no BI-RADS labels to validate them against.
        st.info(f"{BIRADS_UNAVAILABLE} The experimental rule-based proposal is hidden.")
        st.markdown("**Rationale (heuristic descriptors)**")
        st.dataframe(
            [{"Descriptor": r["descriptor"], "Value": r["value"], "Supports": r["supports"], "Source": r["source"]}
             for r in result["rationale"]],
            hide_index=True, use_container_width=True,
        )
        if review and review.get("final_category"):
            _category_badge(review["final_category"], "set by clinician")
        if review and review.get("notes"):
            st.markdown("**Clinician notes**")
            st.write(review["notes"])
        st.markdown("**Limitations**")
        st.markdown("\n".join(f"- {item}" for item in result["limitations"]))
        return
    st.warning("Experimental rule-based proposal — unvalidated, not a radiologist BI-RADS assessment.")
    _category_badge(result["category"], f"likelihood of malignancy {result['likelihood']}")
    if review and review["status"] == "edited" and review["final_category"] != result["category"]:
        st.markdown("Clinician-edited final category:")
        _category_badge(review["final_category"], "set by clinician")

    reasons = "; ".join(result["confidence_reasons"])
    if result["confidence"] == "low":
        st.warning(f"⚠️ **Low confidence.** {reasons}")
    elif result["confidence"] == "moderate":
        st.info(f"Moderate confidence. {reasons}")
    else:
        st.success("High confidence: the lexicon features and the classifier agree.")

    st.markdown("**Rationale**")
    st.dataframe(
        [{"Descriptor": r["descriptor"], "Value": r["value"], "Supports": r["supports"], "Source": r["source"]}
         for r in result["rationale"]],
        hide_index=True, use_container_width=True,
    )
    st.markdown("**Recommendation**")
    if review and review["status"] == "edited" and review["recommendation"] != result["recommendation"]:
        st.write(review["recommendation"])
        st.caption(f"Edited by clinician. AI draft: {result['recommendation']}")
    else:
        st.write(result["recommendation"])
    if review and review.get("notes"):
        st.markdown("**Clinician notes**")
        st.write(review["notes"])
    st.markdown("**Limitations**")
    st.markdown("\n".join(f"- {item}" for item in result["limitations"]))
    st.caption(
        f"Model {result['model_id']} ({result['backend']}) · prompt {result['prompt_version']} · "
        f"knowledge base {result['kb_version']} · rule score {result['rule_score']} · "
        f"attempts {result['attempts']} · "
        f"{sum(result['timings'].values()):.1f}s"
    )


def _render_patient_tab(result: dict, review: dict | None, show_category: bool) -> None:
    if not show_category:
        st.info("🔒 The patient summary was drafted from the hidden experimental BI-RADS proposal, so it is hidden too.")
        return
    if result["status"] == "ok" and review and review["status"] in ("accepted", "edited"):
        st.write(result["patient_summary"])
    else:
        st.info("🔒 The patient summary is locked until the clinician accepts the report.")


def render_report_section(pipeline: CADPipeline, case_id: str, state: dict) -> None:
    """Stage 6 panel. Only reachable once the box and mask are approved (``ui_stage == "report"``)."""
    cfg = get_reporting_config()
    backend = cfg.get("backend", "off")
    if backend == "off":
        return
    st.markdown("### Stage 6 — Structured report")
    if not cfg.get("enabled", False):
        st.info("Structured reporting is turned off on this deployment.")
        return
    missing = reporting.missing_dependencies(backend)
    if missing:
        logger.warning("Reporting dependencies missing: %s", missing)
        st.info(
            "Structured reporting isn't installed on this deployment "
            f"(missing: {', '.join(missing)}; see requirements-reporting.txt)."
        )
        return
    if state.get("mask") is None or state.get("bbox") is None:
        st.caption("Available once the detection box and segmentation mask are accepted or corrected.")
        return

    key = reporting.report_key(state["mask"], state["padded_bbox"])
    entry = _report_entry(case_id)
    stale = entry is not None and entry["key"] != key

    if entry is None:
        button_label = "Generate draft report"
    elif stale:
        button_label = "Regenerate draft report"
    elif entry["result"]["status"] != "ok":
        button_label = "Retry"
    else:
        button_label = None  # unchanged inputs: never recompute
    if button_label and st.button(button_label, key=f"generate_report_{case_id}", type="primary"):
        _generate_report(pipeline, case_id, state, key)
        st.rerun()

    if entry is None:
        st.caption("Drafts a BI-RADS report from the approved mask. Runs only when you click the button.")
        return

    result, review = entry["result"], entry["review"]
    if stale:
        st.warning(
            "⚠️ **Stale report.** The box or mask changed after this report was generated. "
            "Regenerate it before accepting."
        )
    if result["status"] == "error" and result["error_kind"] in ("rate_limit", "timeout"):
        st.error(f"{result['error']} Please wait a moment and click **Retry**.")
    elif result["status"] != "ok":
        st.error(result["error"])

    show_category = st.toggle(
        "Show experimental rule-based BI-RADS proposal (unvalidated)",
        value=bool(cfg.get("show_rule_category", False)),
        key=_show_category_key(case_id),
        help="Heuristic rules over heuristic descriptors plus the ResNet50 probability. BUSI has no BI-RADS "
             "labels, so this proposal has never been validated against radiologist assessments.",
    )
    tabs = st.tabs(["Features", "Evidence", "Draft report", "Patient summary"])
    renderers = (
        lambda: _render_features_tab(result),
        lambda: _render_evidence_tab(result),
        lambda: _render_draft_tab(result, review, show_category),
        lambda: _render_patient_tab(result, review, show_category),
    )
    for tab, render in zip(tabs, renderers):
        with tab:
            st.warning(REPORT_DISCLAIMER, icon="🩺")
            render()

    if result["status"] != "ok":
        return
    if review is not None:
        verdict = {"accepted": "Accepted", "edited": "Edited and accepted", "rejected": "Rejected"}[review["status"]]
        final = f" · final category {review['final_category']}" if review["final_category"] else ""
        (st.error if review["status"] == "rejected" else st.success)(f"{verdict} by clinician{final}.")
        return

    edit_flag = f"editing_report_{case_id}"
    accept_col, edit_col, reject_col = st.columns(3)
    if accept_col.button("Accept report", key=f"accept_report_{case_id}", disabled=stale):
        # A hidden proposal is never silently recorded as the accepted category.
        _record_review(case_id, entry, "accepted", result["category"] if show_category else None,
                       result["recommendation"] if show_category else None, "")
        st.rerun()
    if edit_col.button("Edit and accept", key=f"edit_report_{case_id}", disabled=stale):
        st.session_state[edit_flag] = True
    if reject_col.button("Reject", key=f"reject_report_{case_id}"):
        _record_review(case_id, entry, "rejected", None, None, "")
        st.rerun()

    if st.session_state.get(edit_flag) and not stale:
        with st.form(f"edit_report_form_{case_id}"):
            category = st.selectbox(
                "Final BI-RADS category", CLINICIAN_CATEGORIES,
                index=CLINICIAN_CATEGORIES.index(result["category"]) if show_category else None,
                placeholder="Clinician's own assessment",
                help="Category 6 (known biopsy-proven malignancy) can only be set by the clinician.",
            )
            recommendation = st.text_area("Recommendation", value=result["recommendation"] if show_category else "")
            notes = st.text_area("Clinician notes")
            if st.form_submit_button("Save and accept"):
                st.session_state.pop(edit_flag, None)
                _record_review(case_id, entry, "edited", category, recommendation, notes)
                st.rerun()


# --------------------------------------------------------------------------- #
# Quantitative analysis (research): radiomics record -> classifier -> RAG -> report.
# Runtime lives in bus_research/; every model/index is a notebook 03/04 artifact,
# loaded here, never trained here.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading research classifier...")
def get_research_classifier():
    from bus_research.classifier import load_if_present

    return load_if_present(load_config()["research"]["models"]["classifier_dir"])


@st.cache_resource(show_spinner="Loading reference index...")
def get_reference_index():
    from bus_research.rag import ReferenceIndex

    index_dir = Path(load_config()["research"]["rag"]["index_dir"])
    return ReferenceIndex.load(index_dir) if (index_dir / "index_manifest.json").exists() else None


@st.cache_resource(show_spinner="Loading report model (first research report only)...")
def get_research_llm():
    from bus_research.report import build_provider

    return build_provider(load_config(), hf_token=_hf_token())


def _research_case(case_id: str, state: dict) -> dict:
    """Record + prediction + evidence for the approved mask, recomputed only when the image or mask changes.

    Keyed by pixel content and the approved mask/box, not by the upload's filename, so a
    re-uploaded file with the same name can never reuse another image's results.
    """
    from bus_research import birads
    from bus_research.feature_store import build_record, image_content_id, save_record
    from bus_research.radiomics_extract import (UNET_ACTIVATION, UNET_THRESHOLD, RadiomicsSettings,
                                                extract_radiomics_features, native_mask_from_roi)

    image_key = image_content_id(state["image"])
    key = f"{image_key}:{reporting.report_key(state['mask'], state['padded_bbox'])}"
    cache = st.session_state.setdefault("research", {})
    if case_id in cache and cache[case_id]["key"] == key:
        return cache[case_id]

    config = load_config()
    image = state["image"]
    native = native_mask_from_roi(state["mask"], state["padded_bbox"], image.shape)
    result = extract_radiomics_features(image, native, {"roi_bbox": list(state["padded_bbox"])},
                                        RadiomicsSettings.from_config(config))
    manual = state["mask_source"] == "manual"
    record = build_record(
        image_key=image_key, lesion_key="approved_mask", image=image, radiomics_result=result,
        segmentation={"model": "clinician" if manual else "unet", "checkpoint_version": None,
                      "mask_source": "clinician_corrected" if manual else "predicted",
                      "activation": None if manual else UNET_ACTIVATION, "threshold": None if manual else UNET_THRESHOLD,
                      "bbox_source": "manual" if state["bbox_source"] == "manual" else "yolo",
                      "detection_confidence": state.get("detection_confidence"),
                      "roi_bbox_xyxy": [int(v) for v in state["padded_bbox"]]},
        descriptors=birads.empty_descriptors(),
        limitations=["Shape features in pixel units (no physical calibration)."],
        provenance={"pipeline": "app.py research panel"},
    )
    clf = get_research_classifier()
    prediction = clf.predict(record) if clf else {"available": False, "reason": "no trained classifier artifact"}
    if prediction.get("available"):
        record["prediction"] = {k: prediction[k] for k in ("model_id", "prediction_type", "class_probabilities",
                                                            "calibration_status", "calibration_method")}
    save_record(RESEARCH_OUTPUT_DIR, record)
    hits, index_version = [], None
    index = get_reference_index()
    if index is not None:
        from bus_research.rag import build_query

        query = build_query(record, prediction)
        hits = list({h["chunk_id"]: h for h in index.search(query, k=2, tiers=("primary_reference",))
                     + index.search(query, k=config["research"]["rag"]["top_k"])}.values())
        index_version = index.version
    cache[case_id] = {"key": key, "record": record, "prediction": prediction, "hits": hits,
                      "index_version": index_version, "report": None}
    return cache[case_id]


def render_research_section(case_id: str, state: dict) -> None:
    """Optional panel; any missing dependency/artifact degrades to an explanation, never an error."""
    research_cfg = load_config().get("research")
    if not research_cfg:
        return
    st.markdown("### Quantitative analysis (research)")
    st.caption("Research model output — not a clinical diagnosis.")
    import bus_research

    missing = bus_research.missing_dependencies()
    if missing:
        st.info(f"Quantitative analysis isn't installed on this deployment (missing: {', '.join(missing)}; "
                "see requirements-research.txt).")
        return
    if state.get("mask") is None or state.get("padded_bbox") is None:
        st.caption("Available once the detection box and segmentation mask are accepted or corrected.")
        return
    try:
        case = _research_case(case_id, state)
    except Exception as exc:  # noqa: BLE001 - research panel must never break the clinical workflow
        logger.exception("Research analysis failed")
        st.warning(f"Quantitative analysis failed for this case: {exc}")
        return
    record, prediction = case["record"], case["prediction"]
    rad, seg = record["radiomics"], record["segmentation"]

    for issue in seg["quality_issues"]:
        (st.error if issue["severity"] == "fail" else st.warning)(f"Mask quality: {issue['message']}")
    if rad["status"] != "ok":
        st.error("Radiomics extraction failed: " + "; ".join(e["message"] for e in rad["errors"]))
    else:
        from bus_research.report import KEY_FEATURES

        rows = []
        for name, unit in KEY_FEATURES.items():
            family, feature = name.split(".")
            value = rad["features"].get(family, {}).get(feature)
            if value is not None:
                rows.append({"Feature": name, "Value": f"{value:.4g}", "Unit": unit})
        n_feats = sum(len(v) for v in rad["features"].values())
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(f"{n_feats} PyRadiomics features from the original-resolution image and the approved "
                   f"({seg['mask_source'].replace('_', ' ')}) mask · pixel units, no physical calibration · "
                   f"settings {rad['settings_hash']}")

    col_clf, col_birads = st.columns(2)
    with col_clf:
        if prediction.get("available"):
            p = prediction["class_probabilities"]["malignant"]
            st.metric("Radiomics model p(malignant)", f"{p:.1%}")
            calibrated = prediction["calibration_status"] == "calibrated"
            st.caption(f"{prediction['model_id']} · benign vs malignant · "
                       + ("Platt-calibrated on BUSI validation data (not clinical likelihood)" if calibrated
                          else "Probability calibration unavailable."))
            if prediction["ood"]["out_of_distribution"]:
                st.warning("Features are outside the training distribution; this output is unreliable.")
        else:
            st.info(f"Radiomics classifier unavailable: {prediction['reason']}")
    with col_birads:
        st.info(BIRADS_UNAVAILABLE)
        st.caption("BI-RADS descriptors (shape, orientation, margin, echo pattern, posterior features, "
                   "calcifications): Descriptor not assessed. Vascularity and elasticity: not visible on B-mode.")

    if case["hits"]:
        with st.expander(f"Retrieved references ({len(case['hits'])}, index {case['index_version']})"):
            for h in case["hits"]:
                where = f"p{h['page']}" if h.get("page") else (h.get("section") or "")
                st.markdown(f"**`{h['chunk_id']}`** · {h['title']} · {where} · *{h['evidence_tier']}*")
                st.caption(h["text"][:600])
    else:
        st.caption("Reference index not built (run section 03 of notebooks/Breast_CAD_Master.ipynb).")

    if case["report"] is None:
        if st.button("Generate research report", key=f"research_report_{case_id}"):
            from bus_research.report import generate_report

            with st.spinner("Generating grounded report..."):
                try:
                    llm = get_research_llm()
                except Exception as exc:  # noqa: BLE001
                    llm = None
                    st.warning(f"Report model unavailable: {exc}")
                report = generate_report(record, prediction, case["hits"], llm,
                                         max_retries=research_cfg["reports"]["max_retries"],
                                         index_version=case["index_version"])
            case["report"] = report
            log_decision(case_id, "research_report", "generated", {
                "status": report["status"], "record_id": record["record_id"], "provider": report["provider"],
                "llm_model_id": report["llm_model_id"], "rag_index_version": report["rag_index_version"],
                "classifier_model_id": prediction.get("model_id"), "attempts": report["attempts"]})
            st.rerun()
    else:
        report = case["report"]
        st.warning(REPORT_DISCLAIMER, icon="🩺")
        if report["status"] in ("ok", "abstained"):
            st.write(report["summary"])
            for e in report["supporting_evidence"]:
                st.markdown(f"- {e['claim']} " + " ".join(f"`{c}`" for c in e["citations"]))
            st.markdown("**Uncertainty and limitations**")
            st.markdown("\n".join(f"- {x}" for x in report["uncertainty_and_limitations"]))
            st.markdown(f"**Recommended review:** {report['recommended_review']}")
        else:
            st.error(f"No report shown ({report['status']}): {report['error']}")
        st.caption(f"Provider {report['provider']} / {report['llm_model_id']} · attempts {report['attempts']} · "
                   "requires radiologist review")

    from bus_research.feature_store import dumps_strict

    dl_record, dl_report = st.columns(2)
    dl_record.download_button("Download feature record (JSON)", data=dumps_strict(record),
                              file_name=f"{record['record_id']}.json", mime="application/json",
                              key=f"dl_record_{case_id}")
    if case["report"] is not None:
        from bus_research.report import report_to_markdown as research_report_md

        dl_report.download_button("Download research report (Markdown)", data=research_report_md(case["report"]),
                                  file_name=f"{record['record_id']}_report.md", mime="text/markdown",
                                  key=f"dl_report_{case_id}")


def build_case_bundle(case_id: str, state: dict) -> bytes:
    """Zip: result JSON + original/ROI/masked-ROI PNGs (+ Stage 6 report) — the per-case download."""
    from PIL import Image

    entry = _report_entry(case_id)
    status = report_status(case_id, state)
    summary = {
        "case_id": case_id,
        "bbox": list(state["bbox"]),
        "bbox_source": state["bbox_source"],
        "mask_source": state["mask_source"],
        "predicted_class": state["predicted_class"],
        "confidence": state["confidence"],
        "class_probs": state["class_probs"],
        "report_status": status,
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(f"{case_id}_result.json", json.dumps(summary, indent=2))
        if entry is not None:
            from reporting.export import report_to_markdown

            exported = _gated_report(entry["result"], _category_shown(case_id))
            zf.writestr(
                f"{case_id}_report.json",
                json.dumps({"report_status": status, "report": exported, "review": entry["review"]}, indent=2),
            )
            zf.writestr(f"{case_id}_report.md", report_to_markdown(exported, entry["review"], status == "stale"))
        research = st.session_state.get("research", {}).get(case_id)
        if research is not None:
            from bus_research.feature_store import dumps_strict

            zf.writestr(f"{case_id}_radiomics_record.json", dumps_strict(research["record"]))
            if research.get("report") is not None:
                from bus_research.report import report_to_markdown as research_report_md

                zf.writestr(f"{case_id}_research_report.json", dumps_strict(research["report"]))
                zf.writestr(f"{case_id}_research_report.md", research_report_md(research["report"]))
        for name, array in (
            ("original", state["image"]),
            ("roi_crop", state["roi_crop"]),
            ("masked_roi", state["masked_roi"]),
        ):
            img_buffer = io.BytesIO()
            Image.fromarray(array.astype(np.uint8)).save(img_buffer, format="PNG")
            zf.writestr(f"{case_id}_{name}.png", img_buffer.getvalue())

    buffer.seek(0)
    return buffer.getvalue()


def render_case_preview_only(case_id: str, image: np.ndarray) -> None:
    """Shown in place of the AI cascade when the pipeline couldn't be loaded —
    still lets a physician confirm the right image came through and view it."""
    st.subheader(f"Case: {case_id}")
    st.image(image, caption="Uploaded image", use_container_width=True)
    st.caption(
        "Automated detection, segmentation, and classification are unavailable "
        "while the system is under maintenance."
    )


def build_batch_summary_csv() -> bytes:
    """Combined CSV summary across every processed case in this session."""
    import csv

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    # "confidence" is the classifier's; the Stage 6 report's is "report_confidence".
    writer.writerow(
        ["case_id", "bbox_source", "mask_source", "predicted_class", "confidence",
         "ai_category", "final_category", "changed", "report_confidence", "report_status"]
    )
    for case_id, state in st.session_state.get("case_states", {}).items():
        if state.get("ui_stage") != "report":
            continue
        entry = _report_entry(case_id)
        result = entry["result"] if entry else {}
        review = (entry or {}).get("review") or {}
        ai_category = result.get("category") if result.get("status") == "ok" and _category_shown(case_id) else None
        final_category = review.get("final_category")
        writer.writerow(
            [case_id, state["bbox_source"], state["mask_source"], state["predicted_class"], state["confidence"],
             ai_category or "", final_category or "",
             bool(final_category and ai_category and final_category != ai_category),
             result.get("confidence") or "", report_status(case_id, state)]
        )
    return buffer.getvalue().encode()


# --------------------------------------------------------------------------- #
# App layout
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Breast CAD — Clinician Review", layout="wide")
    apply_theme()
    st.title("Breast Ultrasound CAD — Clinician-in-the-Loop Review")

    pipeline: CADPipeline | None = None
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as exc:
        # Physician-facing message stays non-technical; the actual missing-checkpoint
        # detail (for whoever operates this app — run `python train.py all`) goes to
        # the log only, not the screen. Non-blocking: upload/preview still work below,
        # only the AI cascade itself is unavailable.
        logger.warning("Checkpoints unavailable, showing maintenance notice: %s", exc)
        st.info(
            "🛠️ This system is currently under maintenance. We'll be back in a few minutes. "
            "You can still upload and preview images below."
        )

    uploaded_files = st.sidebar.file_uploader(
        "Upload ultrasound image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Upload one or more ultrasound images from the sidebar to begin.")
        return

    case_names = [f.name for f in uploaded_files]
    selected_name = st.sidebar.selectbox(
        f"Case ({len(case_names)} uploaded)", case_names, key="selected_case"
    )
    st.sidebar.progress(
        (case_names.index(selected_name) + 1) / len(case_names), text="Batch progress"
    )

    selected_file = next(f for f in uploaded_files if f.name == selected_name)
    image = _load_uploaded_image(selected_file)

    if pipeline is not None:
        output_dir = Path("results/inference")
        render_case(pipeline, selected_name, image, output_dir)
    else:
        render_case_preview_only(selected_name, image)

    if len(uploaded_files) > 1:
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            "Download batch summary (.csv)",
            data=build_batch_summary_csv(),
            file_name="batch_summary.csv",
            mime="text/csv",
        )


def _load_uploaded_image(uploaded_file) -> np.ndarray:
    from PIL import Image

    uploaded_file.seek(0)
    pil_image = Image.open(uploaded_file).convert("RGB")
    return np.array(pil_image)


if __name__ == "__main__":
    main()
