"""Stage 5: clinician-in-the-loop Streamlit demo.

Loads the three trained checkpoints once (``st.cache_resource``) and walks a
physician through detect -> review/correct -> segment -> review/correct ->
classify for one or many uploaded ultrasound images, logging every
accept/reject/manual-correction decision for later audit.

All cascade logic (detect/crop/segment/mask/classify) is imported from
``inference.pipeline`` — nothing here duplicates that chaining logic.
"""

from __future__ import annotations

import io
import json
import os
import sys
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

from inference.pipeline import CADPipeline, apply_mask, to_chw_tensor  # noqa: E402
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

        bundle = build_case_bundle(case_id, state)
        st.download_button(
            "Download case report (JSON + images, .zip)",
            data=bundle,
            file_name=f"{case_id}_report.zip",
            mime="application/zip",
            key=f"download_{case_id}",
        )


def build_case_bundle(case_id: str, state: dict) -> bytes:
    """Zip: result JSON + original/ROI/masked-ROI PNGs — the per-case download."""
    from PIL import Image

    summary = {
        "case_id": case_id,
        "bbox": list(state["bbox"]),
        "bbox_source": state["bbox_source"],
        "mask_source": state["mask_source"],
        "predicted_class": state["predicted_class"],
        "confidence": state["confidence"],
        "class_probs": state["class_probs"],
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(f"{case_id}_result.json", json.dumps(summary, indent=2))
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
    writer.writerow(
        ["case_id", "bbox_source", "mask_source", "predicted_class", "confidence"]
    )
    for case_id, state in st.session_state.get("case_states", {}).items():
        if state.get("ui_stage") != "report":
            continue
        writer.writerow(
            [case_id, state["bbox_source"], state["mask_source"], state["predicted_class"], state["confidence"]]
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
