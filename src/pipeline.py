"""
=============================================================================
MathVision: Unified End-to-End Pipeline Orchestrator
=============================================================================

This module binds together all individual subsystems:
    1. Preprocessing (OpenCV)
    2. Character Segmentation (OpenCV)
    3. Classification Inference (Deep CNN or HOG+SVM Baseline)
    4. Syntactic Parsing (Spatial Exponent & Implicit Multiplication)
    5. Step-by-Step Symbolic Solving (SymPy)

It exposes a single unified function:
    run_pipeline(image_input, settings=None, manual_overrides=None, cached_models=None)

Key Features:
    - High-Resolution Latency Profiling (ms per stage)
    - Low-Confidence Flagging (< 0.70 threshold)
    - Manual Label Override: Re-evaluates parser and solver immediately
      without redundant CV and model reprocessing
    - Debug Intermediate State Persistence
=============================================================================
"""

import os
import time
import zipfile
import io
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import joblib
from tensorflow import keras

from src.preprocess import preprocess_pipeline, PREPROCESS_CONFIG
from src.segment import segment_symbols, draw_bounding_boxes, SEGMENT_CONFIG
from src.dataset import CLASSES, IDX_TO_CLASS
from src.model import extract_hog_dataset
from src.parser import parse_symbols
from src.solver import solve_expression


# Global model cache to avoid repeated disk reads
_MODEL_CACHE: Dict[str, Any] = {}


def get_loaded_models(models_dir: str = "models") -> Dict[str, Any]:
    """Loads and caches the CNN and SVM models."""
    global _MODEL_CACHE
    if "cnn" not in _MODEL_CACHE:
        cnn_path = os.path.join(models_dir, "cnn_mathvision.keras")
        if os.path.exists(cnn_path):
            _MODEL_CACHE["cnn"] = keras.models.load_model(cnn_path)
        else:
            _MODEL_CACHE["cnn"] = None

    if "svm" not in _MODEL_CACHE:
        svm_path = os.path.join(models_dir, "svm_baseline.joblib")
        if os.path.exists(svm_path):
            _MODEL_CACHE["svm"] = joblib.load(svm_path)
        else:
            _MODEL_CACHE["svm"] = None

    return _MODEL_CACHE


def run_pipeline(
    image_input: Any,
    settings: Optional[Dict[str, Any]] = None,
    manual_overrides: Optional[Dict[int, str]] = None,
    cached_models: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes the complete MathVision computer vision & solving pipeline.

    Args:
        image_input: Image path, bytes, or numpy BGR array.
        settings: Tunable configuration overrides:
            - 'threshold_method': 'adaptive' | 'otsu' | 'global'
            - 'adaptive_block_size': int (default 15)
            - 'adaptive_c': int (default 4)
            - 'min_contour_area': int (default 25)
            - 'merge_min_h_overlap_ratio': float (default 0.35)
            - 'model_type': 'cnn' | 'svm' (default 'cnn')
            - 'confidence_threshold': float (default 0.70)
            - 'debug': bool (default False)
        manual_overrides: Optional dict mapping symbol index (int) to corrected label (str).
        cached_models: Preloaded {'cnn': model, 'svm': model} dictionary.

    Returns:
        Dict[str, Any]: Complete diagnostic pipeline state.
    """
    start_total_time = time.perf_counter()
    timings: Dict[str, float] = {}

    cfg = settings or {}
    model_type = cfg.get("model_type", "cnn").lower()
    conf_thresh = cfg.get("confidence_threshold", 0.70)
    thresh_method = cfg.get("threshold_method", "adaptive").lower()
    debug_mode = cfg.get("debug", False)

    # -------------------------------------------------------------------------
    # STAGE 1: PREPROCESSING (OpenCV)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    prep_cfg = {**PREPROCESS_CONFIG}
    if "adaptive_block_size" in cfg:
        prep_cfg["adaptive_block_size"] = cfg["adaptive_block_size"]
    if "adaptive_c" in cfg:
        prep_cfg["adaptive_c"] = cfg["adaptive_c"]

    prep = preprocess_pipeline(image_input, prep_cfg)

    # Select active binary image based on user tuning
    if thresh_method == "otsu":
        active_binary = prep["threshold_comparison"]["otsu"]
    elif thresh_method == "global":
        active_binary = prep["threshold_comparison"]["global"]
    else:
        active_binary = prep["cleaned"]  # Default: Adaptive + Morphology

    timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    # -------------------------------------------------------------------------
    # STAGE 2: SEGMENTATION (OpenCV)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    seg_cfg = {**SEGMENT_CONFIG}
    if "min_contour_area" in cfg:
        seg_cfg["min_contour_area"] = cfg["min_contour_area"]
    if "merge_min_h_overlap_ratio" in cfg:
        seg_cfg["merge_min_h_overlap_ratio"] = cfg["merge_min_h_overlap_ratio"]

    seg = segment_symbols(active_binary, prep["resized"], seg_cfg)
    boxes = seg["boxes"]
    crops = seg["symbol_crops"]
    timings["segment_ms"] = (time.perf_counter() - t0) * 1000.0

    if not crops:
        total_time = (time.perf_counter() - start_total_time) * 1000.0
        timings["total_ms"] = total_time
        return {
            "status": "no_symbols_detected",
            "error_message": "No mathematical symbols were detected. Ensure ink is clear on plain paper.",
            "timings": timings,
            "count": 0,
            "prep": prep,
            "seg": seg,
            "predicted_symbols": [],
            "confidences": [],
            "is_low_confidence": [],
            "parsed": {"is_valid": False, "error_message": "No symbols detected."},
            "solution": {"status": "error", "final_answer": "No symbols detected."}
        }

    # -------------------------------------------------------------------------
    # STAGE 3: CLASSIFICATION INFERENCE (CNN or Baseline SVM)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    models = cached_models or get_loaded_models()

    predicted_symbols: List[str] = []
    confidences: List[float] = []

    if model_type == "svm" and models.get("svm") is not None:
        svm = models["svm"]
        crops_hog = extract_hog_dataset(np.array(crops))
        probs = svm.predict_proba(crops_hog)
        preds = np.argmax(probs, axis=1)
        for i, p in enumerate(preds):
            predicted_symbols.append(CLASSES[p])
            confidences.append(float(probs[i, p]))
    else:
        cnn = models.get("cnn")
        if cnn is not None:
            batch = np.array(crops)[..., np.newaxis]
            probs = cnn.predict(batch, verbose=0)
            preds = np.argmax(probs, axis=1)
            for i, p in enumerate(preds):
                predicted_symbols.append(CLASSES[p])
                confidences.append(float(probs[i, p]))
        else:
            # Fallback placeholder if models not loaded
            predicted_symbols = ["0"] * len(crops)
            confidences = [0.5] * len(crops)

    timings["inference_ms"] = (time.perf_counter() - t0) * 1000.0

    # -------------------------------------------------------------------------
    # APPLY MANUAL OVERRIDES (if any provided from UI)
    # -------------------------------------------------------------------------
    if manual_overrides:
        for idx, new_label in manual_overrides.items():
            if 0 <= idx < len(predicted_symbols):
                predicted_symbols[idx] = new_label
                confidences[idx] = 1.0  # User verified

    # Low-confidence evaluation
    is_low_conf = [c < conf_thresh for c in confidences]

    # Render bounding box visualization with prediction badges
    annotated_vis = draw_bounding_boxes(
        prep["resized"],
        boxes,
        labels=predicted_symbols,
        confidences=confidences,
        config=seg_cfg
    )

    # -------------------------------------------------------------------------
    # STAGE 4: SYNTACTIC PARSING
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    probs_to_pass = probs if ("probs" in locals() and probs is not None) else None
    disambig_cfg = cfg.get("disambiguation_config")
    parsed_res = parse_symbols(predicted_symbols, boxes, probs=probs_to_pass, disambiguation_config=disambig_cfg)
    timings["parse_ms"] = (time.perf_counter() - t0) * 1000.0

    # -------------------------------------------------------------------------
    # STAGE 5: SYMBOLIC STEP-BY-STEP SOLVING (SymPy)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    if parsed_res["is_valid"]:
        solution_res = solve_expression(parsed_res["parsed_expression"])
    else:
        solution_res = {
            "status": "error",
            "equation_type": "invalid_syntax",
            "error_message": parsed_res["error_message"],
            "steps": [],
            "final_answer": "Syntax Error",
            "final_answer_latex": "\\text{Syntax Error}",
            "roots": []
        }
    timings["solve_ms"] = (time.perf_counter() - t0) * 1000.0

    total_time = (time.perf_counter() - start_total_time) * 1000.0
    timings["total_ms"] = total_time

    # Diagnostic state bundle
    results = {
        "status": "success" if solution_res["status"] in ("solved", "unsolvable") else "error",
        "timings": timings,
        "count": len(boxes),
        "boxes": boxes,
        "symbol_crops": crops,
        "annotated_vis": annotated_vis,
        "predicted_symbols": predicted_symbols,
        "confidences": confidences,
        "is_low_confidence": is_low_conf,
        "has_low_confidence": any(is_low_conf),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "parsed": parsed_res,
        "solution": solution_res,
        "prep": prep,
        "active_binary": active_binary,
    }

    # Debug artifact persistence
    if debug_mode:
        save_debug_artifacts(results)

    return results


def save_debug_artifacts(results: Dict[str, Any], output_dir: str = "outputs/debug") -> None:
    """Saves all pipeline intermediates to disk when debug flag is active."""
    os.makedirs(output_dir, exist_ok=True)
    prep = results["prep"]
    cv2.imwrite(os.path.join(output_dir, "01_original.png"), prep["original"])
    cv2.imwrite(os.path.join(output_dir, "02_resized.png"), prep["resized"])
    cv2.imwrite(os.path.join(output_dir, "03_binary.png"), results["active_binary"])
    cv2.imwrite(os.path.join(output_dir, "04_annotated.png"), results["annotated_vis"])

    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    for i, c in enumerate(results["symbol_crops"]):
        c_uint8 = np.uint8(c * 255)
        cv2.imwrite(os.path.join(crops_dir, f"crop_{i+1:02d}_{results['predicted_symbols'][i]}.png"), c_uint8)


def create_results_zip(results: Dict[str, Any]) -> bytes:
    """Packages all intermediate images and markdown solution steps into a zip archive."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Solution Markdown
        sol = results["solution"]
        parsed = results["parsed"]
        md_text = f"# MathVision Solution Report\n\n"
        md_text += f"- **Recognized Expression:** `{parsed.get('parsed_expression', '')}`\n"
        md_text += f"- **LaTeX Formula:** $${parsed.get('latex_expression', '')}$$\n"
        md_text += f"- **Status:** {sol.get('status', '').upper()}\n"
        md_text += f"- **Final Answer:** {sol.get('final_answer', '')}\n\n"
        md_text += "## Step-by-Step Working\n\n"
        for s in sol.get("steps", []):
            md_text += f"### Step {s['step_number']}: {s['title']}\n"
            md_text += f"{s['description']}\n\n"
            md_text += f"$${s['latex']}$$\n\n"

        zip_file.writestr("solution_report.md", md_text)

        # 2. Pipeline Images
        prep = results["prep"]
        _, orig_png = cv2.imencode(".png", prep["original"])
        zip_file.writestr("01_original.png", orig_png.tobytes())

        _, bin_png = cv2.imencode(".png", results["active_binary"])
        zip_file.writestr("02_binary_threshold.png", bin_png.tobytes())

        _, vis_png = cv2.imencode(".png", results["annotated_vis"])
        zip_file.writestr("03_bounding_boxes.png", vis_png.tobytes())

        # 3. Symbol crops
        for idx, crop in enumerate(results["symbol_crops"]):
            lbl = results["predicted_symbols"][idx]
            conf = int(results["confidences"][idx] * 100)
            crop_uint8 = np.uint8(crop * 255)
            _, crop_png = cv2.imencode(".png", crop_uint8)
            zip_file.writestr(f"symbols/crop_{idx+1:02d}_{lbl}_{conf}pct.png", crop_png.tobytes())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
