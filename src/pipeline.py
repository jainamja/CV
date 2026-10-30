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
import json
import itertools
from datetime import datetime
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


def check_division_topology(
    crop: np.ndarray,
    min_bar_span_ratio: float = 0.45,
    min_bar_span_px: int = 12,
    min_gap_rows: int = 2,
    intensity_threshold: float = 0.20
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Analyzes actual foreground pixel distribution in a 45x45 character crop
    to determine whether it exhibits the physical topology of an orthogonal
    division symbol ('div' / '÷') as opposed to an intersecting plus sign ('+').

    Morphological signature of orthogonal division:
      1. A prominent, centrally located horizontal bar spanning >= 45% of character width.
      2. At least two distinct connected components separated by one or more empty horizontal rows.
      3. At least one significant vertical gap (>= min_gap_rows consecutive zero-intensity rows)
         separating the horizontal bar from an upper or lower stroke component.
    """
    metrics: Dict[str, Any] = {
        "is_division": False,
        "reason": "uninitialized",
        "bar_y": None,
        "bar_span": 0,
        "char_width": 0,
        "char_height": 0,
        "zeros_above": 0,
        "zeros_below": 0,
        "num_components": 0
    }

    if crop is None or crop.size == 0:
        metrics["reason"] = "crop is empty or None"
        return False, metrics["reason"], metrics

    norm = crop.astype(float) if crop.max() <= 1.0 else crop.astype(float) / 255.0
    mask = (norm > intensity_threshold).astype(np.uint8)
    pts = np.argwhere(mask > 0)

    if len(pts) < 10:
        metrics["reason"] = "insufficient foreground pixels (<10)"
        return False, metrics["reason"], metrics

    min_y, max_y = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
    min_x, max_x = int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))
    h = max_y - min_y + 1
    w = max_x - min_x + 1
    metrics["char_width"] = w
    metrics["char_height"] = h

    if h < 12 or w < 12:
        metrics["reason"] = f"bounding extent too small ({w}x{h} < 12x12)"
        return False, metrics["reason"], metrics

    row_counts = [int(np.sum(mask[y, :])) for y in range(min_y, max_y + 1)]
    max_idx = int(np.argmax(row_counts))
    bar_y = min_y + max_idx
    metrics["bar_y"] = bar_y

    bar_pts = np.where(mask[bar_y, :] > 0)[0]
    bar_span = int(np.max(bar_pts) - np.min(bar_pts) + 1) if len(bar_pts) > 0 else 0
    metrics["bar_span"] = bar_span

    if bar_span < (min_bar_span_ratio * w) or bar_span < min_bar_span_px:
        metrics["reason"] = f"horizontal bar span ({bar_span}px) too narrow relative to width ({w}px)"
        return False, metrics["reason"], metrics

    rel_bar_y = (bar_y - min_y) / float(h)
    if rel_bar_y < 0.15 or rel_bar_y > 0.85:
        metrics["reason"] = f"horizontal bar is not vertically centered (rel_y={rel_bar_y:.2f})"
        return False, metrics["reason"], metrics

    upper_rows = row_counts[:max_idx]
    lower_rows = row_counts[max_idx + 1:]

    def get_max_consecutive_zeros(rows: List[int]) -> int:
        if not rows:
            return 0
        return max([sum(1 for _ in g) for k, g in itertools.groupby(rows, lambda x: x == 0) if k], default=0)

    zeros_above = get_max_consecutive_zeros(upper_rows)
    zeros_below = get_max_consecutive_zeros(lower_rows)
    metrics["zeros_above"] = zeros_above
    metrics["zeros_below"] = zeros_below

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_fg_components = num_labels - 1
    metrics["num_components"] = num_fg_components

    upper_mass = sum(upper_rows)
    lower_mass = sum(lower_rows)

    has_both_gaps = (zeros_above >= 1 and zeros_below >= 1)
    has_asymmetric_gap = (zeros_above >= min_gap_rows and upper_mass >= 3) or (zeros_below >= min_gap_rows and lower_mass >= 3)

    if (has_both_gaps or has_asymmetric_gap) and (num_fg_components >= 2):
        metrics["is_division"] = True
        reason = (
            f"Division topology confirmed: bar_span={bar_span}px ({bar_span/w*100:.1f}% w), "
            f"zeros_above={zeros_above}, zeros_below={zeros_below}, components={num_fg_components}."
        )
        metrics["reason"] = reason
        return True, reason, metrics

    metrics["reason"] = f"Continuous vertical stroke: zeros_above={zeros_above}, zeros_below={zeros_below}, comps={num_fg_components}"
    return False, metrics["reason"], metrics


def _run_pipeline_single(
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
    thresh_method = cfg.get("threshold_method", PREPROCESS_CONFIG.get("threshold_method", "adaptive")).lower()
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
    prep_cfg["threshold_method"] = thresh_method

    prep = preprocess_pipeline(image_input, prep_cfg)

    # Select active binary image based on user tuning
    cleaned_comp = prep.get("threshold_comparison_cleaned", {})
    raw_comp = prep.get("threshold_comparison", {})

    if thresh_method == "otsu":
        active_binary = raw_comp.get("otsu", prep["binary"])
    elif thresh_method == "global":
        active_binary = raw_comp.get("global", prep["binary"])
    elif thresh_method == "adaptive":
        active_binary = cleaned_comp.get("adaptive", raw_comp.get("adaptive", prep["cleaned"]))
    else:
        active_binary = prep["cleaned"]

    # Automatic fallback if selected binary produces empty image or severe binary inversion
    fg_ratio = cv2.countNonZero(active_binary) / float(active_binary.size) if active_binary.size > 0 else 0.0
    if cv2.countNonZero(active_binary) == 0 or fg_ratio > 0.35:
        adaptive_candidate = cleaned_comp.get("adaptive", raw_comp.get("adaptive", prep.get("cleaned")))
        if adaptive_candidate is not None and cv2.countNonZero(adaptive_candidate) > 0:
            active_binary = adaptive_candidate

    timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    # -------------------------------------------------------------------------
    # STAGE 2: SEGMENTATION (OpenCV)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    seg_cfg = {**SEGMENT_CONFIG}
    if "min_contour_area" in cfg:
        seg_cfg["min_contour_area"] = cfg["min_contour_area"]
    if "min_stroke_pixels" in cfg:
        seg_cfg["min_stroke_pixels"] = cfg["min_stroke_pixels"]
    if "merge_min_h_overlap_ratio" in cfg:
        seg_cfg["merge_min_h_overlap_ratio"] = cfg["merge_min_h_overlap_ratio"]
    if "padding_margin_ratio" in cfg:
        seg_cfg["padding_margin_ratio"] = cfg["padding_margin_ratio"]

    seg = segment_symbols(active_binary, prep["resized"], seg_cfg)
    boxes = seg["boxes"]
    crops = seg["symbol_crops"]

    sanity = seg.get("sanity", {})
    # If selected threshold under-segments or fails sanity while adaptive passes sanity, fallback
    if (len(crops) < 4 or not sanity.get("is_valid", True)) and thresh_method != "adaptive":
        adaptive_candidate = cleaned_comp.get("adaptive", raw_comp.get("adaptive", prep.get("cleaned")))
        if adaptive_candidate is not None:
            seg_alt = segment_symbols(adaptive_candidate, prep["resized"], seg_cfg)
            if len(seg_alt["symbol_crops"]) >= 4 or seg_alt.get("sanity", {}).get("is_valid", True):
                seg = seg_alt
                active_binary = adaptive_candidate
                boxes = seg["boxes"]
                crops = seg["symbol_crops"]
                sanity = seg.get("sanity", {})

    timings["segment_ms"] = (time.perf_counter() - t0) * 1000.0

    if not crops:
        total_time = (time.perf_counter() - start_total_time) * 1000.0
        timings["total_ms"] = total_time
        return {
            "status": "no_symbols_detected",
            "error_message": "No mathematical symbols were detected. Ensure ink is clear on plain paper.",
            "timings": timings,
            "count": 0,
            "boxes": [],
            "symbol_crops": [],
            "annotated_vis": prep["resized"],
            "prep": prep,
            "seg": seg,
            "active_binary": active_binary,
            "predicted_symbols": [],
            "confidences": [],
            "symbol_diagnostics": [],
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
    probs = None

    if model_type == "svm" and models.get("svm") is not None:
        svm = models["svm"]
        crops_hog = extract_hog_dataset(np.array(crops))
        try:
            probs = svm.predict_proba(crops_hog)
        except Exception:
            probs = None

        if probs is None:
            if hasattr(svm, "decision_function"):
                df = svm.decision_function(crops_hog)
                if df.ndim == 1:
                    df = np.column_stack([-df, df])
                exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                probs = exp_df / np.sum(exp_df, axis=1, keepdims=True)
            else:
                raw_p = svm.predict(crops_hog)
                probs = np.zeros((len(raw_p), len(CLASSES)), dtype=np.float32)
                for i, p in enumerate(raw_p):
                    probs[i, p] = 1.0

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

    # Collect per-symbol diagnostics
    symbol_diagnostics = []
    for i in range(len(predicted_symbols)):
        top3 = []
        if probs is not None and i < len(probs):
            top3_idx = np.argsort(probs[i])[::-1][:3]
            top3 = [(CLASSES[k], float(probs[i][k])) for k in top3_idx]
        symbol_diagnostics.append({
            "index": i,
            "bbox": boxes[i] if i < len(boxes) else None,
            "symbol": predicted_symbols[i],
            "confidence": float(confidences[i]) if i < len(confidences) else 0.0,
            "top3": top3,
            "crop_shape": crops[i].shape if i < len(crops) else None
        })

    # -------------------------------------------------------------------------
    # STAGE 3.5: GEOMETRIC STROKE TOPOLOGY DISAMBIGUATION (e.g. '2' vs '7' and '(' vs '1')
    # -------------------------------------------------------------------------
    idx_7 = CLASSES.index("7") if "7" in CLASSES else -1
    for i in range(len(predicted_symbols)):
        if predicted_symbols[i] == "2" and i < len(crops) and idx_7 >= 0:
            c = crops[i]
            active = np.where(np.any(c > 0.2, axis=1))[0]
            if len(active) > 0:
                h = active[-1] - active[0] + 1
                top_t = int(active[0] + 0.20 * h)
                t_pts = np.argwhere(c[active[0]:top_t + 1, :] > 0.2)
                t_s = int(np.max(t_pts[:, 1]) - np.min(t_pts[:, 1]) + 1) if len(t_pts) else 0
                bot_t = int(active[-1] - 0.20 * h)
                b_pts = np.argwhere(c[bot_t:active[-1] + 1, :] > 0.2)
                b_s = int(np.max(b_pts[:, 1]) - np.min(b_pts[:, 1]) + 1) if len(b_pts) else 0
                ratio = b_s / float(max(1, t_s))
                # A true '2' requires a bottom horizontal base (empirical p10=21px, min=8px).
                # A true '7' has a wide top bar (span >= 16px) and terminates at a single pointed
                # bottom endpoint (empirical mean=4.2px, p90=7px) with no horizontal base.
                if b_s < 10 and t_s >= 16 and ratio <= 0.45:
                    predicted_symbols[i] = "7"
                    if probs is not None and i < len(probs):
                        confidences[i] = float(probs[i, idx_7])
                    symbol_diagnostics[i]["symbol"] = "7"
                    symbol_diagnostics[i]["confidence"] = confidences[i]
                    symbol_diagnostics[i]["disambiguation_note"] = (
                        "Disambiguated '2' -> '7': stroke topology exhibits top horizontal bar "
                        f"(span={t_s}px) and pointed bottom endpoint (span={b_s}px, ratio={ratio:.2f}) lacking a horizontal base."
                    )

    # Disambiguation for '(' vs '1'
    idx_1 = CLASSES.index("1") if "1" in CLASSES else -1
    for i in range(len(predicted_symbols)):
        if predicted_symbols[i] == "(" and i < len(crops) and idx_1 >= 0:
            top3 = symbol_diagnostics[i].get("top3", [])
            prob_1 = next((prob for sym, prob in top3 if sym == "1"), 0.0)
            bw = boxes[i][2] if i < len(boxes) else 10
            # Check if adjacent to next digit (forming a 2-digit number like 12)
            has_adjacent_digit = False
            if i + 1 < len(boxes) and i + 1 < len(predicted_symbols):
                if str(predicted_symbols[i+1]).isdigit():
                    gap = boxes[i+1][0] - (boxes[i][0] + boxes[i][2])
                    if gap <= 8:
                        has_adjacent_digit = True
            no_closing_paren = (predicted_symbols.count(")") == 0)
            # Only convert '(' to '1' if there is NO matching closing parenthesis and it sits next to a digit
            if no_closing_paren and has_adjacent_digit and (prob_1 >= 0.20 or bw <= 5):
                predicted_symbols[i] = "1"
                if probs is not None and i < len(probs):
                    confidences[i] = float(probs[i, idx_1])
                symbol_diagnostics[i]["symbol"] = "1"
                symbol_diagnostics[i]["confidence"] = confidences[i]
                symbol_diagnostics[i]["disambiguation_note"] = (
                    f"Disambiguated '(' -> '1': narrow vertical stroke (w={bw}px, prob_1={prob_1:.2f}) "
                    f"adjacent to digit with no matching closing parenthesis."
                )

    # Disambiguation for ')' vs '1'
    for i in range(len(predicted_symbols)):
        if predicted_symbols[i] == ")" and i < len(crops) and idx_1 >= 0:
            no_opening_paren = (predicted_symbols.count("(") == 0)
            bw = boxes[i][2] if i < len(boxes) else 10
            has_adjacent_digit = False
            if i + 1 < len(boxes) and i + 1 < len(predicted_symbols):
                if str(predicted_symbols[i+1]).isdigit():
                    gap = boxes[i+1][0] - (boxes[i][0] + boxes[i][2])
                    if gap <= 8:
                        has_adjacent_digit = True
            top3 = symbol_diagnostics[i].get("top3", [])
            prob_1 = next((prob for sym, prob in top3 if sym == "1"), 0.0)
            if no_opening_paren and has_adjacent_digit and (prob_1 >= 0.20 or bw <= 5):
                predicted_symbols[i] = "1"
                if probs is not None and i < len(probs):
                    confidences[i] = float(probs[i, idx_1])
                symbol_diagnostics[i]["symbol"] = "1"
                symbol_diagnostics[i]["confidence"] = confidences[i]
                symbol_diagnostics[i]["disambiguation_note"] = (
                    f"Disambiguated ')' -> '1': narrow vertical stroke (w={bw}px, prob_1={prob_1:.2f}) "
                    f"adjacent to digit with no matching opening parenthesis."
                )

    # Disambiguation for '(' vs '6' (detect enclosed bottom bowl)
    idx_6 = CLASSES.index("6") if "6" in CLASSES else -1
    for i in range(len(predicted_symbols)):
        if predicted_symbols[i] == "(" and i < len(crops) and idx_6 >= 0:
            c = crops[i]
            pts = np.argwhere(c > 0.2)
            if len(pts) > 0:
                ys = pts[:, 0]
                min_y, max_y = np.min(ys), np.max(ys)
                h_c = max_y - min_y + 1
                bot_y = int(min_y + 0.80 * h_c)
                row_pts = np.where(c[bot_y, :] > 0.2)[0]
                bot_gap = np.max(row_pts) - np.min(row_pts) - len(row_pts) + 1 if len(row_pts) >= 2 else 0
                top3 = symbol_diagnostics[i].get("top3", [])
                prob_6 = next((prob for sym, prob in top3 if sym == "6"), 0.0)
                is_before_closing = (i + 1 < len(predicted_symbols) and predicted_symbols[i+1] == ")")
                if (bot_gap >= 5 and prob_6 >= 0.10) or (is_before_closing and prob_6 >= 0.15):
                    predicted_symbols[i] = "6"
                    if probs is not None and i < len(probs):
                        confidences[i] = float(probs[i, idx_6])
                    symbol_diagnostics[i]["symbol"] = "6"
                    symbol_diagnostics[i]["confidence"] = confidences[i]
                    symbol_diagnostics[i]["disambiguation_note"] = (
                        f"Disambiguated '(' -> '6': stroke topology exhibits enclosed bottom bowl "
                        f"(gap={bot_gap}px, prob_6={prob_6:.2f})."
                    )

    # Disambiguation for '6' vs '8' (detect enclosed upper loop)
    idx_8 = CLASSES.index("8") if "8" in CLASSES else -1
    for i in range(len(predicted_symbols)):
        if predicted_symbols[i] == "6" and i < len(crops) and idx_8 >= 0:
            c = crops[i]
            pts = np.argwhere(c > 0.2)
            if len(pts) > 0:
                ys = pts[:, 0]
                min_y, max_y = np.min(ys), np.max(ys)
                h_c = max_y - min_y + 1
                # Check row in upper third (25% from top)
                top_y = int(min_y + 0.25 * h_c)
                row_pts = np.where(c[top_y, :] > 0.2)[0]
                top_inner_gap = np.max(row_pts) - np.min(row_pts) - len(row_pts) + 1 if len(row_pts) >= 2 else 0
                top3 = symbol_diagnostics[i].get("top3", [])
                prob_8 = next((prob for sym, prob in top3 if sym == "8"), 0.0)
                if top_inner_gap >= 6:
                    predicted_symbols[i] = "8"
                    if probs is not None and i < len(probs):
                        confidences[i] = float(probs[i, idx_8])
                    symbol_diagnostics[i]["symbol"] = "8"
                    symbol_diagnostics[i]["confidence"] = confidences[i]
                    symbol_diagnostics[i]["disambiguation_note"] = (
                        f"Disambiguated '6' -> '8': stroke topology exhibits upper enclosed loop "
                        f"(inner_gap={top_inner_gap}px, prob_8={prob_8:.2f})."
                    )

    # Disambiguation for Division Slash ('/') vs Digit '1'
    idx_div = CLASSES.index("div") if "div" in CLASSES else -1
    for i in range(len(predicted_symbols)):
        curr_sym = predicted_symbols[i]
        if curr_sym in ("1", "div", "(", ")") and i < len(crops) and idx_div >= 0 and idx_1 >= 0:
            top3 = symbol_diagnostics[i].get("top3", [])
            prob_div = next((prob for sym, prob in top3 if sym in ("div", "/")), 0.0)
            prob_1 = next((prob for sym, prob in top3 if sym == "1"), 0.0)

            # Check if this token is a candidate division slash competing with '1'
            if prob_div >= 0.15 or (curr_sym in ("1", "(") and prob_div >= 0.10):
                c = crops[i]
                pts = np.argwhere(c > 0.2)
                if len(pts) >= 10:
                    ys, xs = pts[:, 0].astype(float), pts[:, 1].astype(float)
                    mid_y = float(np.median(ys))
                    top_half = pts[ys < mid_y]
                    bot_half = pts[ys >= mid_y]

                    if len(top_half) > 0 and len(bot_half) > 0:
                        top_cx = float(np.mean(top_half[:, 1]))
                        bot_cx = float(np.mean(bot_half[:, 1]))
                        top_cy = float(np.mean(top_half[:, 0]))
                        bot_cy = float(np.mean(bot_half[:, 0]))
                        slant_dx = top_cx - bot_cx
                        slant_dy = bot_cy - top_cy
                        angle_deg = float(np.degrees(np.arctan2(slant_dx, max(1.0, slant_dy))))

                        bx, by, bw, bh = boxes[i] if i < len(boxes) else (0, 0, 10, 10)

                        prev_sym = predicted_symbols[i - 1] if i > 0 else None
                        next_sym = predicted_symbols[i + 1] if i + 1 < len(predicted_symbols) else None

                        is_after_close_paren = (prev_sym == ")")
                        is_between_operands = (
                            (prev_sym is not None and (str(prev_sym).isdigit() or prev_sym == ")")) and
                            (next_sym is not None and (str(next_sym).isdigit() or next_sym == "("))
                        )

                        # A forward division slash '/' runs diagonally from bottom-left to top-right:
                        # slant_dx >= 3.5px and angle_deg >= 12.0 degrees
                        # Spans full character height (bh >= 20px) and width (bw >= 8px)
                        is_forward_slash = (
                            slant_dx >= 3.5 and
                            angle_deg >= 12.0 and
                            bh >= 20 and
                            bw >= 8
                        )

                        if is_forward_slash and (is_after_close_paren or is_between_operands):
                            if curr_sym != "div":
                                predicted_symbols[i] = "div"
                                if probs is not None and i < len(probs):
                                    confidences[i] = float(probs[i, idx_div])
                                symbol_diagnostics[i]["symbol"] = "div"
                                symbol_diagnostics[i]["confidence"] = confidences[i]
                                symbol_diagnostics[i]["disambiguation_note"] = (
                                    f"Disambiguated '{curr_sym}' -> 'div': forward diagonal slash stroke topology "
                                    f"(slant_dx={slant_dx:.2f}px, angle={angle_deg:.1f}°, w={bw}px, h={bh}px) "
                                    f"positioned between operands ({prev_sym} / {next_sym})."
                                )
                        elif is_after_close_paren and curr_sym == "1" and not is_forward_slash:
                            # Flag low-margin ambiguous classification where an uncertain '1' follows ')'
                            symbol_diagnostics[i]["is_ambiguous"] = True
                            symbol_diagnostics[i]["ambiguity_warning"] = (
                                f"Ambiguous token following ')': predicted as '1' (conf={prob_1:.2f}) "
                                f"with competing operator 'div' (conf={prob_div:.2f})."
                            )

    # Disambiguation for '3' vs '5' (detect single convex right bowl with open middle-left whitespace)
    idx_5 = CLASSES.index("5") if "5" in CLASSES else -1
    for i in range(len(predicted_symbols)):
        if predicted_symbols[i] == "3" and i < len(crops) and idx_5 >= 0:
            top3 = symbol_diagnostics[i].get("top3", [])
            prob_3 = confidences[i]
            prob_5 = next((prob for sym, prob in top3 if sym == "5"), 0.0)

            # Check if this token is a low/moderate confidence '3' competing with '5'
            if prob_5 >= 0.15 and prob_3 < 0.85:
                c = crops[i]
                pts = np.argwhere(c > 0.2)
                if len(pts) >= 10:
                    y_min, x_min = np.min(pts[:, 0]), np.min(pts[:, 1])
                    y_max, x_max = np.max(pts[:, 0]), np.max(pts[:, 1])
                    h_c = y_max - y_min + 1
                    w_c = x_max - x_min + 1

                    if h_c >= 15 and w_c >= 8:
                        # Inspect lower-middle vertical section (55% to 75% of height, below the neck):
                        # In a handwritten '5', the stroke loops around the right, so the leftmost column
                        # in this section is on the far right (>= 60% of width).
                        # In a handwritten '3', the middle cusp points leftward, so the leftmost column
                        # is near the left edge (<= 35% of width).
                        y_sub1 = int(y_min + 0.55 * h_c)
                        y_sub2 = int(y_min + 0.75 * h_c)
                        sub_pts = np.argwhere(c[y_sub1:y_sub2+1, :] > 0.2)
                        if len(sub_pts) > 0:
                            sub_left_col = np.min(sub_pts[:, 1]) - x_min
                            sub_left_ratio = float(sub_left_col) / float(w_c)

                            if sub_left_ratio >= 0.60:
                                predicted_symbols[i] = "5"
                                if probs is not None and i < len(probs):
                                    confidences[i] = float(probs[i, idx_5])
                                symbol_diagnostics[i]["symbol"] = "5"
                                symbol_diagnostics[i]["confidence"] = confidences[i]
                                symbol_diagnostics[i]["disambiguation_note"] = (
                                    f"Disambiguated '3' -> '5': stroke topology exhibits open middle-left whitespace "
                                    f"with lower-middle stroke on far right (left_ratio={sub_left_ratio:.2f}, prob_5={prob_5:.2f}) "
                                    f"characteristic of digit '5' bowl rather than digit '3' central cusp."
                                )

    # Disambiguation for Orthogonal Division ('div' / '÷') vs Plus ('+')
    # Resolves ambiguity when CNN predicts '+' with low confidence (<0.65) and competing 'div' (>=0.15),
    # verifying presence of a prominent horizontal bar and detached upper/lower dots with empty gap rows.
    for i in range(len(predicted_symbols)):
        if predicted_symbols[i] == "+" and i < len(crops) and idx_div >= 0:
            top3 = symbol_diagnostics[i].get("top3", [])
            prob_plus = confidences[i]
            prob_div = next((prob for sym, prob in top3 if sym in ("div", "/")), 0.0)

            # Two-layer defense: Layer 1 Neural Gating
            if prob_plus < 0.65 and prob_div >= 0.15:
                crop = crops[i]
                # Layer 2 Pixel Morphology
                is_div, reason, metrics = check_division_topology(crop)
                if is_div:
                    predicted_symbols[i] = "div"
                    if probs is not None and i < len(probs):
                        confidences[i] = float(probs[i, idx_div])
                    else:
                        confidences[i] = float(prob_div)
                    symbol_diagnostics[i]["symbol"] = "div"
                    symbol_diagnostics[i]["confidence"] = confidences[i]
                    symbol_diagnostics[i]["disambiguation_note"] = (
                        f"Disambiguated '+' -> 'div': {reason} "
                        f"(prob_+={prob_plus:.3f}, prob_div={prob_div:.3f})"
                    )
                    symbol_diagnostics[i]["topological_metrics"] = metrics

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
        "status": "success" if (solution_res["status"] in ("solved", "unsolvable", "simplified") and sanity.get("quality") != "failed") else "error",
        "timings": timings,
        "count": len(boxes),
        "boxes": boxes,
        "raw_boxes": seg.get("raw_boxes", boxes),
        "merged_boxes": seg.get("merged_boxes", boxes),
        "contour_diagnostics": seg.get("contour_diagnostics", []),
        "symbol_crops": crops,
        "annotated_vis": annotated_vis,
        "predicted_symbols": predicted_symbols,
        "confidences": confidences,
        "symbol_diagnostics": symbol_diagnostics,
        "is_low_confidence": is_low_conf,
        "has_low_confidence": any(is_low_conf),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "segmentation_quality": sanity.get("quality", "valid"),
        "segmentation_warnings": sanity.get("warnings", []),
        "seg_sanity": sanity,
        "parsed": parsed_res,
        "solution": solution_res,
        "expression_parsed": parsed_res.get("parsed_expression", ""),
        "expression_validated": parsed_res.get("is_valid", False),
        "answer_verified": False,
        "verification_note": "Answer is symbolically computed from recognized expression; independent ground truth verification only occurs in automated regression tests.",
        "prep": prep,
        "seg": seg,
        "active_binary": active_binary,
    }

    # Debug artifact persistence
    if debug_mode:
        save_debug_artifacts(results)

    return results


def score_pipeline_result(res: Dict[str, Any]) -> float:
    """
    Evaluates the quality, syntactic validity, and geometric plausibility of a pipeline result.
    Used for multi-threshold candidate arbitration (e.g. Adaptive vs Otsu).
    """
    if res.get("status") == "error":
        return -1000.0
    parsed = res.get("parsed", {})
    if not parsed.get("is_valid", False):
        return -1000.0
    sol = res.get("solution", {})
    if sol.get("status") not in ("solved", "simplified", "unsolvable"):
        return -1000.0
    if res.get("segmentation_quality") == "failed":
        return -1000.0

    score = 100.0
    if res.get("segmentation_quality") == "suspicious":
        score -= 40.0

    boxes = res.get("boxes", [])
    if boxes:
        heights = [b[3] for b in boxes]
        med_h = np.median(heights)
        # Heavy penalty for severely fragmented boxes (less than 45% of median height)
        tiny_count = sum(1 for h in heights if h < 0.45 * med_h)
        score -= tiny_count * 40.0
        # Bonus for height consistency (low coefficient of variation)
        if med_h > 0:
            cv_h = float(np.std(heights) / med_h)
            score += max(0.0, 15.0 * (1.0 - cv_h))

    # Mean confidence bonus
    score += res.get("mean_confidence", 0.0) * 15.0

    # Only penalize very low confidence (< 0.35)
    confs = res.get("confidences", [])
    for c in confs:
        if c < 0.35:
            score -= 20.0

    syms = res.get("predicted_symbols", [])
    # Strong bonus for recognizing clean division symbols
    if "div" in syms:
        score += 20.0

    # Balanced parentheses bonus / penalty
    p_open = syms.count("(")
    p_close = syms.count(")")
    if p_open != p_close:
        score -= 60.0
    elif p_open > 0 and p_open == p_close:
        score += 25.0

    # Consecutive operators like '+ +'
    for i in range(len(syms) - 1):
        if syms[i] in ("+", "-", "*", "/", "times", "div") and syms[i+1] in ("+", "-", "*", "/", "times", "div"):
            score -= 50.0

    return float(score)


def run_pipeline(
    image_input: Any,
    settings: Optional[Dict[str, Any]] = None,
    manual_overrides: Optional[Dict[int, str]] = None,
    cached_models: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes the complete MathVision computer vision & solving pipeline.
    Includes multi-threshold candidate arbitration between Adaptive and Otsu
    binarization to maximize accuracy across varying contrast conditions.
    """
    cfg = dict(settings or {})

    # If caller explicitly requested a specific threshold method, execute directly
    if "threshold_method" in cfg and cfg["threshold_method"] not in ("auto", None):
        return _run_pipeline_single(image_input, cfg, manual_overrides, cached_models)

    cfg_base = {k: v for k, v in cfg.items() if k != "threshold_method"}

    # Primary pass: adaptive thresholding (default)
    res_primary = _run_pipeline_single(image_input, cfg_base, manual_overrides, cached_models)
    score_primary = score_pipeline_result(res_primary)

    # Candidate evaluation pass: Otsu thresholding
    cfg_otsu = {**cfg_base, "threshold_method": "otsu"}
    res_otsu = _run_pipeline_single(image_input, cfg_otsu, manual_overrides, cached_models)
    score_otsu = score_pipeline_result(res_otsu)

    if score_otsu > score_primary:
        res_otsu["threshold_arbitration"] = {
            "chosen": "otsu",
            "reason": f"Otsu score ({score_otsu:.1f}) exceeded adaptive score ({score_primary:.1f})",
            "primary_score": score_primary,
            "alternative_score": score_otsu
        }
        return res_otsu

    res_primary["threshold_arbitration"] = {
        "chosen": "adaptive",
        "reason": f"Adaptive score ({score_primary:.1f}) >= Otsu score ({score_otsu:.1f})",
        "primary_score": score_primary,
        "alternative_score": score_otsu
    }
    return res_primary


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
    """Packages all intermediate images, diagnostics metadata, and markdown solution steps into a zip archive."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Solution Markdown
        sol = results["solution"]
        parsed = results["parsed"]
        prep = results["prep"]
        md_text = f"# MathVision Solution Report\n\n"
        md_text += f"- **Recognized Expression:** `{parsed.get('parsed_expression', '')}`\n"
        md_text += f"- **Expression Parsed:** `{parsed.get('parsed_expression', '')}`\n"
        md_text += f"- **Expression Validated:** `{parsed.get('is_valid', False)}`\n"
        md_text += f"- **Status:** {sol.get('status', '').upper()}\n"
        md_text += f"- **Solver Status:** {sol.get('status', '').upper()}\n"
        md_text += f"- **Answer Verified Against Image:** `False (Symbolic evaluation; independent ground truth verification only occurs in automated regression tests)`\n"
        md_text += f"- **Final Answer:** {sol.get('final_answer', '')}\n"
        md_text += f"- **LaTeX Formula:** $${parsed.get('latex_expression', '')}$$\n\n"

        md_text += "## Step-by-Step Working\n\n"
        for s in sol.get("steps", []):
            md_text += f"### Step {s['step_number']}: {s['title']}\n"
            md_text += f"{s['description']}\n\n"
            md_text += f"$${s['latex']}$$\n\n"

        # Diagnostic Summary Section
        md_text += "## Pipeline Diagnostics & Symbol Classification\n\n"
        md_text += f"- **Margin Auto-Crop Applied:** `{prep.get('margin_crop_applied', False)}`\n"
        if prep.get("margin_crop_applied", False):
            md_text += f"- **Paper Region Bounding Box:** `{prep.get('margin_crop_bbox')}`\n"
        md_text += f"- **Writing Region Crop Applied:** `{prep.get('writing_crop_applied', False)}`\n"
        if prep.get("writing_crop_applied", False):
            md_text += f"- **Writing Region Bounds (y):** `{prep.get('writing_bounds')}`\n"
        
        line_stats = prep.get("line_suppression_stats", {})
        md_text += f"- **Horizontal Line Suppression:** `{'Applied' if line_stats.get('lines_detected', False) else 'None'}` ({line_stats.get('pixels_removed', 0)} line pixels removed)\n"
        md_text += f"- **Segmentation Quality:** `{results.get('segmentation_quality', 'valid').upper()}`\n"
        for w in results.get("segmentation_warnings", []):
            md_text += f"  - *Warning:* {w}\n"
        md_text += f"- **Total Symbols Segmented:** `{len(results.get('predicted_symbols', []))}`\n"
        md_text += f"- **Mean Symbol Confidence:** `{results.get('mean_confidence', 0.0):.1%}`\n\n"

        md_text += "| Index | Bounding Box (x, y, w, h) | Symbol | Confidence | Top Alternatives |\n"
        md_text += "| :---: | :---: | :---: | :---: | :--- |\n"
        for d in results.get("symbol_diagnostics", []):
            top_alt = ", ".join([f"{sym} ({int(p*100)}%)" for sym, p in d.get("top3", [])[1:3]])
            bbox_str = str(d.get("bbox", ""))
            md_text += f"| {d.get('index', 0) + 1} | `{bbox_str}` | `{d.get('symbol')}` | {int(d.get('confidence', 0)*100)}% | {top_alt or 'None'} |\n"

        zip_file.writestr("solution_report.md", md_text)

        active_crop_img = prep.get("active_crop", prep.get("resized", prep["original"]))

        # 2. Comprehensive Diagnostics Metadata (JSON)
        diag_data = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_status": results.get("status"),
            "segmentation_quality": results.get("segmentation_quality", "valid"),
            "segmentation_warnings": results.get("segmentation_warnings", []),
            "timings_ms": results.get("timings", {}),
            "image_dimensions": {
                "original": list(prep["original"].shape) if "original" in prep and hasattr(prep["original"], "shape") else None,
                "margin_crop": list(prep["margin_crop_image"].shape) if "margin_crop_image" in prep and hasattr(prep["margin_crop_image"], "shape") else None,
                "active_crop": list(active_crop_img.shape) if hasattr(active_crop_img, "shape") else None,
                "segmentation_canvas": list(results["active_binary"].shape) if hasattr(results["active_binary"], "shape") else None,
            },
            "crop_coordinates": {
                "margin_crop_applied": prep.get("margin_crop_applied", False),
                "margin_crop_bbox": prep.get("margin_crop_bbox", None),
                "writing_crop_applied": prep.get("writing_crop_applied", False),
                "writing_region_bounds": prep.get("writing_bounds", None),
            },
            "margin_crop": {
                "applied": prep.get("margin_crop_applied", False),
                "bbox": prep.get("margin_crop_bbox", None)
            },
            "writing_region": {
                "applied": prep.get("writing_crop_applied", False),
                "bounds": prep.get("writing_bounds", None),
                "diagnostics": prep.get("writing_diag", {})
            },
            "line_suppression": line_stats,
            "contour_diagnostics": results.get("contour_diagnostics", []),
            "raw_boxes": results.get("raw_boxes", []),
            "merged_boxes": results.get("merged_boxes", []),
            "final_boxes": results.get("boxes", []),
            "predicted_symbols": results.get("predicted_symbols", []),
            "confidences": results.get("confidences", []),
            "symbol_diagnostics": results.get("symbol_diagnostics", []),
            "parsed": {
                "is_valid": parsed.get("is_valid"),
                "expression_parsed": parsed.get("parsed_expression"),
                "expression_validated": parsed.get("is_valid"),
                "expression": parsed.get("parsed_expression"),
                "latex": parsed.get("latex_expression"),
                "error": parsed.get("error_message"),
                "ambiguous_tokens": parsed.get("ambiguous_tokens", [])
            },
            "solution": {
                "status": sol.get("status"),
                "equation_type": sol.get("equation_type"),
                "final_answer": sol.get("final_answer"),
                "answer_verified": False,
                "verification_note": "Answer is symbolically computed from recognized expression; independent ground truth verification only occurs in automated regression tests.",
                "steps_count": len(sol.get("steps", []))
            }
        }
        zip_file.writestr("diagnostics.json", json.dumps(diag_data, indent=2))

        # 3. Pipeline Core Images
        _, orig_png = cv2.imencode(".png", prep["original"])
        zip_file.writestr("01_original.png", orig_png.tobytes())

        # Active writing region crop
        _, crop_bgr_png = cv2.imencode(".png", active_crop_img)
        zip_file.writestr("01b_active_crop.png", crop_bgr_png.tobytes())

        # Margin crop with bounds visualization
        if "margin_crop_with_bounds" in prep and prep["margin_crop_with_bounds"] is not None:
            _, m_bounds_png = cv2.imencode(".png", prep["margin_crop_with_bounds"])
            zip_file.writestr("01c_margin_crop_with_bounds.png", m_bounds_png.tobytes())

        # Threshold image before line suppression
        bin_raw = prep.get("binary_before_line_suppression", results["active_binary"])
        _, bin_raw_png = cv2.imencode(".png", bin_raw)
        zip_file.writestr("02_binary_threshold.png", bin_raw_png.tobytes())

        # Detected horizontal line mask
        line_mask_img = prep.get("horizontal_line_mask", np.zeros_like(bin_raw))
        _, mask_png = cv2.imencode(".png", line_mask_img)
        zip_file.writestr("02b_horizontal_line_mask.png", mask_png.tobytes())

        # Image after line suppression
        _, bin_png = cv2.imencode(".png", results["active_binary"])
        zip_file.writestr("02c_line_suppressed.png", bin_png.tobytes())

        # Annotated bounding boxes visualization
        _, vis_png = cv2.imencode(".png", results["annotated_vis"])
        zip_file.writestr("03_bounding_boxes.png", vis_png.tobytes())

        # 4. Candidate Threshold Masks
        thresh_comp = prep.get("threshold_comparison", {})
        for name, mask in thresh_comp.items():
            if mask is not None:
                _, m_png = cv2.imencode(".png", mask)
                zip_file.writestr(f"thresholds/{name}.png", m_png.tobytes())

        # 5. Normalized Symbol Crops
        for idx, crop in enumerate(results["symbol_crops"]):
            lbl = results["predicted_symbols"][idx]
            conf = int(results["confidences"][idx] * 100)
            crop_uint8 = np.uint8(crop * 255)
            _, crop_png = cv2.imencode(".png", crop_uint8)
            zip_file.writestr(f"symbols/crop_{idx+1:02d}_{lbl}_{conf}pct.png", crop_png.tobytes())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
