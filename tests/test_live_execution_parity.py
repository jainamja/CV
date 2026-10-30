"""
Tests for Live Streamlit Parity, Border Rejection Safety, and Single-Symbol Diagnostics
"""

import os
import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.pipeline import run_pipeline, get_loaded_models
from src.parser import parse_symbols, validate_expression
from src.solver import solve_expression
from src.segment import find_symbol_contours, SEGMENT_CONFIG


def test_apptest_live_upload_exact_parity():
    """Verify live Streamlit app upload of 01_original.png yields 12+8/4 = 14."""
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=60)
    at.run(timeout=60)
    
    with open("outputs/real_test_case/01_original.png", "rb") as f:
        file_bytes = f.read()
        
    uploader = at.get("file_uploader")[0]
    uploader.upload("01_original.png", file_bytes, "image/png")
    at.run(timeout=60)
    
    res = at.session_state.get("pipeline_results")
    assert res is not None
    assert res["count"] == 6
    assert res["predicted_symbols"] == ["1", "2", "+", "8", "div", "4"]
    assert res["parsed"]["parsed_expression"] == "12+8/4"
    assert res["solution"]["final_answer"] == "14"


def test_border_rejection_does_not_drop_narrow_digits():
    """Ensure tall, narrow digits (w<=5) touching the border are not discarded unless spanning >50% height."""
    # Create binary image with a narrow vertical stroke '1' (w=4, h=30) at left edge (x=1) in 200h canvas
    canvas = np.zeros((200, 200), dtype=np.uint8)
    canvas[80:110, 1:5] = 255  # 30 px tall stroke
    
    boxes = find_symbol_contours(canvas, SEGMENT_CONFIG)
    assert len(boxes) == 1
    assert boxes[0] == (1, 80, 4, 30)


def test_threshold_methods_clean_routing():
    """Verify that 'otsu', 'adaptive', and 'global' settings all return non-empty binary masks and 6 symbols."""
    img = cv2.imread("outputs/real_test_case/01_original.png")
    models = get_loaded_models("models")
    
    for method in ["otsu", "adaptive", "global"]:
        settings = {
            "threshold_method": method,
            "adaptive_block_size": 15,
            "adaptive_c": 8,
            "min_contour_area": 10,
            "merge_min_h_overlap_ratio": 0.35,
            "padding_margin_ratio": 0.04,
            "model_type": "cnn",
            "confidence_threshold": 0.70,
            "debug": False,
        }
        res = run_pipeline(img, settings=settings, cached_models=models)
        assert res["count"] == 6
        assert res["predicted_symbols"] == ["1", "2", "+", "8", "div", "4"]
        assert res["solution"]["final_answer"] == "14"
