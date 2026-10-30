"""
MathVision Phase 3.1: Topological Gap Disambiguation Unit & Regression Tests
Validates that Case B (36/6+8 = 14) is cleanly solved under Model V4 while preserving
Case E (7*8+12/4 = 59) and ensuring genuine, faint, or broken plus signs are never converted.
"""

import os
import cv2
import numpy as np
import pytest
from tensorflow import keras

from experiments.phase3_1.topological_disambiguation import (
    check_division_topology,
    apply_topological_plus_div_disambiguation
)
from experiments.phase3_1.experimental_pipeline import run_experimental_pipeline

@pytest.fixture(scope="module")
def v4_model():
    model_path = "models/cnn_experimental_v4.keras"
    if not os.path.exists(model_path):
        pytest.skip(f"Model V4 checkpoint not found at {model_path}")
    return keras.models.load_model(model_path)


def test_case_b_experimental_pipeline_solves_fourteen(v4_model):
    """Case B (36/6+8 = 14) must evaluate to 14 under V4 with topological disambiguation."""
    img_path = "outputs/failure_case_b/01_original.png"
    assert os.path.exists(img_path), f"Case B image not found at {img_path}"
    img = cv2.imread(img_path)
    cached = {"cnn": v4_model, "svm": None}

    res = run_experimental_pipeline(img, cached_models=cached)
    assert res.get("status") == "success"
    assert res.get("expression_parsed") == "36/6+8"
    assert str(res.get("solution", {}).get("final_answer")) == "14"
    assert res.get("predicted_symbols") == ["3", "6", "div", "6", "+", "8"]


def test_case_e_experimental_pipeline_solves_fifty_nine(v4_model):
    """Case E (7*8+12/4 = 59) must evaluate to 59 under V4 with topological disambiguation."""
    img_path = "outputs/failure_case_e/01_original.png"
    assert os.path.exists(img_path), f"Case E image not found at {img_path}"
    img = cv2.imread(img_path)
    cached = {"cnn": v4_model, "svm": None}

    res = run_experimental_pipeline(img, cached_models=cached)
    assert res.get("status") == "success"
    assert res.get("expression_parsed") == "7*8+12/4"
    assert str(res.get("solution", {}).get("final_answer")) == "59"
    assert res.get("predicted_symbols") == ["7", "x", "8", "+", "1", "2", "div", "4"]


def test_genuine_plus_signs_never_converted_to_division():
    """All genuine plus signs from physical acceptance images must reject division topology."""
    genuine_plus_crops = [
        "outputs/phase3_investigation/case_b/baseline/adaptive/crops/crop_04_+.png",
        "outputs/phase3_investigation/case_e/baseline/otsu/crops/crop_03_+.png",
    ]
    for p in genuine_plus_crops:
        if os.path.exists(p):
            crop = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            is_div, reason, metrics = check_division_topology(crop)
            assert not is_div, f"Genuine plus sign falsely classified as division: {p} ({reason})"
            assert metrics["zeros_above"] == 0 or metrics["zeros_below"] == 0


def test_faint_plus_sign_never_converted_to_division():
    """Synthetic faint continuous plus sign must be rejected by division topology."""
    crop = np.zeros((45, 45), dtype=np.float32)
    # Faint horizontal bar (intensity 0.25)
    crop[22, 10:35] = 0.25
    # Faint vertical bar (intensity 0.25)
    crop[10:35, 22] = 0.25

    is_div, reason, _ = check_division_topology(crop)
    assert not is_div, f"Faint plus sign falsely classified as division: {reason}"


def test_broken_plus_sign_with_hairline_gap_rejected():
    """A plus sign with a 1-pixel hairline noise dropout must not trigger division topology."""
    crop = np.zeros((45, 45), dtype=np.float32)
    # Horizontal bar
    crop[22, 10:35] = 0.90
    # Vertical bar with 1-pixel gap at y=21
    crop[10:21, 22] = 0.90
    # gap at y=21
    crop[22:35, 22] = 0.90

    # min_gap_rows is 2, so 1-pixel gap is safely rejected
    is_div, reason, metrics = check_division_topology(crop, min_gap_rows=2)
    assert not is_div, f"Broken plus sign falsely classified as division: {reason}"


def test_orthogonal_division_symbol_detected():
    """Standard division symbol with separated dots must be recognized as division topology."""
    e6_path = "outputs/phase3_investigation/case_e/baseline/otsu/crops/crop_06_div.png"
    if os.path.exists(e6_path):
        crop = cv2.imread(e6_path, cv2.IMREAD_GRAYSCALE)
        is_div, reason, metrics = check_division_topology(crop)
        assert is_div, f"Standard division symbol rejected: {reason}"
        assert metrics["zeros_above"] >= 1
        assert metrics["zeros_below"] >= 1


def test_diagonal_division_slash_not_treated_as_orthogonal_division():
    """Diagonal forward slash '/' must not trigger orthogonal division topology."""
    crop = np.zeros((45, 45), dtype=np.float32)
    # Draw diagonal line from (10, 35) to (35, 10)
    for t in range(25):
        y = 35 - t
        x = 10 + t
        crop[y, x] = 0.90
        crop[y, min(44, x + 1)] = 0.80

    is_div, reason, _ = check_division_topology(crop)
    assert not is_div, f"Diagonal slash falsely triggered orthogonal division: {reason}"


def test_elongated_division_dots_detected():
    """Division symbol with vertically elongated dots (Case B Crop 2) must be recognized."""
    b2_path = "outputs/phase3_investigation/case_b/baseline/adaptive/crops/crop_02_div.png"
    if os.path.exists(b2_path):
        crop = cv2.imread(b2_path, cv2.IMREAD_GRAYSCALE)
        is_div, reason, metrics = check_division_topology(crop)
        assert is_div, f"Elongated division symbol rejected: {reason}"
        assert metrics["zeros_below"] >= 2
        assert metrics["num_components"] >= 2


def test_division_symbols_with_connected_dots_safe_fallback():
    """If noise completely bridges division dots to bar, topology safely rejects without guessing."""
    crop = np.zeros((45, 45), dtype=np.float32)
    # Horizontal bar
    crop[22, 8:37] = 0.90
    # Continuous vertical stroke touching bar without any gap
    crop[10:35, 22] = 0.90

    is_div, reason, _ = check_division_topology(crop)
    assert not is_div, "Connected-stroke crop should safely fall back to classifier output"
