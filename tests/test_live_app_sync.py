"""
=============================================================================
Regression Tests for Live App & Pipeline Synchronization
=============================================================================
Verifies:
1. Original handwritten image 01_original.png -> '12+8/4' -> answer '14'.
2. Live and direct execution parity with the exact app settings.
3. Syntax preview, alerts, and solver consume the exact same expression.
4. Exported ZIP reports strictly describe the current run.
5. Consecutive image uploads maintain clean state isolation (no override leak).
6. Invalid inputs remain invalid and are never silently mutated into false syntax.
"""

import os
import io
import zipfile
import cv2
import numpy as np
import pytest

from src.pipeline import run_pipeline, get_loaded_models, create_results_zip
from src.parser import validate_expression, parse_symbols, to_latex
from src.solver import solve_expression


def test_original_image_full_parity():
    """Verify 01_original.png produces 12+8/4 and evaluates to 14 with 6 crops."""
    img_path = "outputs/real_test_case/01_original.png"
    assert os.path.exists(img_path)

    img = cv2.imread(img_path)
    res = run_pipeline(img)

    assert res["status"] == "success"
    assert res["count"] == 6
    assert res["predicted_symbols"] == ["1", "2", "+", "8", "div", "4"]
    assert res["parsed"]["is_valid"] is True
    assert res["parsed"]["parsed_expression"] == "12+8/4"
    assert res["solution"]["status"] == "solved"
    assert res["solution"]["final_answer"] == "14"


def test_live_app_settings_parity():
    """
    Verify that executing with the exact default settings dictionary from app.py
    yields identical recognition and solving outcomes.
    """
    img_path = "outputs/real_test_case/01_original.png"
    img = cv2.imread(img_path)

    app_default_settings = {
        "threshold_method": "otsu",
        "adaptive_block_size": 15,
        "adaptive_c": 8,
        "min_contour_area": 10,
        "merge_min_h_overlap_ratio": 0.35,
        "padding_margin_ratio": 0.04,
        "model_type": "cnn",
        "confidence_threshold": 0.70,
        "debug": False,
    }

    res = run_pipeline(img, settings=app_default_settings)
    assert res["predicted_symbols"] == ["1", "2", "+", "8", "div", "4"]
    assert res["parsed"]["parsed_expression"] == "12+8/4"
    assert res["solution"]["final_answer"] == "14"


def test_syntax_preview_and_validation_consistency():
    """
    Verify that the syntax status shown in the preview is strictly consistent with
    validation: an invalid expression must never be flagged as valid.
    """
    # Valid equation
    valid_expr = "12+8/4"
    is_val, err = validate_expression(valid_expr)
    assert is_val is True
    assert err == ""

    # Invalid expression with missing closing parenthesis (matching the user's reported bug)
    invalid_expr = "1*(+1"
    is_val2, err2 = validate_expression(invalid_expr)
    assert is_val2 is False
    assert "Unbalanced parentheses" in err2 or "Syntax error" in err2

    # Solver outcome must reflect syntax error
    sol_res = solve_expression(invalid_expr)
    assert sol_res["status"] == "error"
    assert sol_res["final_answer"] != "14"


def test_consecutive_uploads_isolation_no_leak():
    """
    Verify that if a user manually sets an override on Image A,
    Image B does not inherit or leak Image A's overrides.
    """
    img_a = cv2.imread("data/samples/sample_01.png")
    img_b = cv2.imread("outputs/real_test_case/01_original.png")

    # Run A with an override mapping
    overrides_a = {1: "(", 2: "+"}
    res_a = run_pipeline(img_a, manual_overrides=overrides_a)
    assert res_a["predicted_symbols"][1] == "("

    # Run B with cleared overrides (simulating the new image upload reset logic in app.py)
    overrides_b = {}
    res_b = run_pipeline(img_b, manual_overrides=overrides_b)
    # Must predict normal '2' for symbol 1, not '('
    assert res_b["predicted_symbols"][1] == "2"
    assert res_b["predicted_symbols"] == ["1", "2", "+", "8", "div", "4"]
    assert res_b["parsed"]["parsed_expression"] == "12+8/4"


def test_zip_matches_current_run_report():
    """Verify that create_results_zip uses current run artifacts and describes the exact evaluated expression."""
    img = cv2.imread("outputs/real_test_case/01_original.png")
    res = run_pipeline(img)
    zip_bytes = create_results_zip(res)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        report = z.read("solution_report.md").decode("utf-8")
        assert "- **Recognized Expression:** `12+8/4`" in report
        assert "- **Final Answer:** 14" in report
        assert "- **Status:** SOLVED" in report
        # Confirm all 6 symbol crops are in zip
        crops_in_zip = [n for n in z.namelist() if n.startswith("symbols/")]
        assert len(crops_in_zip) == 6


def test_invalid_input_remains_invalid_and_not_silently_repaired():
    """Verify that genuinely invalid syntax remains invalid and is not silently repaired."""
    invalid_cases = [
        "12 + + 4",
        "(12 + 4",
        "* 5 = 10",
        "12 / / 4",
    ]
    for expr in invalid_cases:
        is_val, err = validate_expression(expr)
        assert is_val is False, f"Expression '{expr}' was expected to be invalid, but passed validation."
        assert len(err) > 0
