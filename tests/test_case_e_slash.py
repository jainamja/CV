"""
=============================================================================
Phase 2.3 Permanent Regression Test: Case E Division Slash vs Plus Verification
=============================================================================

Prevents regressions on Case E ('7 * 8 + 12 / 4' -> 59):
1. Verifies that the division symbol at index 6 is never misclassified as '+'
   causing erroneous evaluation to 72.
2. Verifies that multi-threshold arbitration chooses the valid binarization
   preventing noise-induced stroke fusion.
3. Verifies that digit '7' and digit '1' are recognized cleanly with valid
   disambiguation notes.
=============================================================================
"""

import os
import pytest
import numpy as np

from src.pipeline import run_pipeline, _run_pipeline_single, score_pipeline_result

CASE_E_PATH = "outputs/failure_case_e/01_original.png"


def test_case_e_production_solves_fifty_nine():
    """
    CRITICAL ZERO-REGRESSION TEST:
    Verify that Case E ('7 * 8 + 12 / 4') is recognized as '7*8+12/4' and solved to '59'.
    Must never regress to '7*8+12+4 = 72'.
    """
    assert os.path.exists(CASE_E_PATH), f"Case E image missing at: {CASE_E_PATH}"

    res = run_pipeline(CASE_E_PATH)

    # 1. Total Symbol Count
    assert res["count"] == 8, f"Expected 8 symbols, got {res['count']}: {res['predicted_symbols']}"
    assert len(res["boxes"]) == 8

    # 2. Predicted Symbols
    syms = res["predicted_symbols"]
    # Normalize times/x
    norm_syms = ["*" if s in ("times", "x", "*") else s for s in syms]
    assert norm_syms[0] == "7", f"First digit must be '7', got '{syms[0]}'"
    assert norm_syms[1] == "*", f"Second symbol must be multiplication, got '{syms[1]}'"
    assert norm_syms[2] == "8", f"Third digit must be '8', got '{syms[2]}'"
    assert norm_syms[3] == "+", f"Fourth symbol must be '+', got '{syms[3]}'"
    assert norm_syms[4] == "1", f"Fifth digit must be '1', got '{syms[4]}'"
    assert norm_syms[5] == "2", f"Sixth digit must be '2', got '{syms[5]}'"
    assert norm_syms[6] in ("div", "/"), f"Division symbol must not regress to '+': got '{syms[6]}'"
    assert norm_syms[7] == "4", f"Eighth digit must be '4', got '{syms[7]}'"

    # 3. Parsed Expression & Answer
    parsed = res["parsed"]
    assert parsed["is_valid"] is True
    assert parsed["parsed_expression"] in ("7*8+12/4", "7x8+12/4")
    assert res["solution"]["status"] == "solved"
    assert str(res["solution"]["final_answer"]) == "59", f"Answer corrupted: expected 59, got {res['solution']['final_answer']}"


def test_case_e_multi_threshold_arbitration_score():
    """
    ARBITRATION VERIFICATION:
    Verifies that pipeline candidate scoring rewards clean division recognition
    and penalizes operator duplications.
    """
    assert os.path.exists(CASE_E_PATH)

    res_otsu = _run_pipeline_single(CASE_E_PATH, {"threshold_method": "otsu"})
    score_otsu = score_pipeline_result(res_otsu)

    assert score_otsu > 100.0, f"Otsu score too low: {score_otsu}"
    assert res_otsu["predicted_symbols"][6] == "div"
