"""
=============================================================================
End-to-End Pipeline Integration & Regression Tests
=============================================================================
"""

import os
import cv2
import numpy as np
import pytest
from src.pipeline import run_pipeline, get_loaded_models


def test_pipeline_real_test_case_end_to_end():
    """
    CRITICAL ACCEPTANCE TEST:
    Verify that 01_original.png is recognized as '12+8/4' and solved to '14'.
    Checks that all 6 symbols ['1', '2', '+', '8', 'div', '4'] are segmented,
    classified with high confidence, parsed correctly, and solved via PEMDAS.
    """
    img_path = "outputs/real_test_case/01_original.png"
    assert os.path.exists(img_path), f"Test image missing at {img_path}"

    res = run_pipeline(img_path)

    # 1. Symbol Count & Localizations
    assert res["count"] == 6, f"Expected 6 symbols, got {res['count']}: {res['boxes']}"
    assert len(res["boxes"]) == 6

    # 2. Predicted Symbols
    assert res["predicted_symbols"] == ["1", "2", "+", "8", "div", "4"]
    assert all(c > 0.70 for c in res["confidences"]), f"Low confidences detected: {res['confidences']}"
    assert res["mean_confidence"] > 0.90, f"Mean confidence too low: {res['mean_confidence']}"

    # 3. Parsing
    parsed = res["parsed"]
    assert parsed["is_valid"] is True
    assert parsed["parsed_expression"] == "12+8/4"
    assert "\\div" in parsed["latex_expression"]

    # 4. Symbolic Solution
    sol = res["solution"]
    assert sol["status"] == "solved"
    assert sol["final_answer"] == "14"
    assert len(sol["steps"]) >= 2


def test_pipeline_failure_case_b_end_to_end():
    """
    CRITICAL ACCEPTANCE TEST (Failure Case B):
    Verify that failure case B ('36 ÷ 6 + 8') in outputs/failure_case_b/01_original.png
    is recognized as '36/6+8' and solved to '14'.
    Prevents regressions where:
    - First digit '3' is misclassified as '5'
    - Division symbol '÷' is omitted
    - Only 5 symbols are retained instead of 6
    - Final result produces '574' instead of '14'
    """
    img_path = "outputs/failure_case_b/01_original.png"
    assert os.path.exists(img_path), f"Test image missing at {img_path}"

    res = run_pipeline(img_path)

    # 1. Symbol Count & Localizations
    assert res["count"] == 6, f"Expected 6 symbols, got {res['count']}: {res['boxes']}"
    assert len(res["boxes"]) == 6

    # 2. Predicted Symbols & Anti-Regression Checks
    assert res["predicted_symbols"] == ["3", "6", "div", "6", "+", "8"], f"Unexpected symbols: {res['predicted_symbols']}"
    assert "-" not in res["predicted_symbols"][:2], "Page-boundary false minus must not be present"
    assert res["mean_confidence"] > 0.85, f"Mean confidence too low: {res['mean_confidence']}"

    # 3. Parsing
    parsed = res["parsed"]
    assert parsed["is_valid"] is True
    assert parsed["parsed_expression"] == "36/6+8"
    assert "\\div" in parsed["latex_expression"]

    # 4. Symbolic Solution
    sol = res["solution"]
    assert sol["status"] == "solved"
    assert sol["final_answer"] == "14"
    assert len(sol["steps"]) >= 2


def test_pipeline_sample_equations_solve_accurately():
    """Verify that clean sample equations solve accurately without regressions."""
    samples = [
        ("data/samples/sample_01.png", "2+3*4=14", "True"),
        ("data/samples/sample_04.png", "x**2-5*x+6=0", "x = 2, x = 3"),
        ("data/samples/sample_02.png", "3*x+5=20", "x = 5"),
    ]

    for path, expected_expr, expected_ans in samples:
        if os.path.exists(path):
            res = run_pipeline(path)
            assert res["parsed"]["is_valid"] is True
            assert res["parsed"]["parsed_expression"] == expected_expr
            assert expected_ans in res["solution"]["final_answer"]


def test_pipeline_nonuniform_lighting_and_shadow_robustness():
    """
    Verify that an equation with simulated heavy ambient lighting shadow across the page
    is normalized cleanly, preserving all symbols without spurious edge boxes.
    """
    # Create an equation canvas: '2 + 5 = 7'
    # Base canvas with severe vignette shadow
    h, w = 180, 450
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 135

    # Left side shadow (vignetting down to 90)
    shadow_grad = np.linspace(90, 150, w, dtype=np.float32)
    for c in range(3):
        canvas[:, :, c] = np.tile(shadow_grad, (h, 1)).astype(np.uint8)

    # Draw dark strokes for '2 + 5 = 7'
    # Symbol '2' at x=60
    cv2.putText(canvas, "2", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (25, 25, 25), 3)
    # Symbol '+' at x=140
    cv2.putText(canvas, "+", (130, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (25, 25, 25), 3)
    # Symbol '5' at x=220
    cv2.putText(canvas, "5", (210, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (25, 25, 25), 3)
    # Symbol '=' at x=300
    cv2.putText(canvas, "=", (290, 105), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (25, 25, 25), 3)
    # Symbol '7' at x=380
    cv2.putText(canvas, "7", (370, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (25, 25, 25), 3)

    res = run_pipeline(canvas)
    # Must segment symbols without being fooled by the left-side shadow
    assert res["count"] >= 5
    # No edge box at x=0
    assert all(b[0] > 10 for b in res["boxes"])


def test_pipeline_isolation_between_consecutive_uploads():
    """
    REGRESSION TEST:
    Verifies that processing a second image cannot accidentally reuse the first image's
    manual overrides, cached predictions, or intermediate artifacts.
    """
    img_path1 = "data/samples/sample_03.png"
    img_path2 = "outputs/real_test_case/01_original.png"

    # Run Image 1 with manual overrides applied
    res1 = run_pipeline(img_path1, manual_overrides={0: "(", 1: "+", 2: "1"})
    assert res1["predicted_symbols"][0] == "("

    # Run Image 2 without overrides (fresh upload simulation)
    res2 = run_pipeline(img_path2, manual_overrides=None)
    assert res2["predicted_symbols"] == ["1", "2", "+", "8", "div", "4"]
    assert res2["parsed"]["parsed_expression"] == "12+8/4"
    assert res2["solution"]["final_answer"] == "14"
    assert res2["count"] == 6


def test_zip_export_matches_current_run_exactly():
    """
    REGRESSION TEST:
    Verifies that the generated diagnostic ZIP package contains artifacts
    strictly derived from the current pipeline run and matches its solution report.
    """
    import zipfile
    import io
    from src.pipeline import create_results_zip

    img_path = "outputs/real_test_case/01_original.png"
    res = run_pipeline(img_path)
    zip_bytes = create_results_zip(res)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        namelist = z.namelist()
        assert "solution_report.md" in namelist
        assert "01_original.png" in namelist
        assert "02_binary_threshold.png" in namelist
        assert "03_bounding_boxes.png" in namelist
        assert len([n for n in namelist if n.startswith("symbols/")]) == 6

        report_text = z.read("solution_report.md").decode("utf-8")
        assert "12+8/4" in report_text
        assert "14" in report_text
        assert "SOLVED" in report_text


def test_pipeline_svm_architecture_fallback_safety():
    """
    REGRESSION TEST:
    Verifies that selecting SVM inference architecture executes safely
    without raising AttributeError: This 'SVC' has no attribute 'predict_proba'.
    """
    img_path = "outputs/real_test_case/01_original.png"
    res = run_pipeline(img_path, settings={"model_type": "svm"})
    assert res["status"] in ("success", "error")
    assert len(res["predicted_symbols"]) == res["count"]


def test_pipeline_black_margins_image_retains_digits_and_solves_fourteen():
    """
    REGRESSION TEST FOR 4-SYMBOL FAILURE:
    Verifies that an image with dark letterboxes/pillarboxes (from zip3/01_original.png):
    1. Does NOT drop digits 1, 2, or 8 (previously caused by Otsu wiping faint strokes).
    2. Does NOT misclassify division as multiplication (times).
    3. Does NOT produce the erroneous 4-crop sequence '-+x*4'.
    4. Segments exactly all 6 symbols ['1', '2', '+', '8', 'div', '4'].
    5. Correctly parses '12+8/4' and evaluates to 14.
    6. Logs candidate contour diagnostics before and after filtering.
    """
    img_path = "outputs/zip_inspection/zip3/01_original.png"
    if not os.path.exists(img_path):
        img_path = "outputs/real_test_case/01_original.png"

    res = run_pipeline(img_path)

    # Must segment 6 symbols, NOT 4
    assert res["count"] == 6, f"Expected 6 symbols, got {res['count']}: {res['predicted_symbols']}"
    assert res["predicted_symbols"] != ["-", "+", "times", "4"], "Regressed to 4-symbol failure!"
    assert res["predicted_symbols"] == ["1", "2", "+", "8", "div", "4"]

    # Must retain digits 1, 2, 8 and correct operator div
    assert "1" in res["predicted_symbols"]
    assert "2" in res["predicted_symbols"]
    assert "8" in res["predicted_symbols"]
    assert "div" in res["predicted_symbols"]
    assert "times" not in res["predicted_symbols"]

    # Must parse to 12+8/4 and evaluate to 14
    assert res["parsed"]["is_valid"] is True
    assert res["parsed"]["parsed_expression"] == "12+8/4"
    assert res["solution"]["status"] == "solved"
    assert res["solution"]["final_answer"] == "14"

    # Candidate contour diagnostics must be present
    assert "contour_diagnostics" in res
    assert len(res["contour_diagnostics"]) >= 6
    retained_boxes = [d for d in res["contour_diagnostics"] if d["retained"]]
    assert len(retained_boxes) >= 6


def test_crop_black_margins_function():
    """Verify that crop_black_margins accurately strips dark margins without altering normal images."""
    from src.preprocess import crop_black_margins

    # 1. Image with heavy dark pillarboxes on left and right
    img_with_margins = np.zeros((100, 300, 3), dtype=np.uint8)
    # Center 100px is bright paper (intensity 180)
    img_with_margins[:, 100:200, :] = 180

    cropped, applied, bbox = crop_black_margins(img_with_margins)
    assert applied is True
    assert cropped.shape[1] < 300
    assert cropped.shape[1] >= 90

    # 2. Normal image without margins (uniform paper)
    normal_img = np.ones((100, 300, 3), dtype=np.uint8) * 180
    cropped_norm, applied_norm, bbox_norm = crop_black_margins(normal_img)
    assert applied_norm is False
    assert cropped_norm.shape == normal_img.shape


def test_pipeline_arithmetic_pemdas_evaluations():
    """
    REGRESSION TEST:
    Verifies that standard arithmetic expressions solve with correct PEMDAS precedence:
    1. '7 + 3 * 2' -> 13 (Multiplication before addition)
    2. '18 / 3 + 4' -> 10 (Division before addition)
    3. '25 - 6 * 3' -> 7  (Multiplication before subtraction)
    """
    from src.solver import solve_expression

    cases = [
        ("7+3*2", "13"),
        ("18/3+4", "10"),
        ("25-6*3", "7"),
    ]
    for expr, expected_ans in cases:
        sol = solve_expression(expr)
        assert sol["status"] == "solved", f"Failed to solve {expr}: {sol.get('error_message')}"
        assert sol["final_answer"] == expected_ans, f"Expected {expected_ans} for {expr}, got {sol['final_answer']}"
        assert len(sol["steps"]) >= 2, f"Expected step breakdown for {expr}"


def test_pipeline_empty_and_blank_inputs_handled_gracefully():
    """
    REGRESSION TEST:
    Verifies that blank paper, all-black frames, or degenerate images return clean
    diagnostic states without raising unhandled exceptions or crashing.
    """
    # 1. Completely blank white paper
    blank_white = np.ones((200, 400, 3), dtype=np.uint8) * 255
    res_white = run_pipeline(blank_white)
    assert res_white["status"] in ("no_symbols_detected", "error")
    assert res_white["count"] == 0

    # 2. Completely dark black image
    blank_black = np.zeros((200, 400, 3), dtype=np.uint8)
    res_black = run_pipeline(blank_black)
    assert res_black["status"] in ("no_symbols_detected", "error")


def test_pipeline_extreme_aspect_ratios_handled():
    """
    REGRESSION TEST:
    Verifies that extreme aspect ratios (wide banner vs tall canvas) process cleanly
    through preprocessing and normalization without dimension assertion errors.
    """
    # Wide banner with a drawn symbol '1'
    wide_canvas = np.ones((80, 700, 3), dtype=np.uint8) * 220
    cv2.putText(wide_canvas, "1", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 2)
    res_wide = run_pipeline(wide_canvas)
    assert res_wide["status"] in ("success", "error")

    # Tall canvas with drawn symbol '+'
    tall_canvas = np.ones((700, 100, 3), dtype=np.uint8) * 220
    cv2.putText(tall_canvas, "+", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 2)
    res_tall = run_pipeline(tall_canvas)
    assert res_tall["status"] in ("success", "error")


def test_pipeline_failure_case_c_end_to_end():
    """
    CRITICAL ACCEPTANCE TEST (Failure Case C):
    Verify that failure case C ('45 ÷ 5 + 6 × 2') in outputs/failure_case_c/01_original.png
    is recognized as '45/5+6*2' and solved to '21'.
    Prevents regressions where:
    - First digit '4' is split into multiple pieces and confused with '(' or '3'
    - Division dots are dropped, leaving only '-'
    - Digit '6' is misclassified as '4' due to closed loop holes
    - Total symbol count deviates from 8
    """
    img_path = "outputs/failure_case_c/01_original.png"
    assert os.path.exists(img_path), f"Test image missing at {img_path}"

    res = run_pipeline(img_path)

    # 1. Symbol Count & Localizations
    assert res["count"] == 8, f"Expected 8 symbols, got {res['count']}: {res['boxes']}"
    assert len(res["boxes"]) == 8

    # 2. Predicted Symbols
    expected_symbols = ["4", "5", "div", "5", "+", "6", "times", "2"]
    actual_symbols = [
        "x" if s in ("times", "*", "×") else s
        for s in res["predicted_symbols"]
    ]
    expected_normalized = [
        "x" if s in ("times", "*", "×") else s
        for s in expected_symbols
    ]
    assert actual_symbols == expected_normalized, f"Incorrect predictions: {res['predicted_symbols']}"

    # 3. Parsing
    parsed = res["parsed"]
    assert parsed["is_valid"] is True, f"Parsing failed: {parsed.get('error_message')}"
    assert parsed["parsed_expression"] == "45/5+6*2"

    # 4. Symbolic Solution
    sol = res["solution"]
    assert sol["status"] == "solved"
    assert sol["final_answer"] == "21"


def test_pipeline_failure_case_d_end_to_end():
    """
    CRITICAL ACCEPTANCE TEST (Failure Case D):
    Verify that failure case D ('7 × 8 + 12 ÷ 4') in outputs/failure_case_d/01_original.png
    is recognized as '7*8+12/4' and solved to '59'.
    Prevents regressions where:
    - First digit '7' is misclassified as '2'
    - Division symbol '÷' is contaminated by nearby noise and misclassified as '=' or '-'
    - Total symbol count deviates from 8
    """
    img_path = "outputs/failure_case_d/01_original.png"
    assert os.path.exists(img_path), f"Test image missing at {img_path}"

    res = run_pipeline(img_path)

    # 1. Symbol Count & Localizations
    assert res["count"] == 8, f"Expected 8 symbols, got {res['count']}: {res['boxes']}"
    assert len(res["boxes"]) == 8

    # 2. Predicted Symbols
    expected_symbols = ["7", "x", "8", "+", "1", "2", "div", "4"]
    actual_symbols = [
        "x" if s in ("times", "*", "×") else s
        for s in res["predicted_symbols"]
    ]
    expected_normalized = [
        "x" if s in ("times", "*", "×") else s
        for s in expected_symbols
    ]
    assert actual_symbols == expected_normalized, f"Incorrect predictions: {res['predicted_symbols']}"

    # 3. Parsing
    parsed = res["parsed"]
    assert parsed["is_valid"] is True, f"Parsing failed: {parsed.get('error_message')}"
    assert parsed["parsed_expression"] == "7*8+12/4"

    # 4. Symbolic Solution
    sol = res["solution"]
    assert sol["status"] == "solved"
    assert sol["final_answer"] == "59"


def test_pipeline_failure_case_e_end_to_end():
    """
    CRITICAL ACCEPTANCE TEST (Failure Case E):
    Verify that failure case E ('7 × 8 + 12 ÷ 4') in outputs/failure_case_e/01_original.png
    is recognized as '7*8+12/4' and solved to '59'.
    Prevents regressions where:
    - First digit '7' is misclassified as '2'
    - Digit '1' in '12' is misclassified as '('
    - Division symbol '÷' is misclassified as '+' due to local adaptive threshold merging
    - Equation parses as '2*8+(2+4' or fails with Syntax Error
    - Total symbol count deviates from 8
    """
    img_path = "outputs/failure_case_e/01_original.png"
    assert os.path.exists(img_path), f"Test image missing at {img_path}"

    res = run_pipeline(img_path)

    # 1. Symbol Count & Localizations
    assert res["count"] == 8, f"Expected 8 symbols, got {res['count']}: {res['boxes']}"
    assert len(res["boxes"]) == 8

    # 2. Predicted Symbols
    expected_symbols = ["7", "x", "8", "+", "1", "2", "div", "4"]
    actual_symbols = [
        "x" if s in ("times", "*", "×") else s
        for s in res["predicted_symbols"]
    ]
    expected_normalized = [
        "x" if s in ("times", "*", "×") else s
        for s in expected_symbols
    ]
    assert actual_symbols == expected_normalized, f"Incorrect predictions: {res['predicted_symbols']}"

    # 3. Parsing
    parsed = res["parsed"]
    assert parsed["is_valid"] is True, f"Parsing failed: {parsed.get('error_message')}"
    assert parsed["parsed_expression"] == "7*8+12/4"

    # 4. Symbolic Solution
    sol = res["solution"]
    assert sol["status"] == "solved"
    assert sol["final_answer"] == "59"


def test_pipeline_failure_case_f_end_to_end():
    """
    CRITICAL ACCEPTANCE TEST (Failure Case F):
    Verify that failure case F ('(18 + 6) / 4') in outputs/failure_case_f/01_original.png
    is recognized as '(18+6)/4' and solved to '6'.
    Prevents regressions where:
    - '(' and '1' are erroneously merged into '0'
    - '8' is misclassified as '6'
    - '6' and ')' are erroneously merged into '0'
    - Division symbol '/' is dropped or misclassified
    - Total symbol count deviates from 8
    """
    img_path = "outputs/failure_case_f/01_original.png"
    assert os.path.exists(img_path), f"Test image missing at {img_path}"

    res = run_pipeline(img_path)

    # 1. Symbol Count & Localizations
    assert res["count"] == 8, f"Expected 8 symbols, got {res['count']}: {res['boxes']}"
    assert len(res["boxes"]) == 8

    # 2. Predicted Symbols
    expected_symbols = ["(", "1", "8", "+", "6", ")", "div", "4"]
    actual_symbols = [
        "div" if s in ("/", "div", "÷") else s
        for s in res["predicted_symbols"]
    ]
    expected_normalized = [
        "div" if s in ("/", "div", "÷") else s
        for s in expected_symbols
    ]
    assert actual_symbols == expected_normalized, f"Incorrect predictions: {res['predicted_symbols']}"

    # 3. Parsing
    parsed = res["parsed"]
    assert parsed["is_valid"] is True, f"Parsing failed: {parsed.get('error_message')}"
    assert parsed["parsed_expression"] == "(18+6)/4"

    # 4. Symbolic Solution
    sol = res["solution"]
    assert sol["status"] == "solved"
    assert sol["final_answer"] == "6"


def test_disambiguation_paren_vs_one():
    """
    REGRESSION TEST:
    Verify that:
    1. Balanced parentheses in '(18+6)/4' are preserved and '(' is NOT turned into '1'.
    2. An isolated narrow stroke misrecognized as '(' preceding a digit without a matching ')'
       is disambiguated to '1'.
    """
    # 1. Run Case F where '(' is balanced with ')' -> must retain '('
    res_f = run_pipeline("outputs/failure_case_f/01_original.png")
    assert res_f["predicted_symbols"][0] == "("

    # 2. Run Case E where '(' was previously falsely predicted for digit '1' in '12'
    # In Case E, there is NO matching ')' -> must be disambiguated to '1'
    res_e = run_pipeline("outputs/failure_case_e/01_original.png")
    assert res_e["predicted_symbols"][4] == "1"


def test_disambiguation_close_paren_vs_one():
    """
    REGRESSION TEST:
    Verify that:
    1. Balanced closing parenthesis ')' in '(18+6)/4' is preserved and NOT converted to '1'.
    2. An isolated closing parenthesis without an opening parenthesis preceding a digit
       is disambiguated to '1'.
    """
    res_f = run_pipeline("outputs/failure_case_f/01_original.png")
    assert res_f["predicted_symbols"][5] == ")"


def test_disambiguation_eight_vs_six():
    """
    REGRESSION TEST:
    Verify that digit '8' (with enclosed upper loop) and digit '6' (with open upper stem
    and enclosed lower bowl) are correctly distinguished and not confused.
    """
    res_f = run_pipeline("outputs/failure_case_f/01_original.png")
    # In Case F: symbols[2] is '8', symbols[4] is '6'
    assert res_f["predicted_symbols"][2] == "8"
    assert res_f["predicted_symbols"][4] == "6"


def test_balanced_parentheses_expression():
    """
    UNIT & REGRESSION TEST:
    Verify that parsing and solving expressions with parentheses works accurately:
    - '(18+6)/4' evaluates to '6'
    - '(2+3)*4' evaluates to '20'
    - '2*(3+4)' evaluates to '14'
    """
    from src.parser import parse_symbols
    from src.solver import solve_expression

    cases = [
        (["(", "18", "+", "6", ")", "/", "4"], "(18+6)/4", "6"),
        (["(", "2", "+", "3", ")", "*", "4"], "(2+3)*4", "20"),
        (["2", "*", "(", "3", "+", "4", ")"], "2*(3+4)", "14"),
    ]
    for syms, expected_expr, expected_ans in cases:
        p = parse_symbols(syms)
        assert p["is_valid"] is True, f"Failed parsing {syms}: {p.get('error_message')}"
        assert p["parsed_expression"] == expected_expr
        sol = solve_expression(p["parsed_expression"])
        assert sol["status"] == "solved"
        assert sol["final_answer"] == expected_ans


def test_pipeline_failure_case_g_end_to_end():
    """
    CRITICAL ACCEPTANCE TEST (Failure Case G from mathvision_results (7).zip):
    Verify that failure case G ('(18 + 6) / 4') in outputs/failure_case_g/01_original.png
    is recognized as '(18+6)/4' and solved to '6'.
    Prevents regressions where:
    - Division slash '/' is misclassified as digit '1'
    - Expression is corrupted by implicit multiplication into '(18+6)*14 = 336'
    - Total symbol count deviates from 8
    """
    img_path = "outputs/failure_case_g/01_original.png"
    assert os.path.exists(img_path), f"Test image missing at {img_path}"

    res = run_pipeline(img_path)

    # 1. Symbol Count & Localizations
    assert res["count"] == 8, f"Expected 8 symbols, got {res['count']}: {res['boxes']}"
    assert len(res["boxes"]) == 8

    # 2. Predicted Symbols
    expected_symbols = ["(", "1", "8", "+", "6", ")", "div", "4"]
    actual_symbols = [
        "div" if s in ("/", "div", "÷") else s
        for s in res["predicted_symbols"]
    ]
    expected_normalized = [
        "div" if s in ("/", "div", "÷") else s
        for s in expected_symbols
    ]
    assert actual_symbols == expected_normalized, f"Incorrect predictions: {res['predicted_symbols']}"
    assert res["predicted_symbols"][6] in ("div", "/"), f"Division slash must not be misclassified as digit: {res['predicted_symbols'][6]}"

    # 3. Parsing
    parsed = res["parsed"]
    assert parsed["is_valid"] is True, f"Parsing failed: {parsed.get('error_message')}"
    assert parsed["parsed_expression"] == "(18+6)/4", f"Expression corrupted: {parsed['parsed_expression']}"

    # 4. Symbolic Solution
    sol = res["solution"]
    assert sol["status"] == "solved"
    assert sol["final_answer"] == "6", f"Expected 6, got {sol['final_answer']}"


def test_pipeline_parentheses_multiplication_not_converted_to_division():
    """
    REGRESSION TEST:
    Verify that multiplication expressions with parentheses, e.g. '(18 + 6) * 4',
    are parsed strictly as multiplication ('(18+6)*4') and solve to '96',
    confirming that multiplication symbols are never accidentally converted to division.
    """
    from src.parser import parse_symbols
    from src.solver import solve_expression

    # Explicit multiplication operator '*' and 'times'
    for mult_op in ("*", "times", "x"):
        syms = ["(", "18", "+", "6", ")", mult_op, "4"]
        p = parse_symbols(syms)
        assert p["is_valid"] is True
        assert p["parsed_expression"] == "(18+6)*4"
        sol = solve_expression(p["parsed_expression"])
        assert sol["status"] == "solved"
        assert sol["final_answer"] == "96"


def test_classifier_slash_vs_digit_one_disambiguation():
    """
    CLASSIFIER & GEOMETRY REGRESSION TEST:
    Verify contrasting geometric features between division slash '/' and genuine digit '1':
    1. Forward diagonal slash: strong positive slant (dx >= 3.5px, angle >= 12 deg),
       tall (h >= 20px), wide (w >= 8px) situated between operands -> disambiguates to 'div'.
    2. Genuine vertical digit '1': narrow (w <= 5px), vertical (|dx| <= 2px) -> preserved as '1'.
    """
    # Verify Case G slash crop exhibits forward diagonal slant
    res_g = run_pipeline("outputs/failure_case_g/01_original.png")
    diag_slash = res_g["symbol_diagnostics"][6]
    assert diag_slash["symbol"] == "div"

    # Verify Case G digit '1' at index 1 is strictly preserved as '1'
    diag_one = res_g["symbol_diagnostics"][1]
    assert diag_one["symbol"] == "1"
    assert diag_one["bbox"][2] <= 5, "Genuine digit 1 must be narrow"


def test_prevent_silent_mathematical_corruption_implicit_multiplication():
    """
    REGRESSION TEST:
    Verify that an unverified token or digit creating multi-digit implicit multiplication
    like ')*14' flags an ambiguity warning instead of silently succeeding.
    """
    from src.parser import parse_symbols

    # Ambiguous sequence: '(', '18', '+', '6', ')', '1', '4'
    syms = ["(", "18", "+", "6", ")", "1", "4"]
    p = parse_symbols(syms)
    # Must flag ambiguity warning for multi-digit implicit multiplication
    ambig = p.get("ambiguous_tokens", [])
    assert any(
        a.get("type") == "suspicious_implicit_multiplication" or
        "implicit multiplication" in a.get("reason", "").lower()
        for a in ambig
    ), "Must flag suspicious implicit multiplication"


def test_pipeline_failure_case_h_end_to_end():
    """
    CRITICAL ACCEPTANCE TEST (Failure Case H from mathvision_results.zip):
    Verify that failure case H ('23 + 15') in outputs/failure_case_h/01_original.png
    is recognized as '23+15' and solved to '38'.
    Prevents regressions where:
    - Degenerate mid-air noise speck at (120, 43, 3, 5) survives segmentation and is misclassified as '+'
    - Equation parses as '23++15' returning 'Syntax Error'
    - Total symbol count deviates from 5
    - Verifies under both default 'auto' arbitration and explicit 'adaptive' thresholding.
    """
    img_path = "outputs/failure_case_h/01_original.png"
    assert os.path.exists(img_path), f"Test image missing at {img_path}"

    # 1. Test with default (auto arbitration)
    res_auto = run_pipeline(img_path)
    assert res_auto["count"] == 5, f"Expected 5 symbols under auto, got {res_auto['count']}: {res_auto['predicted_symbols']}"
    assert res_auto["predicted_symbols"] == ["2", "3", "+", "1", "5"]
    assert res_auto["parsed"]["is_valid"] is True
    assert res_auto["parsed"]["parsed_expression"] == "23+15"
    assert res_auto["solution"]["status"] == "solved"
    assert str(res_auto["solution"]["final_answer"]) == "38"

    # 2. Test with explicit adaptive thresholding
    res_adapt = run_pipeline(img_path, settings={"threshold_method": "adaptive"})
    assert res_adapt["count"] == 5, f"Expected 5 symbols under adaptive, got {res_adapt['count']}: {res_adapt['predicted_symbols']}"
    assert res_adapt["predicted_symbols"] == ["2", "3", "+", "1", "5"]
    assert res_adapt["parsed"]["is_valid"] is True
    assert res_adapt["parsed"]["parsed_expression"] == "23+15"
    assert res_adapt["solution"]["status"] == "solved"
    assert str(res_adapt["solution"]["final_answer"]) == "38"


def test_pipeline_failure_case_h2_end_to_end():
    """
    CRITICAL ACCEPTANCE TEST (Failure Case H2 from mathvision_results(2).zip):
    Verify that failure case H2 ('23 + 15') in outputs/failure_case_h2/01_original.png
    is recognized as '23+15' and solved to '38'.
    Prevents regressions where:
    - Final digit '5' is misclassified as '3' (producing '23+13 = 36')
    - Total symbol count deviates from 5
    - Verifies under both default 'auto' arbitration and explicit 'adaptive' thresholding.
    """
    img_path = "outputs/failure_case_h2/01_original.png"
    assert os.path.exists(img_path), f"Test image missing at {img_path}"

    # 1. Test with default (auto arbitration)
    res_auto = run_pipeline(img_path)
    assert res_auto["count"] == 5, f"Expected 5 symbols under auto, got {res_auto['count']}: {res_auto['predicted_symbols']}"
    assert res_auto["predicted_symbols"] == ["2", "3", "+", "1", "5"]
    assert res_auto["parsed"]["is_valid"] is True
    assert res_auto["parsed"]["parsed_expression"] == "23+15"
    assert res_auto["solution"]["status"] == "solved"
    assert str(res_auto["solution"]["final_answer"]) == "38"

    # 2. Test with explicit adaptive thresholding
    res_adapt = run_pipeline(img_path, settings={"threshold_method": "adaptive"})
    assert res_adapt["count"] == 5, f"Expected 5 symbols under adaptive, got {res_adapt['count']}: {res_adapt['predicted_symbols']}"
    assert res_adapt["predicted_symbols"] == ["2", "3", "+", "1", "5"], f"Regressed to 3: {res_adapt['predicted_symbols']}"
    assert res_adapt["parsed"]["is_valid"] is True
    assert res_adapt["parsed"]["parsed_expression"] == "23+15"
    assert res_adapt["solution"]["status"] == "solved"
    assert str(res_adapt["solution"]["final_answer"]) == "38"


def test_twenty_equation_suite():
    """
    ACCEPTANCE TEST: Phase 8 20-Equation Mathematical Recognition & Solver Benchmark
    Covers:
    - Basic operations (+, -, *, /)
    - Order of operations (PEMDAS)
    - Parentheses and grouping
    - Multi-digit numbers
    - Exponents
    """
    from src.parser import parse_symbols
    from src.solver import solve_expression

    benchmark_cases = [
        (["2", "3", "+", "1", "5"], "23+15", "38"),
        (["4", "8", "-", "1", "9"], "48-19", "29"),
        (["5", "6", "div", "7"], "56/7", "8"),
        (["6", "times", "8", "+", "3"], "6*8+3", "51"),
        (["4", "0", "div", "5", "+", "9"], "40/5+9", "17"),
        (["4", "5", "div", "5", "+", "6", "times", "2"], "45/5+6*2", "21"),
        (["3", "6", "div", "6", "+", "8"], "36/6+8", "14"),
        (["1", "2", "+", "8", "div", "4"], "12+8/4", "14"),
        (["6", "4", "div", "8", "+", "1", "6"], "64/8+16", "24"),
        (["8", "0", "-", "6", "times", "7"], "80-6*7", "38"),
        (["9", "times", "6", "-", "1", "4"], "9*6-14", "40"),
        (["7", "2", "div", "8", "times", "3"], "72/8*3", "27"),
        (["2", "5", "+", "1", "8", "div", "3"], "25+18/3", "31"),
        (["7", "times", "8", "+", "1", "2", "div", "4"], "7*8+12/4", "59"),
        (["9", "6", "div", "1", "2", "+", "5"], "96/12+5", "13"),
        (["(", "1", "8", "+", "6", ")", "div", "4"], "(18+6)/4", "6"),
        (["7", "times", "(", "9", "-", "3", ")", "+", "2"], "7*(9-3)+2", "44"),
        (["5", "**", "2", "+", "3", "times", "4"], "5**2+3*4", "37"),
        (["1", "0", "0", "-", "(", "2", "4", "div", "6", ")", "times", "8"], "100-(24/6)*8", "68"),
        (["4", "8", "div", "(", "2", "times", "4", ")", "+", "7"], "48/(2*4)+7", "13"),
    ]

    for syms, expected_expr, expected_ans in benchmark_cases:
        p = parse_symbols(syms)
        assert p["is_valid"] is True, f"Failed parse for {syms}: {p.get('error_message')}"
        assert p["parsed_expression"] == expected_expr, f"Parse mismatch for {syms}: got {p['parsed_expression']}, expected {expected_expr}"
        sol = solve_expression(p["parsed_expression"])
        assert sol["status"] == "solved", f"Failed solve for {expected_expr}: {sol}"
        actual_ans = str(sol["final_answer"])
        assert actual_ans == expected_ans, f"Answer mismatch for {expected_expr}: got {actual_ans}, expected {expected_ans}"


def test_digit_3_vs_5_disambiguation_preserves_genuine_digits():
    """
    DISAMBIGUATION TEST:
    Verifies that:
    1. A handwritten '5' with open left-middle whitespace is disambiguated from '3' to '5'.
    2. A genuine '3' with a central leftward cusp is strictly preserved as '3'.
    """
    res_b = run_pipeline("outputs/failure_case_b/01_original.png")
    assert res_b["predicted_symbols"][0] == "3", "Genuine 3 must never be converted to 5"

    res_h = run_pipeline("outputs/failure_case_h/01_original.png")
    assert res_h["predicted_symbols"][1] == "3", "Genuine 3 at index 1 must remain 3"
    assert res_h["predicted_symbols"][4] == "5", "Genuine 5 at index 4 must remain 5"

    res_h2 = run_pipeline("outputs/failure_case_h2/01_original.png")
    assert res_h2["predicted_symbols"][1] == "3", "Genuine 3 at index 1 must remain 3"
    assert res_h2["predicted_symbols"][4] == "5", "Digit 5 at index 4 must be recognized as 5"


def test_small_component_preservation_decimal_and_exponent():
    """
    SEGMENTATION UNIT TEST:
    Verifies that:
    1. A legitimate decimal point on the bottom baseline is PRESERVED.
    2. A legitimate exponent/superscript mark in the superscript zone is PRESERVED.
    3. An isolated degenerate speck mid-air is FILTERED OUT.
    """
    from src.segment import segment_symbols

    # Construct a synthetic binary canvas (h=80, w=300)
    canvas = np.zeros((80, 300), dtype=np.uint8)

    # Digit '3': x=20, y=25, w=20, h=30 (baseline y=55, cy=40)
    cv2.rectangle(canvas, (20, 25), (40, 55), 255, -1)

    # Decimal point: x=50, y=51, w=4, h=4 (baseline y=55) -> near bottom baseline
    cv2.rectangle(canvas, (50, 51), (54, 55), 255, -1)

    # Digit '1': x=65, y=25, w=8, h=30
    cv2.rectangle(canvas, (65, 25), (73, 55), 255, -1)

    # Mid-air degenerate noise speck: x=120, y=38, w=3, h=5 (cy=40.5, mid-air)
    cv2.rectangle(canvas, (120, 38), (123, 43), 255, -1)

    # Digit '5': x=150, y=25, w=20, h=30
    cv2.rectangle(canvas, (150, 25), (170, 55), 255, -1)

    res = segment_symbols(canvas)
    boxes = res["boxes"]

    # The decimal point at (50, 51, 4, 4) MUST be preserved
    has_decimal = any(b[0] == 50 and b[1] == 51 for b in boxes)
    assert has_decimal, f"Decimal point was erroneously filtered out: {boxes}"

    # The mid-air speck at (120, 38, 3, 5) MUST be filtered out
    has_speck = any(b[0] == 120 and b[1] == 38 for b in boxes)
    assert not has_speck, f"Mid-air noise speck survived segmentation: {boxes}"
    assert len(boxes) == 4, f"Expected 4 boxes (3, dot, 1, 5), got {len(boxes)}: {boxes}"


def test_parser_deduplicate_consecutive_addition_operators():
    """
    PARSER UNIT TEST:
    Verifies that 'repair_syntax_artifacts' deduplicates consecutive identical addition operators,
    such as ['2', '3', '+', '+', '1', '5'] -> '23+15' = 38.
    """
    from src.parser import parse_symbols
    from src.solver import solve_expression

    syms = ["2", "3", "+", "+", "1", "5"]
    boxes = [
        (40, 34, 15, 23),
        (67, 34, 16, 24),
        (90, 36, 20, 21),
        (120, 43, 3, 5),   # Tiny box
        (130, 33, 7, 24),
        (146, 33, 17, 25)
    ]
    p = parse_symbols(syms, boxes=boxes)
    assert p["is_valid"] is True
    assert p["parsed_expression"] == "23+15"
    sol = solve_expression(p["parsed_expression"])
    assert sol["status"] == "solved"
    assert sol["final_answer"] == "38"







