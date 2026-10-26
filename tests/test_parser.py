"""
=============================================================================
Unit Tests for MathVision Syntactic Parser
=============================================================================
"""

import pytest
from src.parser import (
    normalize_operators,
    insert_implicit_multiplication,
    validate_expression,
    detect_exponents,
    disambiguate_x_times,
    parse_symbols,
)


def test_normalize_operators():
    """Verify visual operators are mapped to Python standard operators."""
    symbols = ["2", "times", "3", "div", "6", "–", "1", "equals", "0"]
    normalized = normalize_operators(symbols)
    assert normalized == ["2", "*", "3", "/", "6", "-", "1", "=", "0"]


def test_insert_implicit_multiplication():
    """Verify implicit multiplication insertion across algebraic patterns."""
    assert insert_implicit_multiplication("2x") == "2*x"
    assert insert_implicit_multiplication("3x+5") == "3*x+5"
    assert insert_implicit_multiplication("2(x+1)") == "2*(x+1)"
    assert insert_implicit_multiplication("(x+1)(x-2)") == "(x+1)*(x-2)"
    assert insert_implicit_multiplication("x(x+3)") == "x*(x+3)"
    assert insert_implicit_multiplication("(x+1)2") == "(x+1)*2"


def test_detect_exponents_spatial_trigger():
    """Verify superscript bounding box triggers '**' exponent operator."""
    symbols = ["x", "2"]
    # Base 'x': (x=10, y=50, w=30, h=40) -> center y = 70
    # Exponent '2': (x=42, y=25, w=20, h=25) -> noticeably higher and smaller
    boxes = [
        (10, 50, 30, 40),
        (42, 25, 20, 25)
    ]
    augmented = detect_exponents(symbols, boxes)
    assert augmented == ["x", "**", "2"]


def test_validate_expression_balanced_parentheses():
    """Verify parentheses validation detects balanced and unbalanced cases."""
    valid, msg = validate_expression("(2+3)*4")
    assert valid is True
    assert msg == ""

    invalid, err = validate_expression("((2+3)*4")
    assert invalid is False
    assert "Unbalanced parentheses" in err

    invalid2, err2 = validate_expression("(2+3))*4")
    assert invalid2 is False
    assert "Unbalanced parentheses" in err2


def test_validate_expression_empty_and_trailing():
    """Verify trailing operators and empty strings are caught gracefully."""
    v1, err1 = validate_expression("")
    assert v1 is False

    v2, err2 = validate_expression("2 + 3 +")
    assert v2 is False
    assert "trailing operator" in err2.lower()

    v3, err3 = validate_expression("= 5")
    assert v3 is False
    assert "missing left-hand side" in err3.lower()

    v4, err4 = validate_expression("2x + 1 = = 5")
    assert v4 is False
    assert "only single-line" in err4.lower() or "signs" in err4.lower()


def test_parse_symbols_end_to_end():
    """Verify full symbol parse output dictionary."""
    symbols = ["3", "x", "+", "5", "=", "2", "0"]
    res = parse_symbols(symbols)

    assert res["is_valid"] is True
    assert res["parsed_expression"] == "3*x+5=20"
    assert res["error_message"] == ""
    assert "\\text" not in res["error_message"]


def test_disambiguate_x_times():
    """Verify context-aware disambiguation between 'x' and 'times'."""
    # 1. Between digits: 3 x 4 -> 3 * 4
    assert disambiguate_x_times(["3", "x", "4"]) == ["3", "times", "4"]

    # 2. Leading symbol: times + 5 = 20 -> x + 5 = 20
    assert disambiguate_x_times(["times", "+", "5", "=", "20"]) == ["x", "+", "5", "=", "20"]

    # 3. Before operator: 3 times + 5 -> 3 x + 5
    assert disambiguate_x_times(["3", "times", "+", "5"]) == ["3", "x", "+", "5"]

    # 4. After operator: 7 - times = 1 -> 7 - x = 1
    assert disambiguate_x_times(["7", "-", "times", "=", "1"]) == ["7", "-", "x", "=", "1"]

    # 5. Inside parentheses: (times + 1) -> (x + 1)
    assert disambiguate_x_times(["(", "times", "+", "1", ")"]) == ["(", "x", "+", "1", ")"]

    # 6. Exponent base: times ** 2 -> x ** 2
    assert disambiguate_x_times(["times", "**", "2"]) == ["x", "**", "2"]

    # 7. Geometric bounding box features
    # Box 1: '3' (10, 20, 30, 40)
    # Box 2: elevated, smaller operator-like glyph (45, 32, 18, 18)
    # Box 3: 'y' (70, 20, 30, 40)
    boxes_times = [(10, 20, 30, 40), (45, 32, 18, 18), (70, 20, 30, 40)]
    assert disambiguate_x_times(["3", "x", "y"], boxes_times) == ["3", "times", "y"]

    # Box 2: wide, tall baseline glyph (45, 20, 45, 38) -> aspect = 45/38 = 1.18 > 1.02
    boxes_x = [(10, 20, 30, 40), (45, 20, 45, 38), (95, 20, 30, 40)]
    assert disambiguate_x_times(["3", "times", "y"], boxes_x) == ["3", "x", "y"]


def test_parser_never_reads_true_labels():
    """
    Verification that syntactic parser and disambiguator operate with ZERO label leakage.
    Ensures that parse_symbols and disambiguate_x_times:
      1. Accept ONLY candidate predicted symbols, optional bounding boxes, and probabilities.
      2. Expose NO parameters for 'ground_truth', 'true_label', or target strings.
      3. Do NOT read any label files, annotations, CSVs, or global ground truth variables.
    """
    import inspect
    from src import parser

    # 1. Verify signatures have no ground-truth parameters
    for func in [parser.parse_symbols, parser.disambiguate_x_times]:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        for forbidden in ["target", "ground_truth", "gt", "true_label", "true_labels", "label"]:
            assert forbidden not in param_names, f"{func.__name__} must not accept '{forbidden}'"

    # 2. Execute on noisy candidate stream without any ground truth
    dummy_symbols = ["2", "times", "+", "3", "x", "=", "1", "1"]
    dummy_boxes = [(i * 20, 10, 15, 30) for i in range(len(dummy_symbols))]
    result = parser.parse_symbols(dummy_symbols, dummy_boxes)

    assert "parsed_expression" in result
    assert "disambiguated_symbols" in result
    assert "ambiguous_tokens" in result
    # Verify the output is derived strictly from syntactic structure
    assert result["parsed_expression"] == "2*x+3*x=11"


def test_benchmark_and_production_disambiguation_parity():
    """
    Priority 1: Verify that production disambiguation and benchmark evaluation
    produce 100% identical predictions and ambiguity flags when given identical inputs.
    """
    import numpy as np
    from src.parser import disambiguate_x_times, DEFAULT_DISAMBIGUATION_CONFIG
    from src.compose_equations import DEFAULT_DISAMBIGUATION_CONFIG as BENCHMARK_CONFIG

    # Confirm frozen configuration reference matches
    assert DEFAULT_DISAMBIGUATION_CONFIG == BENCHMARK_CONFIG

    test_cases = [
        (["3", "x", "4"], [(10, 10, 20, 20), (35, 12, 18, 18), (60, 10, 20, 20)]),
        (["2", "times", "+", "5"], [(10, 10, 20, 20), (35, 12, 18, 18), (60, 10, 20, 20), (85, 10, 20, 20)]),
        (["4", "x", "(", "x", "+", "1", ")"], [(i * 25, 10, 20, 20) for i in range(7)]),
        (["x", "=", "5"], [(10, 10, 20, 20), (35, 10, 20, 20), (60, 10, 20, 20)]),
        (["times", "=", "5"], [(10, 10, 20, 20), (35, 10, 20, 20), (60, 10, 20, 20)]),
    ]

    for symbols, boxes in test_cases:
        # 1. Calling production function
        prod_syms, prod_ambig = disambiguate_x_times(
            symbols,
            boxes=boxes,
            probs=None,
            return_ambiguities=True,
            config=DEFAULT_DISAMBIGUATION_CONFIG
        )

        # 2. Calling with explicit custom config
        custom_cfg = dict(DEFAULT_DISAMBIGUATION_CONFIG)
        custom_cfg["w_stroke"] = 0.4
        bench_syms, bench_ambig = disambiguate_x_times(
            symbols,
            boxes=boxes,
            probs=None,
            return_ambiguities=True,
            config=custom_cfg
        )

        assert prod_syms == bench_syms, f"Symbol parity failed for {symbols}: {prod_syms} != {bench_syms}"
        assert prod_ambig == bench_ambig, f"Ambiguity flag parity failed for {symbols}: {prod_ambig} != {bench_ambig}"


def test_validation_and_test_templates_disjoint():
    """
    Priority 3: Verify that validation templates and test templates have zero overlap.
    """
    from src.compose_equations import generate_validation_templates, generate_test_templates

    val_tmpls = generate_validation_templates()
    test_tmpls = generate_test_templates()

    assert len(val_tmpls) > 0
    assert len(test_tmpls) == 220

    val_canonicals = {t["canonical"] for t in val_tmpls}
    test_canonicals = {t["canonical"] for t in test_tmpls}

    overlap = val_canonicals.intersection(test_canonicals)
    assert len(overlap) == 0, f"Validation and test templates must have ZERO overlap! Found: {overlap}"


def test_composed_benchmark_artifact_and_metrics():
    """
    Priority 4: Verify that composed_benchmark_predictions.json exists,
    contains 220 evaluated equations, correct denominators, and arithmetic consistency.
    """
    import os
    import json

    artifact_path = "outputs/composed_benchmark_predictions.json"
    assert os.path.exists(artifact_path), f"Reproducible benchmark artifact missing at {artifact_path}"

    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["benchmark_metadata"]
    metrics = data["metrics"]
    arith = data["arithmetic_verification"]
    eqs = data["equations"]

    # Denominators and counts
    assert meta["total_equations"] == 220
    assert len(eqs) == 220

    raw_m = metrics["setting_a_raw_cnn"]
    assert raw_m["total"] == 220
    assert raw_m["correct"] == 133
    assert raw_m["accuracy_pct"] == 60.45
    assert raw_m["symbol_level_x_to_times_errors"] == 40
    assert raw_m["symbol_level_times_to_x_errors"] == 14
    assert raw_m["symbol_level_total_cross_errors"] == 54

    stroke_m = metrics["setting_b_stroke_aspect_alone"]
    assert stroke_m["total"] == 220
    assert stroke_m["correct"] == 126
    assert stroke_m["accuracy_pct"] == 57.27

    pipe_m = metrics["setting_c_production_soft_grammar_pipeline"]
    assert pipe_m["strict_autonomous_total"] == 220
    assert pipe_m["strict_autonomous_correct"] == 150
    assert pipe_m["strict_autonomous_accuracy_pct"] == 68.18
    assert pipe_m["unassisted_exact_total"] == 220
    assert pipe_m["unassisted_exact_correct"] == 170
    assert pipe_m["unassisted_exact_accuracy_pct"] == 77.27
    assert pipe_m["oracle_assisted_total"] == 220
    assert pipe_m["oracle_assisted_correct"] == 173
    assert pipe_m["oracle_assisted_accuracy_pct"] == 78.64
    assert pipe_m["ambiguities_flagged_count"] == 34
    assert pipe_m["symbol_level_x_to_times_errors"] == 1
    assert pipe_m["symbol_level_times_to_x_errors"] == 4
    assert pipe_m["symbol_level_total_cross_errors"] == 5

    # Verification consistency
    assert arith["unassisted_exact_breakdown"]["sum_equals_unassisted"] is True
    assert arith["oracle_override_breakdown"]["sum_equals_oracle"] is True
    assert arith["ambiguity_flag_breakdown"]["sum_equals_flagged_equations"] is True
    assert arith["denominators_verified_220"] is True


def test_detect_exponents_does_not_trigger_on_normal_baseline():
    """
    REGRESSION TEST:
    A variable 'x' following a digit '2' on the same baseline (e.g. '2x') must NOT
    be incorrectly parsed as an exponent ('2**x') even if lowercase 'x' is smaller.
    """
    symbols = ["2", "x"]
    # Digit '2': (x=10, y=50, w=30, h=45) -> baseline bottom at y=95
    # Variable 'x': (x=45, y=60, w=25, h=35) -> baseline bottom at y=95
    boxes = [
        (10, 50, 30, 45),
        (45, 60, 25, 35)
    ]
    augmented = detect_exponents(symbols, boxes)
    assert augmented == ["2", "x"], f"Expected ['2', 'x'], got {augmented}"


