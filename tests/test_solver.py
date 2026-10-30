"""
=============================================================================
Unit Tests for MathVision Step-by-Step Solver
=============================================================================
"""

import pytest
import sympy as sp
from src.solver import (
    solve_expression,
    solve_arithmetic_steps,
    solve_linear_steps,
    solve_quadratic_steps,
    safe_latex,
    safe_str,
    validate_expression_safety,
    is_too_large_number,
    TOO_LARGE_MSG_STR,
    TOO_LARGE_MSG_LATEX,
)


def test_solve_arithmetic_simple():
    """Verify standard arithmetic evaluation."""
    res = solve_expression("2 + 3 * 4")
    assert res["status"] == "solved"
    assert res["equation_type"] == "arithmetic"
    assert res["final_answer"] == "14"
    assert len(res["steps"]) >= 2


def test_solve_arithmetic_parentheses():
    """Verify PEMDAS order of operations with parentheses."""
    res = solve_expression("(8 - 2) / 2")
    assert res["status"] == "solved"
    assert res["equation_type"] == "arithmetic"
    assert res["final_answer"] == "3"


def test_solve_linear_equation():
    """Verify linear equation solving and isolation steps."""
    res = solve_expression("3*x + 5 = 20")
    assert res["status"] == "solved"
    assert res["equation_type"] == "linear"
    assert res["final_answer"] == "x = 5"
    assert res["roots"] == ["5"]
    assert len(res["steps"]) >= 3


def test_solve_linear_equation_negative_coeff():
    """Verify linear equation with subtraction."""
    res = solve_expression("7 - 2*x = 1")
    assert res["status"] == "solved"
    assert res["equation_type"] == "linear"
    assert res["final_answer"] == "x = 3"
    assert res["roots"] == ["3"]


def test_solve_quadratic_equation_distinct_roots():
    """Verify quadratic equation solving with real distinct roots."""
    res = solve_expression("x**2 - 5*x + 6 = 0")
    assert res["status"] == "solved"
    assert res["equation_type"] == "quadratic"
    assert set(res["roots"]) == {"2", "3"}
    assert len(res["steps"]) >= 4

    # Check that discriminant step exists
    discrim_step = next((s for s in res["steps"] if "Discriminant" in s["title"]), None)
    assert discrim_step is not None
    assert "\\Delta" in discrim_step["latex"]


def test_solve_unsolvable_contradiction():
    """Verify contradiction linear equation returns unsolvable status."""
    res = solve_expression("x + 1 = x + 2")
    assert res["status"] == "unsolvable"
    assert "No solution" in res["final_answer"]


def test_solve_algebraic_expression_no_equality():
    """
    REGRESSION TEST:
    An expression containing variable 'x' without an '=' sign (e.g. '1*x**48/4' or '2*x + 5')
    must be classified as 'algebraic_expression' with status 'simplified', NOT as solved arithmetic.
    """
    res = solve_expression("1*x**48/4")
    assert res["status"] == "simplified"
    assert res["equation_type"] == "algebraic_expression"
    assert "Simplified" in res["final_answer"]


def test_oversized_integer_result_handled_safely():
    """
    REGRESSION TEST 1:
    Verifies that an expression evaluating to an integer with >4300 digits
    (which previously crashed with ValueError: Exceeds the limit (4300 digits) for integer string conversion)
    is handled safely without raising an unhandled exception.
    """
    # Test directly inside solve_arithmetic_steps using a constructed huge expression
    # Even if fed directly into solve_arithmetic_steps, it must return status 'error' and not crash
    res = solve_arithmetic_steps("2**15000")
    assert res["status"] == "error"
    assert res["equation_type"] == "arithmetic"
    assert TOO_LARGE_MSG_STR in res["error_message"]
    assert res["final_answer"] == TOO_LARGE_MSG_STR
    assert "\\text" in res["final_answer_latex"]


def test_solve_normal_arithmetic_12_plus_8_div_4():
    """
    REGRESSION TEST 2:
    Verifies that ordinary calculations such as '12 + 8/4' continue to evaluate
    correctly to 14 with full step-by-step formatting:
      - Initial: 12 + 8 ÷ 4
      - Division first: 8 ÷ 4 = 2
      - Addition next: 12 + 2 = 14
      - Final: 14
    """
    res = solve_expression("12 + 8/4")
    assert res["status"] == "solved"
    assert res["equation_type"] == "arithmetic"
    assert res["final_answer"] == "14"
    assert len(res["steps"]) == 3
    # Step 1: Initial Expression
    assert "12 + 8 \\div 4" in res["steps"][0]["latex"]
    # Step 2: Division First
    assert "Division First" in res["steps"][1]["title"]
    assert "8 \\div 4 = 2" in res["steps"][1]["latex"]
    # Step 3: Addition Next
    assert "Addition Next" in res["steps"][2]["title"]
    assert "12 + 2 = 14" in res["steps"][2]["latex"]
    assert "14" in res["steps"][-1]["latex"]


def test_solve_normal_linear_equation_2x_plus_5_eq_15():
    """
    REGRESSION TEST 3:
    Verifies that a standard linear equation such as '2*x + 5 = 15'
    returns x = 5 with status 'solved'.
    """
    res = solve_expression("2*x + 5 = 15")
    assert res["status"] == "solved"
    assert res["equation_type"] == "linear"
    assert res["final_answer"] == "x = 5"
    assert res["roots"] == ["5"]


def test_excessive_exponent_and_huge_literal_rejected_upfront():
    """
    REGRESSION TEST 4:
    Verifies that expressions with excessive exponents or oversized numeric literals
    are rejected upfront with structured errors before expensive symbolic evaluation.
    """
    # 1. Excessive exponent
    res_exp = solve_expression("2**15000")
    assert res_exp["status"] == "error"
    assert res_exp["equation_type"] == "oversized_expression"
    assert "Exponent magnitude" in res_exp["error_message"]

    res_exp2 = solve_expression("8**5000")
    assert res_exp2["status"] == "error"
    assert "Exponent magnitude" in res_exp2["error_message"]

    # 2. Oversized numeric literal (>10 digits)
    res_lit = solve_expression("1234567890123 + 4")
    assert res_lit["status"] == "error"
    assert "Numeric literal" in res_lit["error_message"]

    # 3. Chained exponentiation
    res_chain = solve_expression("2**3**4")
    assert res_chain["status"] == "error"
    assert "Chained exponentiation" in res_chain["error_message"]

    # 4. Excessive expression length
    res_len = solve_expression("1+" * 105 + "1")
    assert res_len["status"] == "error"
    assert "Expression length" in res_len["error_message"]


def test_safe_latex_and_safe_str_never_raise_value_error():
    """
    REGRESSION TEST 5:
    Verifies that safe_latex() and safe_str() never raise ValueError on huge integers,
    rationals, or expressions, while preserving exact output for normal numbers.
    """
    # Normal numbers
    assert safe_str(sp.Integer(14)) == "14"
    assert safe_latex(sp.Integer(14)) == "14"
    assert safe_str(sp.Rational(3, 2)) == "3/2"
    assert safe_latex(sp.Rational(3, 2)) == "\\frac{3}{2}"

    # Huge integer (>4300 digits)
    huge_int = sp.Integer(2)**15000
    assert is_too_large_number(huge_int) is True
    # Calling safe_str and safe_latex MUST NOT raise ValueError
    str_res = safe_str(huge_int)
    latex_res = safe_latex(huge_int)
    assert str_res == TOO_LARGE_MSG_STR
    assert "\\text" in latex_res
    assert TOO_LARGE_MSG_STR in latex_res

    # Huge rational
    huge_rat = sp.Rational(sp.Integer(2)**15000, 3)
    assert is_too_large_number(huge_rat) is True
    assert safe_str(huge_rat) == TOO_LARGE_MSG_STR
    assert safe_latex(huge_rat) == TOO_LARGE_MSG_LATEX


