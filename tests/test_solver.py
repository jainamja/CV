"""
=============================================================================
Unit Tests for MathVision Step-by-Step Solver
=============================================================================
"""

import pytest
from src.solver import (
    solve_expression,
    solve_arithmetic_steps,
    solve_linear_steps,
    solve_quadratic_steps,
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
