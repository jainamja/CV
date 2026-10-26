"""
=============================================================================
MathVision: Step-by-Step Symbolic Mathematical Solver
=============================================================================

This module evaluates mathematical expressions and equations using SymPy,
generating transparent, pedagogical, step-by-step working suitable for
academic viva presentation and classroom explanation.

Supported Problem Classes:
    1. Plain Arithmetic Expressions:
       Evaluates order of operations (PEMDAS / BODMAS) step-by-step:
       Parentheses -> Exponents -> Multiplication/Division -> Addition/Subtraction.
    2. Linear Algebraic Equations (Degree 1 in 'x'):
       Demonstrates algebraic isolation:
       - Transpose variable terms to LHS and constant terms to RHS
       - Combine like terms
       - Divide both sides by the leading coefficient
    3. Quadratic Equations (Degree 2 in 'x'):
       Demonstrates rigorous quadratic formula derivation:
       - Rearrange to standard form: a*x^2 + b*x + c = 0
       - Extract coefficients a, b, c
       - Compute discriminant: Delta = b^2 - 4ac
       - Evaluate nature of roots (real distinct, real repeated, or complex)
       - Apply quadratic formula: x = (-b +- sqrt(Delta)) / (2a)
=============================================================================
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import sympy as sp
from sympy import symbols, Eq, solve, sympify, latex, degree, Poly, simplify, Rational


x = symbols("x")


def solve_arithmetic_steps(expr_str: str) -> Dict[str, Any]:
    """
    Evaluates arithmetic expression with step-by-step order of operations (PEMDAS).
    """
    try:
        expr = sympify(expr_str)
    except Exception as e:
        return {
            "status": "error",
            "equation_type": "arithmetic",
            "error_message": f"Could not parse arithmetic expression: {str(e)}",
            "steps": [],
            "final_answer": "",
            "final_answer_latex": ""
        }

    steps: List[Dict[str, Any]] = []
    step_num = 1

    # Step 1: Initial Expression
    steps.append({
        "step_number": step_num,
        "title": "Initial Expression",
        "description": "Identify the arithmetic expression to evaluate.",
        "latex": f"\\text{{Evaluate: }} {expr_str}"
    })
    step_num += 1

    # Step 2: Simplified Result
    simplified_result = simplify(expr)
    steps.append({
        "step_number": step_num,
        "title": "Apply Order of Operations (PEMDAS)",
        "description": "Evaluate exponents and multiplications/divisions first, followed by additions and subtractions.",
        "latex": f"= {sp.latex(simplified_result)}"
    })

    # Format final answer
    if simplified_result.is_integer:
        final_ans_str = str(int(simplified_result))
        final_ans_latex = str(int(simplified_result))
    elif simplified_result.is_rational:
        val_float = float(simplified_result)
        final_ans_str = f"{simplified_result} (approx {val_float:.4f})"
        final_ans_latex = f"{sp.latex(simplified_result)} \\approx {val_float:.4f}"
    else:
        final_ans_str = str(simplified_result)
        final_ans_latex = sp.latex(simplified_result)

    return {
        "status": "solved",
        "equation_type": "arithmetic",
        "steps": steps,
        "final_answer": final_ans_str,
        "final_answer_latex": final_ans_latex,
        "roots": []
    }


def solve_linear_steps(lhs_expr: sp.Expr, rhs_expr: sp.Expr, var_sym: sp.Symbol = x) -> Dict[str, Any]:
    """
    Generates step-by-step algebraic isolation working for linear equations.
    """
    steps: List[Dict[str, Any]] = []
    step_num = 1

    # Step 1: State original equation
    orig_eq = Eq(lhs_expr, rhs_expr)
    steps.append({
        "step_number": step_num,
        "title": "Given Linear Equation",
        "description": "State the equation to solve for variable x.",
        "latex": sp.latex(orig_eq)
    })
    step_num += 1

    # Expand both sides
    lhs_exp = sp.expand(lhs_expr)
    rhs_exp = sp.expand(rhs_expr)
    if lhs_exp != lhs_expr or rhs_exp != rhs_expr:
        steps.append({
            "step_number": step_num,
            "title": "Expand Expressions",
            "description": "Expand any parentheses or products on both sides of the equation.",
            "latex": sp.latex(Eq(lhs_exp, rhs_exp))
        })
        step_num += 1

    # Step: Collect all terms onto LHS: P(x) = LHS - RHS = 0
    diff_expr = sp.expand(lhs_exp - rhs_exp)
    poly = Poly(diff_expr, var_sym)
    coeffs = poly.all_coeffs()

    if len(coeffs) == 2:
        a = coeffs[0]  # Coefficient of x
        b = coeffs[1]  # Constant term (a*x + b = 0)
    elif len(coeffs) == 1:
        a = coeffs[0]
        b = sp.Integer(0)
    else:
        # Trivial or degenerate
        if diff_expr == 0:
            return {
                "status": "solved",
                "equation_type": "linear",
                "steps": steps,
                "final_answer": "Identity: Infinite solutions (all x in R)",
                "final_answer_latex": "x \\in \\mathbb{R}",
                "roots": []
            }
        else:
            return {
                "status": "unsolvable",
                "equation_type": "linear",
                "steps": steps,
                "final_answer": "Contradiction: No solution",
                "final_answer_latex": "\\emptyset",
                "roots": []
            }

    # Step: Move constant to RHS
    # a*x = -b
    rhs_constant = -b
    steps.append({
        "step_number": step_num,
        "title": "Isolate Variable Terms",
        "description": f"Move constant terms to the right-hand side by subtracting {b} from both sides.",
        "latex": f"{sp.latex(a * var_sym)} = {sp.latex(rhs_constant)}"
    })
    step_num += 1

    # Step: Divide by leading coefficient
    if a != 1:
        steps.append({
            "step_number": step_num,
            "title": f"Divide Both Sides by Coefficient ({a})",
            "description": f"Divide both sides by the variable's coefficient ({a}) to solve for x.",
            "latex": f"{sp.latex(var_sym)} = \\frac{{{sp.latex(rhs_constant)}}}{{{sp.latex(a)}}}"
        })
        step_num += 1

    # Final Root
    root = simplify(rhs_constant / a)
    steps.append({
        "step_number": step_num,
        "title": "Solution Found",
        "description": "Simplify the fraction to obtain the final value of x.",
        "latex": f"x = {sp.latex(root)}"
    })

    root_str = str(int(root)) if root.is_integer else str(root)

    return {
        "status": "solved",
        "equation_type": "linear",
        "steps": steps,
        "final_answer": f"x = {root_str}",
        "final_answer_latex": f"x = {sp.latex(root)}",
        "roots": [root_str]
    }


def solve_quadratic_steps(lhs_expr: sp.Expr, rhs_expr: sp.Expr, var_sym: sp.Symbol = x) -> Dict[str, Any]:
    """
    Generates step-by-step quadratic formula working with discriminant analysis.
    """
    steps: List[Dict[str, Any]] = []
    step_num = 1

    # Step 1: State original equation
    steps.append({
        "step_number": step_num,
        "title": "Given Quadratic Equation",
        "description": "State the second-degree equation to solve for variable x.",
        "latex": sp.latex(Eq(lhs_expr, rhs_expr))
    })
    step_num += 1

    # Step 2: Standard Form a*x^2 + b*x + c = 0
    diff = sp.expand(lhs_expr - rhs_expr)
    poly = Poly(diff, var_sym)
    coeffs = poly.all_coeffs()

    if len(coeffs) == 3:
        a_val, b_val, c_val = coeffs[0], coeffs[1], coeffs[2]
    elif len(coeffs) == 2:
        a_val, b_val, c_val = coeffs[0], coeffs[1], sp.Integer(0)
    elif len(coeffs) == 1:
        a_val, b_val, c_val = coeffs[0], sp.Integer(0), sp.Integer(0)
    else:
        return solve_linear_steps(lhs_expr, rhs_expr, var_sym)

    steps.append({
        "step_number": step_num,
        "title": "Rearrange to Standard Form",
        "description": "Subtract the right-hand side from both sides to form a x^2 + b x + c = 0.",
        "latex": f"{sp.latex(diff)} = 0"
    })
    step_num += 1

    # Step 3: Identify Coefficients
    steps.append({
        "step_number": step_num,
        "title": "Identify Coefficients",
        "description": "Extract the quadratic, linear, and constant coefficients.",
        "latex": f"a = {a_val}, \\quad b = {b_val}, \\quad c = {c_val}"
    })
    step_num += 1

    # Step 4: Compute Discriminant
    # Delta = b^2 - 4ac
    delta = simplify(b_val**2 - 4 * a_val * c_val)
    if delta > 0:
        delta_nature = "\\Delta > 0 \\implies \\text{Two distinct real roots}"
    elif delta == 0:
        delta_nature = "\\Delta = 0 \\implies \\text{One repeated real root}"
    else:
        delta_nature = "\\Delta < 0 \\implies \\text{Two complex conjugate roots}"

    steps.append({
        "step_number": step_num,
        "title": "Calculate Discriminant (Delta)",
        "description": "Compute the discriminant Delta = b^2 - 4ac to determine the nature of the roots.",
        "latex": f"\\Delta = ({b_val})^2 - 4({a_val})({c_val}) = {delta} \\quad ({delta_nature})"
    })
    step_num += 1

    # Step 5: Apply Quadratic Formula
    steps.append({
        "step_number": step_num,
        "title": "Apply Quadratic Formula",
        "description": "Substitute a, b, and Delta into the quadratic formula x = (-b +- sqrt(Delta)) / (2a).",
        "latex": f"x = \\frac{{-({b_val}) \\pm \\sqrt{{{delta}}}}}{{2({a_val})}}"
    })
    step_num += 1

    # Step 6: Compute Individual Roots
    roots_list = solve(Eq(diff, 0), var_sym)
    roots_str_list = []
    roots_latex_list = []

    for idx, r in enumerate(roots_list):
        r_simp = simplify(r)
        roots_str_list.append(str(r_simp))
        roots_latex_list.append(f"x_{{{idx + 1}}} = {sp.latex(r_simp)}")

    combined_roots_latex = ", \\quad ".join(roots_latex_list)
    steps.append({
        "step_number": step_num,
        "title": "Compute Roots",
        "description": "Evaluate the plus and minus branches to find all roots.",
        "latex": combined_roots_latex
    })

    return {
        "status": "solved",
        "equation_type": "quadratic",
        "steps": steps,
        "final_answer": ", ".join([f"x = {s}" for s in roots_str_list]),
        "final_answer_latex": combined_roots_latex,
        "roots": roots_str_list
    }


def solve_expression(expr_str: str) -> Dict[str, Any]:
    """
    Main entry point for solving any parsed mathematical expression or equation.

    Dispatches to:
        - Arithmetic Solver (if no variable and no equality)
        - Linear Equation Solver (if degree 1 in x)
        - Quadratic Equation Solver (if degree 2 in x)
        - General SymPy Algebraic Solver (fallback for other forms)
    """
    clean_expr = expr_str.strip()
    if not clean_expr:
        return {
            "status": "error",
            "equation_type": "unknown",
            "error_message": "Expression string is empty.",
            "steps": [],
            "final_answer": "",
            "final_answer_latex": "",
            "roots": []
        }

    # Case 1: Equation with '='
    if "=" in clean_expr:
        parts = clean_expr.split("=")
        if len(parts) != 2:
            return {
                "status": "error",
                "equation_type": "equation",
                "error_message": "Invalid equation: multiple '=' signs found.",
                "steps": [],
                "final_answer": "",
                "final_answer_latex": "",
                "roots": []
            }

        lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
        try:
            lhs_sym = sympify(lhs_str)
            rhs_sym = sympify(rhs_str)
        except Exception as e:
            return {
                "status": "error",
                "equation_type": "equation",
                "error_message": f"Syntax error in equation sides: {str(e)}",
                "steps": [],
                "final_answer": "",
                "final_answer_latex": "",
                "roots": []
            }

        # Check for variable x
        diff = simplify(lhs_sym - rhs_sym)
        free_symbols = diff.free_symbols

        if x in free_symbols:
            try:
                poly = Poly(diff, x)
                deg = poly.degree()
                if deg == 1:
                    return solve_linear_steps(lhs_sym, rhs_sym, x)
                elif deg == 2:
                    return solve_quadratic_steps(lhs_sym, rhs_sym, x)
            except Exception:
                pass

            # General algebraic fallback
            try:
                roots = solve(Eq(lhs_sym, rhs_sym), x)
                roots_latex = ", ".join([f"x = {sp.latex(r)}" for r in roots])
                return {
                    "status": "solved",
                    "equation_type": "algebraic",
                    "steps": [
                        {
                            "step_number": 1,
                            "title": "Equation Formulation",
                            "description": "Formulate algebraic equation.",
                            "latex": sp.latex(Eq(lhs_sym, rhs_sym))
                        },
                        {
                            "step_number": 2,
                            "title": "Solve for x",
                            "description": "Find values of x satisfying the equation.",
                            "latex": roots_latex
                        }
                    ],
                    "final_answer": ", ".join([f"x = {r}" for r in roots]),
                    "final_answer_latex": roots_latex,
                    "roots": [str(r) for r in roots]
                }
            except Exception as e:
                return {
                    "status": "error",
                    "equation_type": "algebraic",
                    "error_message": f"Could not solve algebraic equation: {str(e)}",
                    "steps": [],
                    "final_answer": "",
                    "final_answer_latex": "",
                    "roots": []
                }
        else:
            # Constant equality / contradiction: e.g. 2 + 3 = 5 or x + 1 = x + 2 (-1 = 0)
            is_identity = bool(lhs_sym == rhs_sym)
            if is_identity:
                status_key = "solved"
                status_text = "True Identity: Infinite solutions"
                latex_ans = "\\text{Identity: All } x \\in \\mathbb{R}"
            else:
                status_key = "unsolvable"
                status_text = "Contradiction: No solution"
                latex_ans = "\\emptyset \\quad (\\text{No solution})"

            return {
                "status": status_key,
                "equation_type": "arithmetic_equality",
                "steps": [
                    {
                        "step_number": 1,
                        "title": "Evaluate Left-Hand Side",
                        "description": "Simplify LHS.",
                        "latex": f"\\text{{LHS}} = {sp.latex(lhs_sym)}"
                    },
                    {
                        "step_number": 2,
                        "title": "Evaluate Right-Hand Side",
                        "description": "Simplify RHS.",
                        "latex": f"\\text{{RHS}} = {sp.latex(rhs_sym)}"
                    },
                    {
                        "step_number": 3,
                        "title": "Verify Equality",
                        "description": "Compare LHS and RHS values.",
                        "latex": f"{sp.latex(lhs_sym)} {'=' if is_identity else '\\neq'} {sp.latex(rhs_sym)} \\implies \\text{{{status_text}}}"
                    }
                ],
                "final_answer": status_text,
                "final_answer_latex": latex_ans,
                "roots": []
            }

    # Case 2: Pure arithmetic expression (e.g. 2 + 3 * 4)
    return solve_arithmetic_steps(clean_expr)


if __name__ == "__main__":
    # Test arithmetic
    res1 = solve_expression("2 + 3 * 4")
    print("\n[Arithmetic] '2 + 3 * 4':", res1["final_answer"])

    # Test linear
    res2 = solve_expression("3*x + 5 = 20")
    print("\n[Linear] '3*x + 5 = 20':", res2["final_answer"])

    # Test quadratic
    res3 = solve_expression("x**2 - 5*x + 6 = 0")
    print("\n[Quadratic] 'x**2 - 5*x + 6 = 0':", res3["final_answer"])
