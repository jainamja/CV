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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import sympy as sp
from sympy import symbols, Eq, solve, sympify, latex, degree, Poly, simplify, Rational


# =============================================================================
# SAFE COMPUTATION & FORMATTING LIMITS
# These limits protect the application from CPU denial-of-service, excessive memory
# allocations, and Python's integer string conversion limit (CVE-2020-10735).
# =============================================================================
MAX_EXPR_LENGTH: int = 200                # Maximum length of input expression string
MAX_INTEGER_LITERAL_DIGITS: int = 10     # Max digits for numeric constants (up to 9,999,999,999)
MAX_EXPONENT_MAGNITUDE: int = 50         # Max exponent power (|b| <= 50)
MAX_RESULT_DIGITS: int = 1000             # Max decimal digits permitted for full rendering
MAX_RESULT_BITS: int = 3322              # 1000 * log2(10) ~ 3322 bits
SOLVER_TIMEOUT_SECONDS: float = 2.0      # Max computation time for symbolic operations

TOO_LARGE_MSG_STR: str = "Result is too large to display safely. Check the recognized equation."
TOO_LARGE_MSG_LATEX: str = r"\text{Result is too large to display safely. Check the recognized equation.}"


x = symbols("x")


def is_too_large_number(val: Any, max_bits: int = MAX_RESULT_BITS) -> bool:
    """
    Checks if a number, SymPy Integer, Rational, or Expression contains integers
    exceeding safe bit/digit limits WITHOUT triggering Python's integer string conversion limit.
    """
    if isinstance(val, (int, sp.Integer)):
        try:
            return int(val).bit_length() > max_bits
        except Exception:
            return True
    elif isinstance(val, (Rational, sp.Rational)):
        try:
            return max(int(val.p).bit_length(), int(val.q).bit_length()) > max_bits
        except Exception:
            return True
    elif isinstance(val, sp.Expr):
        try:
            for atom in val.atoms(sp.Integer):
                if int(atom).bit_length() > max_bits:
                    return True
        except Exception:
            return True
    return False


def safe_str(expr: Any, max_digits: int = MAX_RESULT_DIGITS) -> str:
    """
    Safely converts an expression or result to string without crashing on large integers.
    Catches ValueError (CVE-2020-10735 integer string conversion limit), OverflowError, etc.
    """
    if is_too_large_number(expr):
        return TOO_LARGE_MSG_STR
    try:
        s = str(expr)
        if len(s) > max_digits:
            return TOO_LARGE_MSG_STR
        return s
    except (ValueError, OverflowError, MemoryError, RecursionError):
        return TOO_LARGE_MSG_STR
    except Exception as e:
        return f"Formatting Error: {str(e)}"


def safe_latex(expr: Any, max_digits: int = MAX_RESULT_DIGITS) -> str:
    """
    Safely converts an expression or result to LaTeX without crashing on large integers.
    Protects against integer-string conversion limits and unbounded decimal expansions.
    """
    if is_too_large_number(expr):
        return TOO_LARGE_MSG_LATEX
    try:
        l = sp.latex(expr)
        if len(l) > max_digits * 2:
            return TOO_LARGE_MSG_LATEX
        return l
    except (ValueError, OverflowError, MemoryError, RecursionError):
        return TOO_LARGE_MSG_LATEX
    except Exception:
        return safe_str(expr, max_digits=max_digits)


def validate_expression_safety(expr_str: str) -> Tuple[bool, str]:
    """
    Validates safety limits for mathematical expressions derived from OCR or user input.

    Rejection criteria:
    1. Maximum length: Single-line handwritten equations do not exceed 200 characters.
    2. Maximum integer literal digits: Numeric literals > 10 digits exceed realistic handwriting constants.
    3. Maximum exponent magnitude: Exponents with |b| > 50 lead to combinatorial explosion.
    4. Chained exponentiation: Expressions like a**b**c cause hyper-exponential growth.
    """
    clean = expr_str.strip()
    if not clean:
        return False, "Expression is empty."

    # 1. Length limit
    if len(clean) > MAX_EXPR_LENGTH:
        return False, f"Expression length ({len(clean)} chars) exceeds safe limit ({MAX_EXPR_LENGTH} chars)."

    # 2. Integer literal size limit
    literals = re.findall(r"\b\d+\b", clean)
    for lit in literals:
        if len(lit) > MAX_INTEGER_LITERAL_DIGITS:
            return False, f"Numeric literal '{lit[:12]}...' exceeds maximum allowed size ({MAX_INTEGER_LITERAL_DIGITS} digits). Check the recognized equation."

    # 3. Exponent magnitude limit
    exp_matches = re.findall(r"\*\*\s*([+-]?\d+)", clean)
    for exp_str in exp_matches:
        try:
            exp_val = int(exp_str)
            if abs(exp_val) > MAX_EXPONENT_MAGNITUDE:
                return False, f"Exponent magnitude ({exp_val}) exceeds safe limit (maximum allowed exponent is {MAX_EXPONENT_MAGNITUDE})."
        except ValueError:
            return False, "Invalid exponent format."

    # 4. Chained exponentiation check (e.g. a**b**c)
    if re.search(r"\*\*[^\+\-\*\/=]+\*\*", clean):
        return False, "Chained exponentiation is not supported for safe evaluation."

    return True, ""


def safe_solve(eq: Any, sym: Any, timeout_sec: float = SOLVER_TIMEOUT_SECONDS) -> List[Any]:
    """Runs SymPy solve with a strict timeout to prevent indefinite hangs on transcendental equations."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(solve, eq, sym)
        return future.result(timeout=timeout_sec)


def safe_simplify(expr: Any, timeout_sec: float = SOLVER_TIMEOUT_SECONDS) -> Any:
    """Runs SymPy simplify with a strict timeout to prevent indefinite hangs on complex expressions."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(simplify, expr)
        return future.result(timeout=timeout_sec)


def _to_arithmetic_latex(expr_str: str) -> str:
    """Converts a raw arithmetic expression string to formatted LaTeX math notation."""
    s = expr_str.strip()
    s = re.sub(r"\*\*(\d+)", r"^{\1}", s)
    # Ensure spaces around basic binary operators
    s = re.sub(r"([^\+\-\*\/\^]+)([\+\-\*\/])", r"\1 \2 ", s)
    s = re.sub(r"([\+\-\*\/])([^\+\-\*\/\^]+)", r" \1 \2", s)
    s = s.replace("*", r"\cdot")
    s = s.replace("/", r"\div")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(r"\div", r"\div ")
    s = s.replace(r"\cdot", r"\cdot ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _generate_arithmetic_pemdas_steps(expr_str: str) -> Optional[List[Dict[str, Any]]]:
    """
    Traces arithmetic evaluation step-by-step following PEMDAS:
    Parentheses -> Exponents -> Multiplication/Division -> Addition/Subtraction.
    Returns granular pedagogical steps or None if expression cannot be linearly traced.
    """
    try:
        raw = expr_str.replace(" ", "")
        steps: List[Dict[str, Any]] = []
        current_expr = raw
        step_num = 1

        # Step 1: Initial Expression
        steps.append({
            "step_number": step_num,
            "title": "Initial Expression",
            "description": "Identify the arithmetic expression to evaluate.",
            "latex": f"\\text{{Evaluate: }} {_to_arithmetic_latex(raw)}"
        })
        step_num += 1

        max_iter = 25
        is_first_op = True

        while max_iter > 0:
            max_iter -= 1
            # Remove redundant parentheses around isolated numbers e.g. '(6)' -> '6'
            current_expr = re.sub(r"\((\-?\d+(?:\.\d+)?)\)", r"\1", current_expr)

            # Check if current_expr is a single number
            if re.match(r"^-?\d+(?:\.\d+)?$", current_expr):
                break

            paren_match = re.search(r"\(([^()]+)\)", current_expr)
            if paren_match:
                target = paren_match.group(1)
                in_paren = True
            else:
                target = current_expr
                in_paren = False

            # Priority 1: Exponent (** or ^)
            op_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(\*\*|\^)\s*(-?\d+(?:\.\d+)?)", target)

            # Priority 2: Multiplication / Division (* or /)
            if not op_match:
                op_match = re.search(r"((?<![\d\)])-?\d+(?:\.\d+)?|\d+(?:\.\d+)?)\s*([\*\/])\s*(-?\d+(?:\.\d+)?)", target)

            # Priority 3: Addition / Subtraction (+ or -)
            if not op_match:
                op_match = re.search(r"((?<![\d\)])-?\d+(?:\.\d+)?|\d+(?:\.\d+)?)\s*([\+\-])\s*(-?\d+(?:\.\d+)?)", target)

            if not op_match:
                break

            left_str, op, right_str = op_match.groups()
            left_val = sp.Rational(left_str) if "/" in left_str else sp.sympify(left_str)
            right_val = sp.Rational(right_str) if "/" in right_str else sp.sympify(right_str)

            if op in ("**", "^"):
                val = left_val ** right_val
            elif op == "*":
                val = left_val * right_val
            elif op == "/":
                val = left_val / right_val
            elif op == "+":
                val = left_val + right_val
            elif op == "-":
                val = left_val - right_val
            else:
                break

            # Format evaluated value string
            if isinstance(val, sp.Rational) and val.q == 1:
                val_str = str(val.p)
            elif isinstance(val, (int, sp.Integer)):
                val_str = str(val)
            else:
                val_str = str(val)

            # Operator LaTeX representation
            if op == "/":
                op_latex = r" \div "
            elif op == "*":
                op_latex = r" \cdot "
            elif op in ("**", "^"):
                op_latex = "^{"
            else:
                op_latex = f" {op} "

            if op in ("**", "^"):
                sub_latex = f"{left_str}^{{{right_str}}} = {val_str}"
            else:
                sub_latex = f"{left_str}{op_latex}{right_str} = {val_str}"

            # Replace operation in target
            new_target = target[:op_match.start()] + val_str + target[op_match.end():]
            if in_paren:
                current_expr = current_expr[:paren_match.start()] + "(" + new_target + ")" + current_expr[paren_match.end():]
            else:
                current_expr = new_target

            current_expr = re.sub(r"\((\-?\d+(?:\.\d+)?)\)", r"\1", current_expr)

            # Construct pedagogical step title and description
            if in_paren and not re.search(r"[\+\-\*\/\^]", new_target):
                title = "Evaluate Inside Parentheses"
                desc = f"Evaluate the operation inside parentheses: {sub_latex}."
            elif op == "/":
                title = "Division First" if is_first_op else "Division Next"
                desc = f"Perform division before addition/subtraction according to PEMDAS order of operations: {sub_latex}."
            elif op == "*":
                title = "Multiplication First" if is_first_op else "Multiplication Next"
                desc = f"Perform multiplication according to PEMDAS order of operations: {sub_latex}."
            elif op in ("**", "^"):
                title = "Exponent First" if is_first_op else "Exponent Next"
                desc = f"Evaluate exponent power according to PEMDAS order of operations: {sub_latex}."
            elif op == "+":
                title = "Addition Next" if not is_first_op else "Addition"
                desc = f"Perform addition: {sub_latex}."
            elif op == "-":
                title = "Subtraction Next" if not is_first_op else "Subtraction"
                desc = f"Perform subtraction: {sub_latex}."
            else:
                title = "Apply Operation"
                desc = f"Evaluate: {sub_latex}."

            is_first_op = False

            # Format step LaTeX
            if re.match(r"^-?\d+(?:\.\d+)?$", current_expr):
                step_latex = f"{sub_latex}"
            else:
                step_latex = f"{sub_latex} \\implies {_to_arithmetic_latex(current_expr)}"

            steps.append({
                "step_number": step_num,
                "title": title,
                "description": desc,
                "latex": step_latex
            })
            step_num += 1

        if len(steps) >= 2:
            return steps
        return None
    except Exception:
        return None


def solve_arithmetic_steps(expr_str: str) -> Dict[str, Any]:
    """
    Evaluates arithmetic expression with step-by-step order of operations (PEMDAS).
    Includes safe bounds, timeout protection, and safe LaTeX/string rendering.
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

    # Step 1: Simplified Result with timeout protection
    try:
        simplified_result = safe_simplify(expr, timeout_sec=SOLVER_TIMEOUT_SECONDS)
    except (FutureTimeoutError, TimeoutError):
        return {
            "status": "error",
            "equation_type": "arithmetic",
            "error_message": f"Computation timed out (exceeded {SOLVER_TIMEOUT_SECONDS}s). Expression may be too complex.",
            "steps": [],
            "final_answer": "Computation Timed Out",
            "final_answer_latex": "\\text{Computation Timed Out}",
            "roots": []
        }
    except Exception as e:
        return {
            "status": "error",
            "equation_type": "arithmetic",
            "error_message": f"Computation error: {str(e)}",
            "steps": [],
            "final_answer": "Computation Error",
            "final_answer_latex": "\\text{Computation Error}",
            "roots": []
        }

    # Check if result exceeds safe size
    if is_too_large_number(simplified_result):
        steps = [{
            "step_number": 1,
            "title": "Initial Expression",
            "description": "Identify the arithmetic expression to evaluate.",
            "latex": f"\\text{{Evaluate: }} {_to_arithmetic_latex(expr_str)}"
        }, {
            "step_number": 2,
            "title": "Apply Order of Operations (PEMDAS)",
            "description": "Evaluation produced a magnitude exceeding safe display limits.",
            "latex": f"= {TOO_LARGE_MSG_LATEX}"
        }]
        return {
            "status": "error",
            "equation_type": "arithmetic",
            "error_message": TOO_LARGE_MSG_STR,
            "steps": steps,
            "final_answer": TOO_LARGE_MSG_STR,
            "final_answer_latex": TOO_LARGE_MSG_LATEX,
            "roots": []
        }

    # Attempt granular PEMDAS step breakdown
    steps = _generate_arithmetic_pemdas_steps(expr_str)
    if not steps:
        # Fallback to standard 2-step presentation
        steps = [
            {
                "step_number": 1,
                "title": "Initial Expression",
                "description": "Identify the arithmetic expression to evaluate.",
                "latex": f"\\text{{Evaluate: }} {_to_arithmetic_latex(expr_str)}"
            },
            {
                "step_number": 2,
                "title": "Apply Order of Operations (PEMDAS)",
                "description": "Evaluate exponents and multiplications/divisions first, followed by additions and subtractions.",
                "latex": f"= {safe_latex(simplified_result)}"
            }
        ]

    # Format final answer safely
    if simplified_result.is_integer:
        final_ans_str = safe_str(simplified_result)
        final_ans_latex = safe_latex(simplified_result)
    elif simplified_result.is_rational:
        try:
            val_float = float(simplified_result)
            final_ans_str = f"{safe_str(simplified_result)} (approx {val_float:.4f})"
            final_ans_latex = f"{safe_latex(simplified_result)} \\approx {val_float:.4f}"
        except (OverflowError, ValueError):
            final_ans_str = safe_str(simplified_result)
            final_ans_latex = safe_latex(simplified_result)
    else:
        final_ans_str = safe_str(simplified_result)
        final_ans_latex = safe_latex(simplified_result)

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
    Includes safe bounds, timeout protection, and safe LaTeX/string rendering.
    """
    steps: List[Dict[str, Any]] = []
    step_num = 1

    # Step 1: State original equation
    orig_eq = Eq(lhs_expr, rhs_expr)
    steps.append({
        "step_number": step_num,
        "title": "Given Linear Equation",
        "description": "State the equation to solve for variable x.",
        "latex": safe_latex(orig_eq)
    })
    step_num += 1

    # Expand both sides safely
    try:
        lhs_exp = sp.expand(lhs_expr)
        rhs_exp = sp.expand(rhs_expr)
    except Exception:
        lhs_exp = lhs_expr
        rhs_exp = rhs_expr

    if lhs_exp != lhs_expr or rhs_exp != rhs_expr:
        steps.append({
            "step_number": step_num,
            "title": "Expand Expressions",
            "description": "Expand any parentheses or products on both sides of the equation.",
            "latex": safe_latex(Eq(lhs_exp, rhs_exp))
        })
        step_num += 1

    # Step: Collect all terms onto LHS: P(x) = LHS - RHS = 0
    try:
        diff_expr = sp.expand(lhs_exp - rhs_exp)
        poly = Poly(diff_expr, var_sym)
        coeffs = poly.all_coeffs()
    except Exception as e:
        return {
            "status": "error",
            "equation_type": "linear",
            "error_message": f"Could not extract linear polynomial coefficients: {str(e)}",
            "steps": steps,
            "final_answer": "Error in Coefficient Extraction",
            "final_answer_latex": "\\text{Error in Coefficient Extraction}",
            "roots": []
        }

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

    # Step: Move constant to RHS: a*x = -b
    rhs_constant = -b
    steps.append({
        "step_number": step_num,
        "title": "Isolate Variable Terms",
        "description": f"Move constant terms to the right-hand side by subtracting {safe_str(b)} from both sides.",
        "latex": f"{safe_latex(a * var_sym)} = {safe_latex(rhs_constant)}"
    })
    step_num += 1

    # Step: Divide by leading coefficient
    if a != 1:
        steps.append({
            "step_number": step_num,
            "title": f"Divide Both Sides by Coefficient ({safe_str(a)})",
            "description": f"Divide both sides by the variable's coefficient ({safe_str(a)}) to solve for x.",
            "latex": f"{safe_latex(var_sym)} = \\frac{{{safe_latex(rhs_constant)}}}{{{safe_latex(a)}}}"
        })
        step_num += 1

    # Final Root
    try:
        root = safe_simplify(rhs_constant / a)
    except Exception as e:
        return {
            "status": "error",
            "equation_type": "linear",
            "error_message": f"Could not simplify root: {str(e)}",
            "steps": steps,
            "final_answer": "Root Simplification Error",
            "final_answer_latex": "\\text{Root Simplification Error}",
            "roots": []
        }

    if is_too_large_number(root):
        return {
            "status": "error",
            "equation_type": "linear",
            "error_message": TOO_LARGE_MSG_STR,
            "steps": steps,
            "final_answer": TOO_LARGE_MSG_STR,
            "final_answer_latex": TOO_LARGE_MSG_LATEX,
            "roots": []
        }

    steps.append({
        "step_number": step_num,
        "title": "Solution Found",
        "description": "Simplify the fraction to obtain the final value of x.",
        "latex": f"x = {safe_latex(root)}"
    })

    root_str = safe_str(root)

    return {
        "status": "solved",
        "equation_type": "linear",
        "steps": steps,
        "final_answer": f"x = {root_str}",
        "final_answer_latex": f"x = {safe_latex(root)}",
        "roots": [root_str]
    }


def solve_quadratic_steps(lhs_expr: sp.Expr, rhs_expr: sp.Expr, var_sym: sp.Symbol = x) -> Dict[str, Any]:
    """
    Generates step-by-step quadratic formula working with discriminant analysis.
    Includes safe bounds, timeout protection, and safe LaTeX/string rendering.
    """
    steps: List[Dict[str, Any]] = []
    step_num = 1

    # Step 1: State original equation
    steps.append({
        "step_number": step_num,
        "title": "Given Quadratic Equation",
        "description": "State the second-degree equation to solve for variable x.",
        "latex": safe_latex(Eq(lhs_expr, rhs_expr))
    })
    step_num += 1

    # Step 2: Standard Form a*x^2 + b*x + c = 0
    try:
        diff = sp.expand(lhs_expr - rhs_expr)
        poly = Poly(diff, var_sym)
        coeffs = poly.all_coeffs()
    except Exception as e:
        return {
            "status": "error",
            "equation_type": "quadratic",
            "error_message": f"Could not expand quadratic equation: {str(e)}",
            "steps": steps,
            "final_answer": "Expansion Error",
            "final_answer_latex": "\\text{Expansion Error}",
            "roots": []
        }

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
        "latex": f"{safe_latex(diff)} = 0"
    })
    step_num += 1

    # Step 3: Identify Coefficients
    steps.append({
        "step_number": step_num,
        "title": "Identify Coefficients",
        "description": "Extract the quadratic, linear, and constant coefficients.",
        "latex": f"a = {safe_str(a_val)}, \\quad b = {safe_str(b_val)}, \\quad c = {safe_str(c_val)}"
    })
    step_num += 1

    # Step 4: Compute Discriminant: Delta = b^2 - 4ac
    try:
        delta = safe_simplify(b_val**2 - 4 * a_val * c_val)
    except Exception as e:
        return {
            "status": "error",
            "equation_type": "quadratic",
            "error_message": f"Could not evaluate discriminant: {str(e)}",
            "steps": steps,
            "final_answer": "Discriminant Error",
            "final_answer_latex": "\\text{Discriminant Error}",
            "roots": []
        }

    if is_too_large_number(delta):
        return {
            "status": "error",
            "equation_type": "quadratic",
            "error_message": TOO_LARGE_MSG_STR,
            "steps": steps,
            "final_answer": TOO_LARGE_MSG_STR,
            "final_answer_latex": TOO_LARGE_MSG_LATEX,
            "roots": []
        }

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
        "latex": f"\\Delta = ({safe_str(b_val)})^2 - 4({safe_str(a_val)})({safe_str(c_val)}) = {safe_str(delta)} \\quad ({delta_nature})"
    })
    step_num += 1

    # Step 5: Apply Quadratic Formula
    steps.append({
        "step_number": step_num,
        "title": "Apply Quadratic Formula",
        "description": "Substitute a, b, and Delta into the quadratic formula x = (-b +- sqrt(Delta)) / (2a).",
        "latex": f"x = \\frac{{-({safe_str(b_val)}) \\pm \\sqrt{{{safe_latex(delta)}}}}}{{2({safe_str(a_val)})}}"
    })
    step_num += 1

    # Step 6: Compute Individual Roots
    try:
        roots_list = safe_solve(Eq(diff, 0), var_sym)
    except Exception as e:
        return {
            "status": "error",
            "equation_type": "quadratic",
            "error_message": f"Could not solve quadratic roots: {str(e)}",
            "steps": steps,
            "final_answer": "Root Solution Error",
            "final_answer_latex": "\\text{Root Solution Error}",
            "roots": []
        }

    roots_str_list = []
    roots_latex_list = []

    for idx, r in enumerate(roots_list):
        r_simp = safe_simplify(r)
        if is_too_large_number(r_simp):
            return {
                "status": "error",
                "equation_type": "quadratic",
                "error_message": TOO_LARGE_MSG_STR,
                "steps": steps,
                "final_answer": TOO_LARGE_MSG_STR,
                "final_answer_latex": TOO_LARGE_MSG_LATEX,
                "roots": []
            }
        roots_str_list.append(safe_str(r_simp))
        roots_latex_list.append(f"x_{{{idx + 1}}} = {safe_latex(r_simp)}")

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
    Guarantees strict safety bounds, execution timeout, and non-crashing formatting.

    Dispatches to:
        - Safety Validator (Rejects oversized literals, lengths, or exponents upfront)
        - Linear Equation Solver (if degree 1 in x)
        - Quadratic Equation Solver (if degree 2 in x)
        - General SymPy Algebraic Solver (fallback for other forms)
        - Algebraic Expression Simplifier (if free variables without '=')
        - Arithmetic Solver (if no variable and no equality)
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

    # Step 0: Validate safety limits upfront before expensive evaluation
    is_safe, safety_err = validate_expression_safety(clean_expr)
    if not is_safe:
        return {
            "status": "error",
            "equation_type": "oversized_expression",
            "error_message": safety_err,
            "steps": [],
            "final_answer": f"Error: {safety_err}",
            "final_answer_latex": f"\\text{{{safety_err}}}",
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
        try:
            diff = safe_simplify(lhs_sym - rhs_sym)
            free_symbols = diff.free_symbols
        except Exception as e:
            return {
                "status": "error",
                "equation_type": "equation",
                "error_message": f"Could not simplify equation difference: {str(e)}",
                "steps": [],
                "final_answer": "Simplification Error",
                "final_answer_latex": "\\text{Simplification Error}",
                "roots": []
            }

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

            # General algebraic fallback with timeout protection
            try:
                roots = safe_solve(Eq(lhs_sym, rhs_sym), x, timeout_sec=SOLVER_TIMEOUT_SECONDS)
                if any(is_too_large_number(r) for r in roots):
                    return {
                        "status": "error",
                        "equation_type": "algebraic",
                        "error_message": TOO_LARGE_MSG_STR,
                        "steps": [],
                        "final_answer": TOO_LARGE_MSG_STR,
                        "final_answer_latex": TOO_LARGE_MSG_LATEX,
                        "roots": []
                    }
                roots_latex = ", ".join([f"x = {safe_latex(r)}" for r in roots])
                roots_str = ", ".join([f"x = {safe_str(r)}" for r in roots])
                return {
                    "status": "solved",
                    "equation_type": "algebraic",
                    "steps": [
                        {
                            "step_number": 1,
                            "title": "Equation Formulation",
                            "description": "Formulate algebraic equation.",
                            "latex": safe_latex(Eq(lhs_sym, rhs_sym))
                        },
                        {
                            "step_number": 2,
                            "title": "Solve for x",
                            "description": "Find values of x satisfying the equation.",
                            "latex": roots_latex
                        }
                    ],
                    "final_answer": roots_str,
                    "final_answer_latex": roots_latex,
                    "roots": [safe_str(r) for r in roots]
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

            rel_op = "=" if is_identity else "\\neq"
            return {
                "status": status_key,
                "equation_type": "arithmetic_equality",
                "steps": [
                    {
                        "step_number": 1,
                        "title": "Evaluate Left-Hand Side",
                        "description": "Simplify LHS.",
                        "latex": f"\\text{{LHS}} = {safe_latex(lhs_sym)}"
                    },
                    {
                        "step_number": 2,
                        "title": "Evaluate Right-Hand Side",
                        "description": "Simplify RHS.",
                        "latex": f"\\text{{RHS}} = {safe_latex(rhs_sym)}"
                    },
                    {
                        "step_number": 3,
                        "title": "Verify Equality",
                        "description": "Compare LHS and RHS values.",
                        "latex": f"{safe_latex(lhs_sym)} {rel_op} {safe_latex(rhs_sym)} \\implies \\text{{{status_text}}}"
                    }
                ],
                "final_answer": status_text,
                "final_answer_latex": latex_ans,
                "roots": []
            }

    # Case 2: Expression without '='
    # Check if expression contains free variable 'x' (algebraic expression rather than pure arithmetic)
    try:
        parsed_sym = sympify(clean_expr)
        if x in parsed_sym.free_symbols:
            return solve_algebraic_expression_steps(clean_expr)
    except Exception:
        pass

    # Pure arithmetic expression (e.g. 2 + 3 * 4)
    return solve_arithmetic_steps(clean_expr)


def solve_algebraic_expression_steps(expr_str: str) -> Dict[str, Any]:
    """
    Handles expressions with free variables that lack an '=' sign (e.g. '2*x + 5' or 'x**2 - 4').
    Performs algebraic expansion and simplification rather than claiming an arithmetic solution.
    Guarantees strict safety bounds and safe LaTeX/string formatting.
    """
    try:
        expr = sympify(expr_str)
    except Exception as e:
        return {
            "status": "error",
            "equation_type": "algebraic_expression",
            "error_message": f"Could not parse algebraic expression: {str(e)}",
            "steps": [],
            "final_answer": "Parse Error",
            "final_answer_latex": "\\text{Parse Error}",
            "roots": []
        }

    try:
        simplified_expr = safe_simplify(expr, timeout_sec=SOLVER_TIMEOUT_SECONDS)
    except (FutureTimeoutError, TimeoutError):
        return {
            "status": "error",
            "equation_type": "algebraic_expression",
            "error_message": f"Simplification timed out (exceeded {SOLVER_TIMEOUT_SECONDS}s).",
            "steps": [],
            "final_answer": "Simplification Timed Out",
            "final_answer_latex": "\\text{Simplification Timed Out}",
            "roots": []
        }
    except Exception as e:
        return {
            "status": "error",
            "equation_type": "algebraic_expression",
            "error_message": f"Simplification error: {str(e)}",
            "steps": [],
            "final_answer": "Simplification Error",
            "final_answer_latex": "\\text{Simplification Error}",
            "roots": []
        }

    if is_too_large_number(simplified_expr):
        return {
            "status": "error",
            "equation_type": "algebraic_expression",
            "error_message": TOO_LARGE_MSG_STR,
            "steps": [
                {
                    "step_number": 1,
                    "title": "Given Expression (No Equality)",
                    "description": "Expression contains variable 'x' without an '=' sign.",
                    "latex": f"\\text{{Given: }} {safe_latex(expr)}"
                },
                {
                    "step_number": 2,
                    "title": "Computation Overflow",
                    "description": "Simplification produced a magnitude exceeding safe display limits.",
                    "latex": TOO_LARGE_MSG_LATEX
                }
            ],
            "final_answer": TOO_LARGE_MSG_STR,
            "final_answer_latex": TOO_LARGE_MSG_LATEX,
            "roots": []
        }

    steps = [
        {
            "step_number": 1,
            "title": "Given Expression (No Equality)",
            "description": "Expression contains variable 'x' without an '=' sign. Cannot solve for roots; performing algebraic simplification.",
            "latex": f"\\text{{Given: }} {safe_latex(expr)}"
        },
        {
            "step_number": 2,
            "title": "Algebraic Simplification",
            "description": "Combine like terms and simplify polynomial structure.",
            "latex": f"= {safe_latex(simplified_expr)}"
        }
    ]

    return {
        "status": "simplified",
        "equation_type": "algebraic_expression",
        "steps": steps,
        "final_answer": f"Simplified: {safe_str(simplified_expr)}",
        "final_answer_latex": f"\\text{{Simplified: }} {safe_latex(simplified_expr)}",
        "roots": []
    }


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
