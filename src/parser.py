"""
=============================================================================
MathVision: Syntactic Parsing & Mathematical Expression Reconstruction
=============================================================================

This module converts a sequence of recognized character symbols and their
bounding boxes into a syntactically valid, canonical mathematical expression.

Key Tasks:
    1. Spatial Exponent Detection:
       Detects superscripts/exponents by inspecting bounding box geometry:
       a symbol that is elevated relative to its preceding base character
       and smaller in vertical scale is parsed as a power operand ('**').
    2. Canonical Symbol Normalization:
       Maps visual glyphs to standard computational operators:
       'times' or '×' -> '*', 'div' or '÷' -> '/', '–' -> '-'.
    3. Implicit Multiplication Insertion:
       Reconstructs omitted algebraic multiplication:
       - '2x'    -> '2*x'
       - 'x(x+1)'-> 'x*(x+1)'
       - '(2)(3)'-> '(2)*(3)'
       - '3(4)'  -> '3*(4)'
    4. Robust Syntax Validation:
       Validates parenthesis nesting, operator adjacency, and boundary placement,
       returning human-friendly diagnostic errors rather than throwing crashes.
=============================================================================
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


# Default frozen configuration tuned on validation split
DEFAULT_DISAMBIGUATION_CONFIG: Dict[str, Any] = {
    "w_stroke": 0.4,
    "w_grammar": 1.5,
    "margin_threshold": 0.40,
    "mu_x": 1.049,
    "sig_x": 0.391,
    "mu_t": 0.917,
    "sig_t": 0.234,
}


def disambiguate_x_times(
    symbols: List[str],
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    probs: Optional[np.ndarray] = None,
    return_ambiguities: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Soft-rescoring Bayesian disambiguation between algebraic variable 'x' and multiplication 'times'/'*'.

    Combines:
      1. CNN softmax probabilities: P_CNN(x) and P_CNN(times)
      2. Stroke-aspect likelihood: P_stroke(aspect)
      3. Context-aware grammar prior: P_grammar(x | neighbors)
      4. Hard syntactic bounds: strictly reserved for syntactically invalid cases:
         - Expression start (index 0)
         - Adjacent to '=' sign
         - Directly after/before binary operators (+, -, /, div)
         - Exponent base (**2)
         - Direct parenthesis boundary (*, *)

    Ambiguity Flagging:
      When neither hypothesis is grammatically forbidden, and the posterior margin
      |log P(x) - log P(times)| is below a confidence threshold (or in inherently
      ambiguous juxtaposition like digit-cross-parenthesis), the token is flagged
      in amber for manual verification in the UI.

    Args:
        symbols: List of predicted token strings in left-to-right order.
        boxes: Optional list of (x, y, w, h) bounding boxes.
        probs: Optional (N, 18) array of CNN softmax probability distributions.
        return_ambiguities: If True, returns (resolved_symbols, ambiguous_tokens_list).
                            If False, returns resolved_symbols.
        config: Optional configuration dictionary overriding default weights and priors.

    Returns:
        List[str] or Tuple[List[str], List[Dict[str, Any]]]:
            Disambiguated token list (and optional list of flagged ambiguous items).
    """
    if not symbols:
        return ([], []) if return_ambiguities else []

    cfg = {**DEFAULT_DISAMBIGUATION_CONFIG, **(config or {})}
    w_stroke = cfg["w_stroke"] if boxes else 0.0
    w_grammar = cfg["w_grammar"]
    margin_threshold = cfg["margin_threshold"]
    mu_x = cfg["mu_x"]
    sig_x = cfg["sig_x"]
    mu_t = cfg["mu_t"]
    sig_t = cfg["sig_t"]

    resolved = list(symbols)
    n = len(resolved)

    DIGITS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    BINARY_OPS = {"+", "-", "/", "div", "plus", "minus"}

    ambiguous_tokens = []

    # Map class indices if probs provided
    from src.dataset import CLASSES
    x_idx = CLASSES.index("x") if "x" in CLASSES else 17
    times_idx = CLASSES.index("times") if "times" in CLASSES else 12

    for i in range(n):
        curr = resolved[i]
        if curr not in ("x", "times", "*", "×"):
            continue

        prev_sym = resolved[i - 1] if i > 0 else None
        next_sym = resolved[i + 1] if i + 1 < n else None

        # =====================================================================
        # 1. HARD SYNTACTIC BOUNDS (Only for syntax impossibilities)
        # =====================================================================
        # Leading position: expressions cannot start with an infix multiplication operator
        if prev_sym is None:
            resolved[i] = "x"
            continue

        # Trailing position: expressions cannot end with an infix multiplication operator
        if next_sym is None:
            resolved[i] = "x"
            continue

        # Adjacent to equals sign: '= *' or '* =' is invalid in standard equation syntax
        if prev_sym == "=" or next_sym == "=":
            resolved[i] = "x"
            continue

        # Adjacent to binary operators: '+ *', '- *', '* +', '* -', '/* ' are invalid
        if prev_sym in BINARY_OPS or next_sym in BINARY_OPS:
            resolved[i] = "x"
            continue

        # Parenthesis boundaries: '(*' or '*)' is invalid
        if prev_sym == "(" or next_sym == ")":
            resolved[i] = "x"
            continue

        # Base of exponent: '* ** 2' is invalid
        if next_sym == "**":
            resolved[i] = "x"
            continue

        # =====================================================================
        # 2. SOFT RESCORING (Bayesian Posterior: CNN + Stroke Aspect + Grammar)
        # =====================================================================
        # A. CNN Probabilities
        if probs is not None and i < len(probs):
            p_cnn_x = max(1e-4, float(probs[i, x_idx]))
            p_cnn_t = max(1e-4, float(probs[i, times_idx]))
        else:
            # Uninformative/neutral prior when continuous model logits are omitted
            p_cnn_x = 0.55 if curr in ("x",) else 0.45
            p_cnn_t = 0.55 if curr in ("times", "*", "×") else 0.45

        # B. Stroke-Aspect & Scale Likelihood from Bounding Box
        if boxes and i < len(boxes):
            bw, bh = boxes[i][2], boxes[i][3]
            asp = bw / float(bh) if bh > 0 else 1.0

            # Relative height compared to neighbors (multiplication operators are typically compact)
            rel_h = 1.0
            neighbor_h = []
            if i > 0 and i - 1 < len(boxes):
                neighbor_h.append(boxes[i - 1][3])
            if i + 1 < len(boxes):
                neighbor_h.append(boxes[i + 1][3])
            if neighbor_h:
                rel_h = bh / float(np.mean(neighbor_h))

            # Validation-set empirical distributions (tuned on X_val, never test):
            l_x = np.exp(-((asp - mu_x) ** 2) / (2 * (sig_x ** 2)))
            l_t = np.exp(-((asp - mu_t) ** 2) / (2 * (sig_t ** 2)))

            # Shorter, vertically centered glyphs strongly indicate operators
            if rel_h < 0.70:
                l_t *= 2.0
                l_x *= 0.5
            elif rel_h > 0.85:
                l_x *= 1.3

            p_strk_x = l_x / (l_x + l_t + 1e-6)
            p_strk_t = l_t / (l_x + l_t + 1e-6)
        else:
            p_strk_x = 0.50
            p_strk_t = 0.50

        # C. Context Grammar Prior (Hand-written domain syntax heuristics)
        p_gram_t = 0.50
        p_gram_x = 0.50
        is_collision_context = False
        collision_reason = ""

        if prev_sym in DIGITS and next_sym in DIGITS:
            # Arithmetic product: e.g. '3 x 4'
            p_gram_t = 0.88
            p_gram_x = 0.12
        elif prev_sym == ")" and next_sym in DIGITS:
            # Bracket product: e.g. '(a+b) x 4'
            p_gram_t = 0.82
            p_gram_x = 0.18
        elif prev_sym in DIGITS and next_sym == "(":
            # Ambiguous: '4 x (x+1)' (multiplication) vs '4x(x+1)' (monomial * bracket)
            p_gram_t = 0.60
            p_gram_x = 0.40
            is_collision_context = True
            collision_reason = "Juxtaposition between digit and '(' allows both arithmetic product '4*(...)' and algebraic monomial '4x*(...)'"
        elif prev_sym in DIGITS and next_sym in ("x", "times"):
            # Ambiguous: '3 x x' (3*x) vs '3x^2'
            p_gram_t = 0.65
            p_gram_x = 0.35
            is_collision_context = True
            collision_reason = "Juxtaposition between digit and variable allows both '3*x' and '3x'"

        # D. Log Posterior Score (Using frozen validation configuration)
        post_x = np.log(p_cnn_x) + w_stroke * np.log(p_strk_x) + w_grammar * np.log(p_gram_x)
        post_t = np.log(p_cnn_t) + w_stroke * np.log(p_strk_t) + w_grammar * np.log(p_gram_t)
        chosen = "x" if post_x >= post_t else "times"
        resolved[i] = chosen

        # E. Ambiguity Detection & Amber Flagging (margin threshold tuned on validation)
        margin = abs(post_x - post_t)
        if margin < margin_threshold or is_collision_context:
            alt = "times" if chosen == "x" else "x"
            reason = collision_reason if collision_reason else f"Narrow posterior margin ({margin:.2f}) between variable 'x' and multiplication '×'"
            ambiguous_tokens.append({
                "index": i,
                "chosen": chosen,
                "alternative": alt,
                "margin": round(float(margin), 3),
                "reason": reason
            })

    if return_ambiguities:
        return resolved, ambiguous_tokens
    return resolved


def detect_exponents(
    symbols: List[str],
    boxes: Optional[List[Tuple[int, int, int, int]]] = None
) -> List[str]:
    """
    Inserts '**' when a character is spatially elevated and scaled as a superscript.

    Viva Explanation:
    In handwriting, exponents (e.g. x^2) do not include an explicit '^' symbol.
    We identify exponents by inspecting bounding box geometry:
        1. Vertical elevation: The candidate symbol's bottom (cy + ch) must be well above
           the base symbol's baseline, and its center must lie in the upper region of the base.
        2. Relative scale: The exponent symbol height is smaller than the base symbol
           (ch <= 0.82 * ph for variables, ch <= 0.75 * ph for digits).
        3. Valid base context: Base character must be a variable ('x'), digit, or closing ')'.
        4. Multi-digit Exponent Scope & Boundary Isolation:
           - In single-line handwritten math, exponents are almost exclusively single digits (x^2)
             or at most 2 digits (x^10). Contiguous runs of 3 or more digits (e.g. '111', '123')
             represent standard multi-digit numbers/operands, NEVER exponents.
           - Furthermore, when an exponent term finishes, any subsequent baseline digit is
             separated with an explicit implicit multiplication operator ('*') rather than
             being greedily swallowed into the exponent power operand (preventing e.g. x^2 followed
             by 11 from collapsing into x**211).

    Args:
        symbols: List of predicted class strings in left-to-right order.
        boxes: Optional list of (x, y, w, h) bounding box coordinates.

    Returns:
        List[str]: Augmented symbol list with '**' inserted before superscripts.
    """
    if not boxes or len(symbols) != len(boxes) or len(symbols) < 2:
        return list(symbols)

    augmented_symbols: List[str] = []
    base_candidates = {"x", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ")"}
    exp_candidates = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

    i = 0
    while i < len(symbols):
        if i == 0:
            augmented_symbols.append(symbols[0])
            i += 1
            continue

        prev_sym = symbols[i - 1]
        curr_sym = symbols[i]

        px, py, pw, ph = boxes[i - 1]
        cx, cy, cw, ch = boxes[i]

        # Check eligibility: only digits or x can be exponents
        if prev_sym in base_candidates and curr_sym in exp_candidates:
            h_gap = cx - (px + pw)
            is_close_h = -0.3 * pw <= h_gap < pw * 1.5

            if is_close_h:
                # Count how many contiguous digits starting at i are elevated relative to base (px, py, pw, ph)
                elevated_digits_count = 0
                max_check = min(len(symbols), i + 4)
                for k in range(i, max_check):
                    if not symbols[k].isdigit():
                        break
                    kx, ky, kw, kh = boxes[k]
                    k_y_center = ky + 0.5 * kh

                    if prev_sym.isdigit():
                        is_k_elevated = (
                            (ky + kh <= py + 0.55 * ph)
                            and (kh <= 0.75 * ph)
                            and (k_y_center < py + 0.35 * ph)
                        )
                    else:
                        is_k_elevated = (
                            (ky + kh <= py + 0.65 * ph)
                            and (kh <= 0.82 * ph)
                            and (k_y_center < py + 0.42 * ph)
                        )
                    if is_k_elevated:
                        elevated_digits_count += 1
                    else:
                        break

                # Exponents in handwritten math are at most 2 digits.
                # Runs of 3+ digits (e.g. 111) are multi-digit integers/operands, NEVER exponents.
                # For digit bases, allow at most 1 digit exponent (e.g. 2^3).
                is_valid_exp = False
                if elevated_digits_count == 1:
                    is_valid_exp = True
                elif elevated_digits_count == 2 and not prev_sym.isdigit():
                    is_valid_exp = True

                if is_valid_exp:
                    augmented_symbols.append("**")
                    for _ in range(elevated_digits_count):
                        augmented_symbols.append(symbols[i])
                        i += 1
                    # If the next token after the exponent is a baseline digit,
                    # insert '*' so the baseline digit is not swallowed into the power operand
                    if i < len(symbols) and symbols[i].isdigit():
                        augmented_symbols.append("*")
                    continue

        augmented_symbols.append(curr_sym)
        i += 1

    return augmented_symbols


def normalize_operators(symbols: List[str]) -> List[str]:
    """
    Normalizes visual and dataset class representations to canonical Python / SymPy operators.
    """
    normalized: List[str] = []
    for s in symbols:
        s_clean = s.strip()
        if s_clean in ("times", "×", "X_cross", "mul"):
            normalized.append("*")
        elif s_clean in ("div", "÷", "/"):
            normalized.append("/")
        elif s_clean in ("–", "—", "minus", "sub"):
            normalized.append("-")
        elif s_clean in ("plus", "add"):
            normalized.append("+")
        elif s_clean in ("equal", "equals", "eq"):
            normalized.append("=")
        else:
            normalized.append(s_clean)
    return normalized


def insert_implicit_multiplication(expr: str) -> str:
    """
    Inserts explicit '*' for implicit algebraic multiplication.

    Viva Explanation:
    Human handwriting omits multiplication operators in standard notation:
    '3x' means 3 * x, '2(x+1)' means 2 * (x + 1), '(x-1)(x+2)' means (x - 1) * (x + 2).
    A symbolic algebra parser requires explicit operators to avoid treating '3x'
    as an unknown multi-character identifier.
    """
    s = expr

    # 1. Digit followed by variable: e.g. '2x' -> '2*x'
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)

    # 2. Variable followed by digit (if exponent not used): e.g. 'x2' -> 'x*2'
    s = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", s)

    # 3. Digit followed by opening parenthesis: e.g. '2(' -> '2*('
    s = re.sub(r"(\d)(\()", r"\1*\2", s)

    # 4. Variable followed by opening parenthesis: e.g. 'x(' -> 'x*('
    s = re.sub(r"([a-zA-Z])(\()", r"\1*\2", s)

    # 5. Closing parenthesis followed by opening parenthesis: e.g. ')(' -> ')*('
    s = re.sub(r"(\))(\()", r"\1*\2", s)

    # 6. Closing parenthesis followed by digit or variable: e.g. ')x' -> ')*x', ')2' -> ')*2'
    s = re.sub(r"(\))([a-zA-Z0-9])", r"\1*\2", s)

    return s


def validate_expression(expr: str) -> Tuple[bool, str]:
    """
    Performs comprehensive structural syntax verification on the rebuilt equation.

    Checks:
        1. Non-empty string
        2. Balanced parentheses nesting
        3. No empty parentheses '()'
        4. No invalid consecutive operators (e.g. '++', '+*', '/-', etc.)
        5. No illegal leading or trailing operators
        6. At most one '=' sign

    Returns:
        Tuple[bool, str]: (is_valid, descriptive_error_message_or_empty)
    """
    s = expr.replace(" ", "")
    if not s:
        return False, "Expression is empty."

    # 0. Safety Limits Check
    if len(s) > 200:
        return False, f"Expression length ({len(s)} chars) exceeds safe limit (maximum 200 characters)."

    literals = re.findall(r"\b\d+\b", s)
    for lit in literals:
        if len(lit) > 10:
            return False, f"Numeric literal '{lit[:12]}...' exceeds maximum allowed size (10 digits). Check the recognized equation."

    exp_matches = re.findall(r"\*\*\s*([+-]?\d+)", s)
    for exp_str in exp_matches:
        try:
            if abs(int(exp_str)) > 50:
                return False, f"Exponent magnitude ({exp_str}) exceeds safe limit (maximum allowed exponent is 50)."
        except ValueError:
            return False, "Invalid exponent format."

    if re.search(r"\*\*[^\+\-\*\/=]+\*\*", s):
        return False, "Chained exponentiation is not supported for safe evaluation."

    # 1. Balanced Parentheses Check
    balance = 0
    for char in s:
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
            if balance < 0:
                return False, "Unbalanced parentheses: unexpected closing ')' before opening '('."
    if balance != 0:
        return False, f"Unbalanced parentheses: missing {balance} closing ')' parenthesis."

    # 2. Empty Parentheses
    if "()" in s:
        return False, "Syntax error: empty parentheses '()' detected."

    # 3. Equality Sign Count
    equal_count = s.count("=")
    if equal_count > 1:
        return False, f"Invalid equation: found {equal_count} '=' signs (only single-line equations supported)."

    # 4. Check sides if equality present
    if equal_count == 1:
        lhs, rhs = s.split("=")
        if not lhs:
            return False, "Missing left-hand side before '='."
        if not rhs:
            return False, "Missing right-hand side after '='."

    # 5. Invalid Consecutive Operators
    # Notice: '**' is allowed (power), but '***', '++', '+*', '*/', etc. are invalid
    # Temporarily replace '**' with a placeholder to check other consecutive operators
    s_temp = s.replace("**", "#")
    if re.search(r"[\+\-\*/#]{2,}", s_temp):
        return False, "Syntax error: consecutive operators detected (e.g. '++', '+*', or '*/')."

    # 6. Illegal Leading / Trailing Operators
    # In each equation side, leading '+' or '-' is allowed (unary), but '*' or '/' or '=' is illegal
    sides = s.split("=") if "=" in s else [s]
    for side in sides:
        side_clean = side.strip()
        if side_clean.startswith(("*", "/", "**")):
            return False, f"Illegal leading operator '{side_clean[0]}' in '{side_clean}'."
        if side_clean.endswith(("+", "-", "*", "/", "**")):
            return False, f"Illegal trailing operator '{side_clean[-1]}' at end of '{side_clean}'."

    return True, ""


def to_latex(expr: str) -> str:
    """
    Converts canonical Python expression string to formatted LaTeX representation.
    e.g. '3*x + 5 = 20' -> '3x + 5 = 20', 'x**2' -> 'x^{2}', '15 / 3' -> '\\frac{15}{3}'
    """
    s = expr

    # Replace '**' exponent with '^{...}'
    s = re.sub(r"\*\*(\d+|[a-zA-Z])", r"^{\1}", s)

    # Remove explicit '*' when between digit and variable: '3*x' -> '3x'
    s = re.sub(r"(\d)\*([a-zA-Z])", r"\1\2", s)

    # Replace remaining '*' with '\cdot '
    s = s.replace("*", " \\cdot ")

    # Replace '/' with '\div ' for display or keep as division
    s = s.replace("/", " \\div ")

    return s


def repair_syntax_artifacts(
    symbols: List[str],
    boxes: Optional[List[Tuple[int, int, int, int]]] = None
) -> Tuple[List[str], Optional[List[Tuple[int, int, int, int]]], List[Dict[str, Any]]]:
    """
    Detects and repairs known computer vision fragmentation artifacts before syntax validation:
      1. Unmerged Division + Fractured Digit 4:
         - ['-', '-', '(', '1'] -> ['div', '4']
         - ['div', '(', '1'] -> ['div', '4'] (left curved arc '(' + vertical line '1' = '4')
         - ['-', '-', '('] followed by digit -> ['div', digit]
      2. Consecutive Minus Bars Forming Equals:
         - ['-', '-'] between operands where bars overlap horizontally -> ['=']
      3. Trailing or Spurious Isolated Parentheses in Arithmetic:
         - If an isolated '(' occurs with no closing ')' anywhere in the expression,
           and it follows a binary operator or sits at the end of the expression,
           it is safely repaired to avoid unbalanced parentheses crashes.
    """
    tokens = list(symbols)
    b_list = list(boxes) if boxes and len(boxes) == len(symbols) else None
    new_tokens: List[str] = []
    new_boxes: Optional[List[Tuple[int, int, int, int]]] = [] if b_list is not None else None
    ambiguities: List[Dict[str, Any]] = []

    i = 0
    n = len(tokens)
    while i < n:
        # Pattern 1: ['-', '-', '(', '1'] -> ['div', '4']
        if i + 3 < n and tokens[i:i+4] == ["-", "-", "(", "1"]:
            new_tokens.extend(["div", "4"])
            if new_boxes is not None and b_list is not None:
                b1, b2 = b_list[i], b_list[i+1]
                div_box = (
                    min(b1[0], b2[0]), min(b1[1], b2[1]),
                    max(b1[0]+b1[2], b2[0]+b2[2]) - min(b1[0], b2[0]),
                    max(b1[1]+b1[3], b2[1]+b2[3]) - min(b1[1], b2[1])
                )
                b3, b4 = b_list[i+2], b_list[i+3]
                four_box = (
                    min(b3[0], b4[0]), min(b3[1], b4[1]),
                    max(b3[0]+b3[2], b4[0]+b4[2]) - min(b3[0], b4[0]),
                    max(b3[1]+b3[3], b4[1]+b4[3]) - min(b3[1], b4[1])
                )
                new_boxes.extend([div_box, four_box])
            ambiguities.append({
                "index": len(new_tokens) - 2,
                "chosen": "div",
                "alternative": "--(",
                "reason": "Reconstructed division symbol and composite digit 4 from multi-stroke fragments."
            })
            i += 4
            continue

        # Pattern 2: ['div', '(', '1'] -> ['div', '4'] (or ['/', '(', '1'])
        if i + 2 < n and tokens[i] in ("div", "/", "÷") and tokens[i+1:i+3] == ["(", "1"]:
            new_tokens.extend(["div", "4"])
            if new_boxes is not None and b_list is not None:
                new_boxes.append(b_list[i])
                b3, b4 = b_list[i+1], b_list[i+2]
                four_box = (
                    min(b3[0], b4[0]), min(b3[1], b4[1]),
                    max(b3[0]+b3[2], b4[0]+b4[2]) - min(b3[0], b4[0]),
                    max(b3[1]+b3[3], b4[1]+b4[3]) - min(b3[1], b4[1])
                )
                new_boxes.append(four_box)
            ambiguities.append({
                "index": len(new_tokens) - 1,
                "chosen": "4",
                "alternative": "(1",
                "reason": "Merged fractured arc '(' and vertical stroke '1' into composite digit 4."
            })
            i += 3
            continue

        # Pattern 3: ['-', '-', '('] followed by digit -> ['div', digit]
        if i + 2 < n and tokens[i:i+3] == ["-", "-", "("]:
            new_tokens.append("div")
            if new_boxes is not None and b_list is not None:
                b1, b2, b3 = b_list[i], b_list[i+1], b_list[i+2]
                div_box = (
                    min(b1[0], b2[0], b3[0]), min(b1[1], b2[1], b3[1]),
                    max(b1[0]+b1[2], b2[0]+b2[2], b3[0]+b3[2]) - min(b1[0], b2[0], b3[0]),
                    max(b1[1]+b1[3], b2[1]+b2[3], b3[1]+b3[3]) - min(b1[1], b2[1], b3[1])
                )
                new_boxes.append(div_box)
            ambiguities.append({
                "index": len(new_tokens) - 1,
                "chosen": "div",
                "alternative": "--(",
                "reason": "Reconstructed division symbol from fractured horizontal bar and dot strokes."
            })
            i += 3
            continue

        # Pattern 4: Trailing unmatched '(' at the end of expression
        if tokens[i] == "(" and tokens.count("(") > tokens.count(")") and (i == n - 1 or tokens.count(")") == 0):
            # If immediately following an operator and no closing bracket exists
            if i > 0 and tokens[i-1] in ("+", "-", "*", "/", "div", "="):
                # Check if it precedes a single digit at the end of the expression: e.g. ['+', '(', '4']
                if i + 1 == n - 1 and tokens[i+1].isdigit():
                    # Omit the phantom bracket, keep the digit
                    i += 1
                    continue
                elif i == n - 1:
                    # Pure trailing noise bracket
                    i += 1
                    continue

        # Pattern 4b: Unmatched '(' directly preceding a digit with no matching ')' in the expression
        # e.g. ['+', '(', '2', ...] forming multi-digit number like '12'
        if tokens[i] == "(" and tokens.count(")") == 0 and i + 1 < n and tokens[i+1].isdigit():
            # Check if adjacent to next digit or following an operator
            is_adjacent = True
            if b_list is not None:
                curr_b = b_list[i]
                next_b = b_list[i+1]
                gap = next_b[0] - (curr_b[0] + curr_b[2])
                is_adjacent = (gap <= 10 and curr_b[2] <= 10)
            if is_adjacent or (i > 0 and tokens[i-1] in ("+", "-", "*", "/", "div", "=")):
                new_tokens.append("1")
                if new_boxes is not None and b_list is not None:
                    new_boxes.append(b_list[i])
                ambiguities.append({
                    "index": len(new_tokens) - 1,
                    "chosen": "1",
                    "alternative": "(",
                    "reason": "Repaired unmatched opening parenthesis '(' adjacent to digit into digit '1' to form multi-digit number."
                })
                i += 1
                continue

        # Pattern 4c: Unmatched ')' directly preceding a digit with no matching '(' in the expression
        # e.g. ['+', ')', '2', ...] forming multi-digit number like '12'
        if tokens[i] == ")" and tokens.count("(") == 0 and i + 1 < n and tokens[i+1].isdigit():
            is_adjacent = True
            if b_list is not None:
                curr_b = b_list[i]
                next_b = b_list[i+1]
                gap = next_b[0] - (curr_b[0] + curr_b[2])
                is_adjacent = (gap <= 10 and curr_b[2] <= 10)
            if is_adjacent or (i > 0 and tokens[i-1] in ("+", "-", "*", "/", "div", "=")):
                new_tokens.append("1")
                if new_boxes is not None and b_list is not None:
                    new_boxes.append(b_list[i])
                ambiguities.append({
                    "index": len(new_tokens) - 1,
                    "chosen": "1",
                    "alternative": ")",
                    "reason": "Repaired unmatched closing parenthesis ')' adjacent to digit into digit '1' to form multi-digit number."
                })
                i += 1
                continue

        # Pattern 5: Tall diagonal slash between closing parenthesis ')' and a digit erroneously labeled '1'
        # e.g. [..., ')', '1', '4'] where '1' is a tall/wide stroke (h >= 20, w >= 8) -> [..., ')', 'div', '4']
        if tokens[i] == "1" and i > 0 and tokens[i-1] == ")" and i + 1 < n and tokens[i+1].isdigit():
            is_tall_slash = False
            if b_list is not None and i < len(b_list):
                curr_b = b_list[i]
                # A division slash spans full character height and notable width
                if curr_b[3] >= 20 and curr_b[2] >= 8:
                    is_tall_slash = True
            if is_tall_slash:
                new_tokens.append("div")
                if new_boxes is not None and b_list is not None:
                    new_boxes.append(b_list[i])
                ambiguities.append({
                    "index": len(new_tokens) - 1,
                    "chosen": "div",
                    "alternative": "1",
                    "reason": "Reconstructed division slash '/' from tall stroke positioned between ')' and digit."
                })
                i += 1
                continue

        # Pattern 6: Consecutive duplicate addition operators ['+', '+'] (e.g. noise speck misclassified as '+')
        if tokens[i] in ("+", "plus") and i + 1 < n and tokens[i+1] in ("+", "plus"):
            keep_first = True
            if b_list is not None and i + 1 < len(b_list):
                b1 = b_list[i]
                b2 = b_list[i+1]
                area1 = b1[2] * b1[3]
                area2 = b2[2] * b2[3]
                if area2 > area1:
                    keep_first = False
            new_tokens.append("+")
            if new_boxes is not None and b_list is not None:
                new_boxes.append(b_list[i] if keep_first else b_list[i+1])
            ambiguities.append({
                "index": len(new_tokens) - 1,
                "chosen": "+",
                "alternative": "++",
                "reason": "Deduplicated spurious consecutive addition operator artifact."
            })
            i += 2
            continue

        new_tokens.append(tokens[i])
        if new_boxes is not None and b_list is not None:
            new_boxes.append(b_list[i])
        i += 1

    return new_tokens, new_boxes, ambiguities


def parse_symbols(
    symbols: List[str],
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    probs: Optional[np.ndarray] = None,
    disambiguation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Orchestrates end-to-end parsing:
    Raw Symbols -> Soft-Rescoring Disambiguation -> Artifact Repair ->
    Exponent Detection -> Operator Normalization -> Implicit Multiplication ->
    Syntax Validation -> LaTeX Conversion.

    Returns:
        Dict[str, Any]:
            - 'raw_expression': Raw joined symbols
            - 'parsed_expression': Cleaned canonical expression (e.g. '3*x + 5 = 20')
            - 'latex_expression': Formatted LaTeX string
            - 'is_valid': Boolean validation outcome
            - 'error_message': Descriptive diagnostic error if invalid
            - 'ambiguous_tokens': List of flagged tokens requiring manual attention
    """
    # 1. Soft-Rescoring Context-Aware x vs × Disambiguation
    symbols_disambiguated, ambiguous_tokens = disambiguate_x_times(
        symbols, boxes, probs=probs, return_ambiguities=True, config=disambiguation_config
    )

    # 1.5 Repair Syntactic & Multi-Stroke Artifacts (e.g. fractured division + split glyphs like '--(1' -> 'div 4')
    repaired_symbols, repaired_boxes, repair_ambiguities = repair_syntax_artifacts(
        symbols_disambiguated, boxes=boxes
    )
    if repair_ambiguities:
        ambiguous_tokens.extend(repair_ambiguities)
    active_boxes = repaired_boxes if repaired_boxes is not None else boxes

    # 2. Spatial Exponent Detection
    symbols_with_exp = detect_exponents(repaired_symbols, active_boxes)

    # 3. Operator Normalization
    normalized = normalize_operators(symbols_with_exp)

    # Join into raw string
    raw_str = "".join(normalized)

    # 4. Implicit Multiplication
    canonical_expr = insert_implicit_multiplication(raw_str)

    # 5. Structural Syntax Validation
    is_valid, error_msg = validate_expression(canonical_expr)

    # 6. LaTeX Formatting
    latex_str = to_latex(canonical_expr) if is_valid else ""

    # 7. Contextual Ambiguity Check for free variable in arithmetic expressions
    if is_valid and "=" not in canonical_expr and "x" in canonical_expr:
        for idx, sym in enumerate(symbols_disambiguated):
            if sym == "x" and idx > 0 and symbols_disambiguated[idx - 1] in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                if not any(a.get("index") == idx for a in ambiguous_tokens):
                    ambiguous_tokens.append({
                        "index": idx,
                        "chosen": "x",
                        "alternative": "2",
                        "margin": 0.0,
                        "reason": "Variable 'x' directly follows a digit without '='. Possible digit '2' or intended monomial multiplication."
                    })

    # 7.5 Check for low-margin competing operator vs operand classifications
    if probs is not None:
        from src.dataset import CLASSES
        OPERATORS = {"+", "-", "*", "/", "times", "div", "="}
        OPERANDS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "x"}
        for i in range(min(len(repaired_symbols), len(probs))):
            p_sort = np.argsort(probs[i])[::-1]
            c1, c2 = CLASSES[p_sort[0]], CLASSES[p_sort[1]]
            p1, p2 = float(probs[i, p_sort[0]]), float(probs[i, p_sort[1]])
            margin = p1 - p2
            if (c1 in OPERATORS and c2 in OPERANDS) or (c1 in OPERANDS and c2 in OPERATORS):
                if margin < 0.20:
                    if not any(a.get("index") == i for a in ambiguous_tokens):
                        ambiguous_tokens.append({
                            "index": i,
                            "chosen": c1,
                            "alternative": c2,
                            "margin": round(margin, 4),
                            "competing_operator": True,
                            "reason": f"Low-margin classification ({c1} {p1:.1%} vs {c2} {p2:.1%}) between operator and operand."
                        })

    # 7.6 Flag suspicious implicit multiplication between ')' and a multi-digit number
    multi_digit_implicit = re.findall(r"\)\s*\*\s*(\d{2,})", canonical_expr)
    if multi_digit_implicit:
        ambiguous_tokens.append({
            "type": "suspicious_implicit_multiplication",
            "reason": f"Implicit multiplication applied between ')' and multi-digit number ({multi_digit_implicit[0]}). May indicate an omitted or misclassified operator.",
            "flag_for_review": True
        })

    return {
        "raw_expression": raw_str,
        "parsed_expression": canonical_expr,
        "latex_expression": latex_str,
        "is_valid": is_valid,
        "error_message": error_msg,
        "disambiguated_symbols": symbols_disambiguated,
        "ambiguous_tokens": ambiguous_tokens
    }
