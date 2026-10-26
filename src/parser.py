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
    We identify exponents by comparing the bounding box of candidate character (i)
    with the preceding base character (i-1):
        1. Vertical elevation: The center of the exponent is higher than the
           midpoint of the base symbol (y_center_curr < y_top_prev + 0.45 * h_prev).
        2. Relative scale: The exponent symbol height is smaller than the base symbol
           (h_curr <= 0.82 * h_prev).
        3. Valid base context: Base character must be a variable, digit, or closing parenthesis.

    Args:
        symbols: List of predicted class strings in left-to-right order.
        boxes: Optional list of (x, y, w, h) bounding box coordinates.

    Returns:
        List[str]: Augmented symbol list with '**' inserted before superscripts.
    """
    if not boxes or len(symbols) != len(boxes) or len(symbols) < 2:
        return list(symbols)

    augmented_symbols: List[str] = [symbols[0]]
    base_candidates = {"x", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ")"}
    exp_candidates = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "x"}

    for i in range(1, len(symbols)):
        prev_sym = symbols[i - 1]
        curr_sym = symbols[i]

        px, py, pw, ph = boxes[i - 1]
        cx, cy, cw, ch = boxes[i]

        # Check eligibility
        if prev_sym in base_candidates and curr_sym in exp_candidates:
            # Vertical elevation check: current centroid is significantly higher than base
            curr_y_center = cy + 0.5 * ch
            base_upper_region = py + 0.45 * ph

            # Relative scale check: exponent is smaller than base
            is_smaller = ch <= 0.85 * ph

            # Proximity check: horizontal gap is reasonably small
            h_gap = cx - (px + pw)
            is_close_h = h_gap < pw * 1.5

            if curr_y_center < base_upper_region and (is_smaller or curr_y_center < py + 0.25 * ph) and is_close_h:
                augmented_symbols.append("**")

        augmented_symbols.append(curr_sym)

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


def parse_symbols(
    symbols: List[str],
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    probs: Optional[np.ndarray] = None,
    disambiguation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Orchestrates end-to-end parsing:
    Raw Symbols -> Soft-Rescoring Disambiguation -> Exponent Detection ->
    Operator Normalization -> Implicit Multiplication -> Syntax Validation -> LaTeX Conversion.

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

    # 2. Spatial Exponent Detection
    symbols_with_exp = detect_exponents(symbols_disambiguated, boxes)

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

    return {
        "raw_expression": raw_str,
        "parsed_expression": canonical_expr,
        "latex_expression": latex_str,
        "is_valid": is_valid,
        "error_message": error_msg,
        "disambiguated_symbols": symbols_disambiguated,
        "ambiguous_tokens": ambiguous_tokens
    }
