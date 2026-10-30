"""
Audit of context-aware x vs times disambiguation in src/parser.py.
Verifies that downstream syntax priors correctly resolve isolated cross ambiguity
in full mathematical expressions.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parser import disambiguate_x_times, parse_symbols

def test_x_times_disambiguation():
    print("=" * 70)
    print("AUDIT: CONTEXT-AWARE x vs TIMES PARSER DISAMBIGUATION")
    print("=" * 70)

    test_cases = [
        # (Input Tokens with deliberate ambiguity, Expected Resolved Expression, Description)
        (["2", "times", "+", "5", "=", "1", "5"], "2*x+5=15", "Digit-cross-plus: cross must become variable x (2x+5=15)"),
        (["7", "x", "8", "+", "1", "2", "div", "4"], "7*8+12/4", "Digit-cross-digit: cross must become multiplication (7*8)"),
        (["x", "+", "5", "=", "1", "0"], "x+5=10", "Expression start: cross must become variable x"),
        (["times", "+", "5", "=", "1", "0"], "x+5=10", "Expression start with CNN predicting times: must become variable x"),
        (["3", "x", "-", "4", "=", "1", "8"], "3*x-4=18", "Digit-cross-minus: cross must become variable x"),
        (["(", "1", "8", "+", "6", ")", "x", "4"], "(18+6)*4", "Paren-cross-digit: cross must become multiplication *"),
        (["x", "^", "2", "-", "5", "x", "+", "6", "=", "0"], "x**2-5*x+6=0", "Quadratic polynomial with two x variables"),
        (["4", "times", "5", "+", "6", "x", "2"], "4*5+6*2", "Two multiplication operations in arithmetic expression"),
    ]

    all_passed = True
    for tokens, exp_expr, desc in test_cases:
        resolved = disambiguate_x_times(tokens)
        parsed = parse_symbols(tokens)
        got_expr = parsed.get("parsed_expression", "")
        clean_got = got_expr.replace(" ", "")
        passed = (clean_got == exp_expr)
        if not passed:
            all_passed = False
        print(f"  Test: {desc}")
        print(f"    Input tokens : {tokens}")
        print(f"    Resolved expr: {clean_got} (Expected: {exp_expr}) -> [{'PASS' if passed else 'FAIL'}]\n")

    print(f"Summary: {'ALL PASSED (100%)' if all_passed else 'SOME FAILED'}")
    print("=" * 70)

if __name__ == "__main__":
    test_x_times_disambiguation()
