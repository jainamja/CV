import os
import sys
import json
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_pipeline
from src.parser import parse_symbols
from src.solver import solve_expression

PHYSICAL_CASES = [
    ("Case A", "outputs/real_test_case/01_original.png", "12+8/4", "14"),
    ("Case B", "outputs/failure_case_b/01_original.png", "36/6+8", "14"),
    ("Case C", "outputs/failure_case_c/01_original.png", "45/5+6*2", "21"),
    ("Case D", "outputs/failure_case_d/01_original.png", "7*8+12/4", "59"),
    ("Case E", "outputs/failure_case_e/01_original.png", "7*8+12/4", "59"),
    ("Case F", "outputs/failure_case_f/01_original.png", "(18+6)/4", "6"),
    ("Case G", "outputs/failure_case_g/01_original.png", "(18+6)/4", "6"),
    ("Case H1", "outputs/failure_case_h/01_original.png", "23+15", "38"),
    ("Case H2", "outputs/failure_case_h2/01_original.png", "23+15", "38")
]

BENCHMARK_CASES = [
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
    (["4", "8", "div", "(", "2", "times", "4", ")", "+", "7"], "48/(2*4)+7", "13")
]

print("=" * 70)
print("STAGE 1: VERIFYING 9 PHYSICAL CASES UNDER PRODUCTION RUN_PIPELINE")
print("=" * 70)

phys_passed = 0
phys_results = []
for case_id, img_path, exp_expr, exp_ans in PHYSICAL_CASES:
    img = cv2.imread(img_path)
    res = run_pipeline(img)
    parsed_expr = res.get("expression_parsed", "")
    sol_ans = str(res.get("solution", {}).get("final_answer", ""))
    sol_status = res.get("solution", {}).get("status", "")
    is_pass = (sol_status in ("solved", "simplified") and sol_ans == exp_ans)
    if is_pass:
        phys_passed += 1
    p_str = "PASS" if is_pass else "FAIL"
    print(f"  {case_id:<8}: {parsed_expr} = {sol_ans} (exp: {exp_ans}) [{p_str}]")
    phys_results.append({
        "case_id": case_id, "parsed": parsed_expr, "ans": sol_ans, "passed": is_pass
    })

print(f"\nPhysical Cases Result: {phys_passed}/{len(PHYSICAL_CASES)} Passed ({phys_passed/len(PHYSICAL_CASES)*100:.1f}%)")
assert phys_passed == len(PHYSICAL_CASES), "Physical cases failed!"

print("\n" + "=" * 70)
print("STAGE 1: VERIFYING 20 MATHEMATICAL BENCHMARKS")
print("=" * 70)
bench_passed = 0
for idx, (syms, exp_expr, exp_ans) in enumerate(BENCHMARK_CASES, 1):
    p = parse_symbols(syms)
    sol = solve_expression(p["parsed_expression"]) if p["is_valid"] else {"status": "error", "final_answer": "N/A"}
    is_pass = (p["is_valid"] and p["parsed_expression"] == exp_expr and sol.get("status") in ("solved", "simplified") and str(sol.get("final_answer")) == exp_ans)
    if is_pass:
        bench_passed += 1

print(f"Benchmark Result: {bench_passed}/{len(BENCHMARK_CASES)} Passed ({bench_passed/len(BENCHMARK_CASES)*100:.1f}%)")
assert bench_passed == len(BENCHMARK_CASES), "Benchmark failed!"

print("\nSTAGE 1 VERIFICATION COMPLETED WITH ZERO DEFECTS.")
