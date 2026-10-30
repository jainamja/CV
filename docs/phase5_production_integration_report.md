# MathVision Phase 5: Authorized Production Integration Report

**Document ID:** `MATHVISION-PHASE5-INTEGRATION-REPORT`  
**Date:** October 30, 2026  
**Auditor / Integrator:** DeepMind Advanced Agentic Coding Pair Programmer  
**Repository:** `C:\Users\Jainam\Downloads\CV Project`  
**Git Commit HEAD:** `8b57b0c03af4de227af21b347ed068a726ccb764`  
**Authorization:** Phase 5 Authorized Production Integration  
**Governance Standard:** Absolute Zero-Regression Policy ("Never introduce a new bug while fixing an existing bug")  

---

## 1. Executive Summary & Stage-by-Stage Status

In Phase 5, controlled production integration was executed following the approved two-stage integration protocol:
1. **Stage 1 (Heuristic Integration):** The topological disambiguation heuristic was integrated strictly into Stage 3.5 of `src/pipeline.py`.
   - **Status:** **`PASSED`** (125/125 pytest tests, 9/9 physical cases, 20/20 mathematical benchmarks).
2. **Stage 2 (Review and Verification):** Complete diff of `src/pipeline.py` verified; zero extraneous changes.
   - **Status:** **`PASSED`**.
3. **Stage 3 (Checkpoint Promotion):** Atomic replacement of `models/cnn_mathvision.keras` with verified Experimental V4 checkpoint.
   - **Status:** **`PASSED`** (SHA-256 verified: `5d2559c57477cc0646f696a0bff781145bd728a1be5bd630478faaf33e9b3d56`).
4. **Stage 4 (Final Production Verification):** Full test suite, physical cases, benchmarks, and held-out evaluations executed against promoted V4.
   - **Status:** **`ROLLBACK TRIGGERED`**.
   - **Defect Discovered:** `tests/test_pipeline.py::test_pipeline_failure_case_c_end_to_end` failed on line 340. The existing test asserted a literal raw token match `res["predicted_symbols"][6] == "times"`. Under V4 CNN, the cross was classified as raw token `"x"` (`['4', '5', 'div', '5', '+', '6', 'x', '2']`), even though downstream parsing and symbolic solving succeeded (`45/5+6*2 = 21`).
   - In accordance with the mandatory safeguard: *"Never delete, skip, weaken, or bypass existing tests. If any stage fails, stop and roll back to the verified previous state"*, the checkpoint was **immediately rolled back** to the verified baseline checkpoint (`65d10826b...`).
5. **Post-Rollback Verification:**
   - **Status:** **`100% GREEN`** (125/125 pytest tests passed in 41.41s, 9/9 physical cases passed, 20/20 benchmarks passed).
6. **Phase 5.1 Forensic Investigation:**
   - Proved that Case C's literal assertion on `"times"` was an unnormalized Phase 1 legacy inconsistency compared to Cases D and E.
   - Proved mathematical equivalence and that both models parse to `45/5+6*2 = 21`.
7. **Phase 5 Authorized Continuation (Solution C & V4 Promotion):**
   - **Status:** **`SUCCESSFULLY PROMOTED (100% GREEN)`**.
   - Case C alias normalization applied to `tests/test_pipeline.py`.
   - V4 CNN promoted atomically to `models/cnn_mathvision.keras` (SHA-256: `5d2559c57477cc0646f696a0bff781145bd728a1be5bd630478faaf33e9b3d56`).
   - Pytest suite: **126 / 126 passed (100.0%)**.
   - Physical cases: **9 / 9 passed (100.0%)**.
   - Benchmarks: **20 / 20 passed (100.0%)**.
   - Held-out accuracy: **94.35%** (2,689 / 2,850, net gain +5 samples over baseline).
   - Rollback: **NOT REQUIRED**. All validation criteria passed with zero regressions.

---

## 2. Stage 0 — Pre-Implementation Baseline & Hashes

Prior to modifying any files, all working tree assets and cryptographic hashes were recorded:

### 2.1 Cryptographic SHA-256 Hashes
| Asset Description | File Path | Verified SHA-256 Hash | Baseline Verification Status |
| :--- | :--- | :--- | :--- |
| **Original Production CNN** | `models/cnn_mathvision.keras` | `65d10826b59133896efa12a7c3acb9f1475b1486d74acf1abbee9b15d1bb8cce` | **100% Match (Git Commit 8b57b0c)** |
| **Experimental V4 CNN** | `models/cnn_experimental_v4.keras` | `5d2559c57477cc0646f696a0bff781145bd728a1be5bd630478faaf33e9b3d56` | **100% Match (Target)** |
| **Preserved Held-Out Test Split** | `models/test_split.npz` | `14cef4cef2ae6abc712ed1f00a32ddcb55e8a01efe151bbab85a2dc0efc82f79` | **100% Match (Git Commit 8b57b0c)** |
| **Original Production Dataset** | `data/processed/split_data.npz` | `c7aab47e186e3c8bc18aa2ee4adaeb63198ad40e4ffb9bad782fac5bddc4f0cd` | **100% Match (Git Commit 8b57b0c)** |

### 2.2 Verified Backups Created
- `src/pipeline.py.pre_phase5_backup` (SHA-256: `da1ec917913e1cb9163c738dc9e3350b69436777f6adf254c4db9ec5f43400cf`)
- `models/cnn_mathvision.keras.pre_phase5_backup` (SHA-256: `65d10826b59133896efa12a7c3acb9f1475b1486d74acf1abbee9b15d1bb8cce`)

---

## 3. Stage 1 — Heuristic Production Integration

### 3.1 Scope of Production Modification
Strictly **ONE** file was modified: `src/pipeline.py`. Zero modifications were made to `src/segment.py`, `src/preprocess.py`, `src/parser.py`, `src/solver.py`, or `app.py`.

### 3.2 Exact Production Code Diff (`src/pipeline.py`)

```diff
--- a/src/pipeline.py.pre_phase5_backup
+++ b/src/pipeline.py
@@ -27,6 +27,7 @@ import time
 import zipfile
 import io
 import json
+import itertools
 from datetime import datetime
 from typing import Dict, Any, List, Optional, Tuple
 import cv2
@@ -66,6 +67,113 @@ def get_loaded_models(models_dir: str = "models") -> Dict[str, Any]:
     return _MODEL_CACHE
 
 
+def check_division_topology(
+    crop: np.ndarray,
+    min_bar_span_ratio: float = 0.45,
+    min_bar_span_px: int = 12,
+    min_gap_rows: int = 2,
+    intensity_threshold: float = 0.20
+) -> Tuple[bool, str, Dict[str, Any]]:
+    """
+    Analyzes actual foreground pixel distribution in a 45x45 character crop
+    to determine whether it exhibits the physical topology of an orthogonal
+    division symbol ('div' / '÷') as opposed to an intersecting plus sign ('+').
+
+    Morphological signature of orthogonal division:
+      1. A prominent, centrally located horizontal bar spanning >= 45% of character width.
+      2. At least two distinct connected components separated by one or more empty horizontal rows.
+      3. At least one significant vertical gap (>= min_gap_rows consecutive zero-intensity rows)
+         separating the horizontal bar from an upper or lower stroke component.
+    """
+    metrics: Dict[str, Any] = {
+        "is_division": False,
+        "reason": "uninitialized",
+        "bar_y": None,
+        "bar_span": 0,
+        "char_width": 0,
+        "char_height": 0,
+        "zeros_above": 0,
+        "zeros_below": 0,
+        "num_components": 0
+    }
+
+    if crop is None or crop.size == 0:
+        metrics["reason"] = "crop is empty or None"
+        return False, metrics["reason"], metrics
+
+    norm = crop.astype(float) if crop.max() <= 1.0 else crop.astype(float) / 255.0
+    mask = (norm > intensity_threshold).astype(np.uint8)
+    pts = np.argwhere(mask > 0)
+
+    if len(pts) < 10:
+        metrics["reason"] = "insufficient foreground pixels (<10)"
+        return False, metrics["reason"], metrics
+
+    min_y, max_y = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
+    min_x, max_x = int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))
+    h = max_y - min_y + 1
+    w = max_x - min_x + 1
+    metrics["char_width"] = w
+    metrics["char_height"] = h
+
+    if h < 12 or w < 12:
+        metrics["reason"] = f"bounding extent too small ({w}x{h} < 12x12)"
+        return False, metrics["reason"], metrics
+
+    row_counts = [int(np.sum(mask[y, :])) for y in range(min_y, max_y + 1)]
+    max_idx = int(np.argmax(row_counts))
+    bar_y = min_y + max_idx
+    metrics["bar_y"] = bar_y
+
+    bar_pts = np.where(mask[bar_y, :] > 0)[0]
+    bar_span = int(np.max(bar_pts) - np.min(bar_pts) + 1) if len(bar_pts) > 0 else 0
+    metrics["bar_span"] = bar_span
+
+    if bar_span < (min_bar_span_ratio * w) or bar_span < min_bar_span_px:
+        metrics["reason"] = f"horizontal bar span ({bar_span}px) too narrow relative to width ({w}px)"
+        return False, metrics["reason"], metrics
+
+    rel_bar_y = (bar_y - min_y) / float(h)
+    if rel_bar_y < 0.15 or rel_bar_y > 0.85:
+        metrics["reason"] = f"horizontal bar is not vertically centered (rel_y={rel_bar_y:.2f})"
+        return False, metrics["reason"], metrics
+
+    upper_rows = row_counts[:max_idx]
+    lower_rows = row_counts[max_idx + 1:]
+
+    def get_max_consecutive_zeros(rows: List[int]) -> int:
+        if not rows:
+            return 0
+        return max([sum(1 for _ in g) for k, g in itertools.groupby(rows, lambda x: x == 0) if k], default=0)
+
+    zeros_above = get_max_consecutive_zeros(upper_rows)
+    zeros_below = get_max_consecutive_zeros(lower_rows)
+    metrics["zeros_above"] = zeros_above
+    metrics["zeros_below"] = zeros_below
+
+    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
+    num_fg_components = num_labels - 1
+    metrics["num_components"] = num_fg_components
+
+    upper_mass = sum(upper_rows)
+    lower_mass = sum(lower_rows)
+
+    has_both_gaps = (zeros_above >= 1 and zeros_below >= 1)
+    has_asymmetric_gap = (zeros_above >= min_gap_rows and upper_mass >= 3) or (zeros_below >= min_gap_rows and lower_mass >= 3)
+
+    if (has_both_gaps or has_asymmetric_gap) and (num_fg_components >= 2):
+        metrics["is_division"] = True
+        reason = (
+            f"Division topology confirmed: bar_span={bar_span}px ({bar_span/w*100:.1f}% w), "
+            f"zeros_above={zeros_above}, zeros_below={zeros_below}, components={num_fg_components}."
+        )
+        metrics["reason"] = reason
+        return True, reason, metrics
+
+    metrics["reason"] = f"Continuous vertical stroke: zeros_above={zeros_above}, zeros_below={zeros_below}, comps={num_fg_components}"
+    return False, metrics["reason"], metrics
+
+
 def _run_pipeline_single(
     image_input: Any,
     settings: Optional[Dict[str, Any]] = None,
@@ -506,6 +614,34 @@ def _run_pipeline_single(
                                     f"characteristic of digit '5' bowl rather than digit '3' central cusp."
                                 )
 
+    # Disambiguation for Orthogonal Division ('div' / '÷') vs Plus ('+')
+    # Resolves ambiguity when CNN predicts '+' with low confidence (<0.65) and competing 'div' (>=0.15),
+    # verifying presence of a prominent horizontal bar and detached upper/lower dots with empty gap rows.
+    for i in range(len(predicted_symbols)):
+        if predicted_symbols[i] == "+" and i < len(crops) and idx_div >= 0:
+            top3 = symbol_diagnostics[i].get("top3", [])
+            prob_plus = confidences[i]
+            prob_div = next((prob for sym, prob in top3 if sym in ("div", "/")), 0.0)
+
+            # Two-layer defense: Layer 1 Neural Gating
+            if prob_plus < 0.65 and prob_div >= 0.15:
+                crop = crops[i]
+                # Layer 2 Pixel Morphology
+                is_div, reason, metrics = check_division_topology(crop)
+                if is_div:
+                    predicted_symbols[i] = "div"
+                    if probs is not None and i < len(probs):
+                        confidences[i] = float(probs[i, idx_div])
+                    else:
+                        confidences[i] = float(prob_div)
+                    symbol_diagnostics[i]["symbol"] = "div"
+                    symbol_diagnostics[i]["confidence"] = confidences[i]
+                    symbol_diagnostics[i]["disambiguation_note"] = (
+                        f"Disambiguated '+' -> 'div': {reason} "
+                        f"(prob_+={prob_plus:.3f}, prob_div={prob_div:.3f})"
+                    )
+                    symbol_diagnostics[i]["topological_metrics"] = metrics
+
     # -------------------------------------------------------------------------
     # APPLY MANUAL OVERRIDES (if any provided from UI)
     # -------------------------------------------------------------------------
```

### 3.3 Stage 1 Test Execution Results (Baseline Model + Heuristic)
- **Pytest Suite:** `python -m pytest tests/ -v`
  - **Result:** **`125 passed in 42.17s`** (100% pass rate).
- **Physical Acceptance Cases (A–H2):**
  - **Result:** **`9 / 9 passed (100.0%)`**.
- **20-Equation Mathematical Benchmark:**
  - **Result:** **`20 / 20 passed (100.0%)`**.
- **Original CNN Hash:** `65d10826b59133896efa12a7c3acb9f1475b1486d74acf1abbee9b15d1bb8cce` (100% untouched).

---

## 4. Stage 3 — Checkpoint Promotion

Following successful validation of Stage 1 and Stage 2, `models/cnn_mathvision.keras` was atomically replaced with `models/cnn_experimental_v4.keras`:

- **Promoted Checkpoint Path:** `models/cnn_mathvision.keras`
- **Promoted Checkpoint Hash:** `5d2559c57477cc0646f696a0bff781145bd728a1be5bd630478faaf33e9b3d56`
- **Target Hash Match:** **100% Match confirmed**.

---

## 5. Stage 4 — Final Production Verification & Root Cause Discovery

### 5.1 Test Suite Failure on Promoted V4 Checkpoint
When `python -m pytest tests/ -v` was executed against the promoted V4 checkpoint, **124 tests passed and 1 test failed**:

```
================================== FAILURES ===================================
___________________ test_pipeline_failure_case_c_end_to_end ___________________
    def test_pipeline_failure_case_c_end_to_end():
        ...
        res = run_pipeline(img_path)
        assert res["count"] == 8
        assert len(res["boxes"]) == 8
    
        # 2. Predicted Symbols
>       assert res["predicted_symbols"] == ["4", "5", "div", "5", "+", "6", "times", "2"], (
            f"Incorrect predictions: {res['predicted_symbols']}"
        )
E       AssertionError: Incorrect predictions: ['4', '5', 'div', '5', '+', '6', 'x', '2']
E       At index 6 diff: 'x' != 'times'
tests\test_pipeline.py:340: AssertionError
======================= 1 failed, 124 passed in 41.86s ========================
```

### 5.2 Forensic Root Cause Analysis of Case C Failure
1. **The Nature of the Glyph:** In failure case C (`45 ÷ 5 + 6 × 2`), the multiplication symbol between `6` and `2` is a handwritten orthogonal cross.
2. **Model Prediction Difference:**
   - The baseline CNN classified this crop with raw token `"times"`.
   - The V4 CNN classified this crop with raw token `"x"`.
3. **Downstream Parsing & Math Correctness:**
   - Downstream, `src/parser.py` uses context-aware syntax disambiguation to recognize that `"x"` positioned between operands (`6` and `2`) is a multiplication operator.
   - The parsed expression was `45/5+6*2` (**valid**).
   - The final answer was `21` (**valid**).
   - Both mathematical assertions passed.
4. **Test Assertion Brittleness:**
   - Unlike Case D (`test_pipeline.py` line 377) and Case E (`test_case_e_slash.py` line 42), which explicitly normalize `"x"`, `"times"`, and `"*"` before comparison:
     ```python
     actual_symbols = ["x" if s in ("times", "*", "×") else s for s in res["predicted_symbols"]]
     ```
   - Case C (line 340) had a hardcoded assertion requiring the exact literal string `"times"`.
5. **Safeguard Enforcement:**
   - MathVision governance strictly prohibits editing, skipping, or weakening existing test assertions during integration.
   - Because `test_pipeline_failure_case_c_end_to_end` failed, **promotion of V4 CNN was immediately halted and rolled back**.

---

## 6. Production Rollback Verification

In strict accordance with the mandatory rollback procedure:
1. `models/cnn_mathvision.keras` was restored from `models/cnn_mathvision.keras.pre_phase5_backup`.
2. The restored checkpoint hash was re-calculated and verified:
   - **Restored Hash:** `65d10826b59133896efa12a7c3acb9f1475b1486d74acf1abbee9b15d1bb8cce`
   - **Baseline Match:** **`100% VERIFIED`**.
3. All bytecode caches (`src/__pycache__`) were purged.
4. Full pytest suite was rerun:
   - **Command:** `python -m pytest tests/ -v`
   - **Result:** **`125 passed in 41.41s`** (100% pass rate, 0 failures, 0 regressions).
5. All 9 physical acceptance cases verified: **9/9 passed (100.0%)**.
6. All 20 mathematical benchmarks verified: **20/20 passed (100.0%)**.

---

## 7. Current Production State Summary

| Component | Active Production Status | Cryptographic Hash / Diff |
| :--- | :--- | :--- |
| **Pipeline Code (`src/pipeline.py`)** | Stage 3.5 Topological Heuristic Integrated | Fully functional, zero regressions |
| **Active CNN Checkpoint (`models/cnn_mathvision.keras`)** | Baseline CNN Restored | `65d10826b59133896efa12a7c3acb9f1475b1486d74acf1abbee9b15d1bb8cce` |
| **Automated Test Suite (`pytest`)** | **125 / 125 Passed (100.0%)** | 0 failures, 0 errors |
| **Physical Acceptance Cases (A–H2)** | **9 / 9 Passed (100.0%)** | 0 failures |
| **20-Equation Benchmark** | **20 / 20 Passed (100.0%)** | 0 failures |

---

## 8. Final Recommendation & Next Steps

### 8.1 Verdict: **PARTIALLY PROMOTED & SAFELY ROLLED BACK**
1. **The Topological Disambiguation Heuristic in `src/pipeline.py` is Production-Ready:** It passed all 125 tests, all 9 physical acceptance cases, and all 20 benchmarks with zero defects.
2. **Model V4 Checkpoint Promotion is BLOCKED Pending Test Normalization:** Model V4 cannot be promoted while `test_pipeline_failure_case_c_end_to_end` strictly asserts literal string `"times"` instead of allowing `"x"`.
3. **Path Forward for Future Checkpoint Promotion:**
   - If authorized in a subsequent phase, normalize the token assertion in `test_pipeline.py::test_pipeline_failure_case_c_end_to_end` to accept either `"times"` or `"x"` (identical to existing assertions in Case D and Case E).
   - Once that test is updated to match Case D/E parity, Model V4 will pass 125/125 tests and can be promoted safely.

*The codebase is in a stable, verified, 100% green state with zero regressions.*

---

## 9. Phase 5 Continuation — Authorized Solution C Implementation

Following the Phase 5.1 Token Representation Compatibility Investigation ([`docs/phase5_1_token_contract_investigation.md`](file:///C:/Users/Jainam/Downloads/CV%20Project/docs/phase5_1_token_contract_investigation.md)), the user explicitly authorized applying **Solution C only** to `tests/test_pipeline.py` for Case C, aligning its multiplication-alias normalization with Cases D and E.

### 9.1 Test Correction Applied
- **File:** `tests/test_pipeline.py` (lines 339-350)
- **Scope:** Modified only the Case C multiplication-token assertion to accept established multiplication aliases: `times`, `x`, `*`, and `×`.
- **Integrity Preserved:** Kept every other expected symbol exact (`["4", "5", "div", "5", "+", "6", ..., "2"]`). Preserved all count, localization, parsing, and symbolic answer assertions (`status == "solved"`, `final_answer == "21"`).
- **Production Code:** Zero production code, models, or datasets modified in this step.

```diff
--- a/tests/test_pipeline.py
+++ b/tests/test_pipeline.py
@@ -339,9 +339,16 @@ def test_pipeline_failure_case_c_end_to_end():
     # 2. Predicted Symbols
-    assert res["predicted_symbols"] == ["4", "5", "div", "5", "+", "6", "times", "2"], (
-        f"Incorrect predictions: {res['predicted_symbols']}"
-    )
+    expected_symbols = ["4", "5", "div", "5", "+", "6", "times", "2"]
+    actual_symbols = [
+        "x" if s in ("times", "*", "×") else s
+        for s in res["predicted_symbols"]
+    ]
+    expected_normalized = [
+        "x" if s in ("times", "*", "×") else s
+        for s in expected_symbols
+    ]
+    assert actual_symbols == expected_normalized, f"Incorrect predictions: {res['predicted_symbols']}"
```

---

## 10. Stage 2 — Validation of Corrected Baseline

Prior to model promotion, the baseline system was re-validated with the corrected test assertion:
1. **Original Production CNN Hash:** `65d10826b59133896efa12a7c3acb9f1475b1486d74acf1abbee9b15d1bb8cce` (**100% Match**).
2. **Held-Out Test Split Hash:** `14cef4cef2ae6abc712ed1f00a32ddcb55e8a01efe151bbab85a2dc0efc82f79` (**100% Match**).
3. **Full Pytest Suite:** **126 / 126 Passed in 169.66s** (100% green).
4. **Physical Acceptance Cases (A–H2):** **9 / 9 Passed (100.0%)**.
5. **20-Equation Benchmark:** **20 / 20 Passed (100.0%)**.
6. **Non-Multiplication Assertions:** Confirmed 100% unweakened.

---

## 11. Stage 3 — Checkpoint Promotion of Model V4

With Stage 2 passed with zero defects, Model V4 was promoted to production:
1. **Verified Backup Checkpoint:** `models/cnn_mathvision.keras.pre_phase5_backup` (SHA-256: `65d10826b59133896efa12a7c3acb9f1475b1486d74acf1abbee9b15d1bb8cce`).
2. **Verified V4 Source Checkpoint:** `models/cnn_experimental_v4.keras` (SHA-256: `5d2559c57477cc0646f696a0bff781145bd728a1be5bd630478faaf33e9b3d56`).
3. **Atomic Replacement:** `models/cnn_experimental_v4.keras` was copied to a temporary staging file and atomically moved into `models/cnn_mathvision.keras`.
4. **Post-Promotion Hash Verification:**
   - Promoted `models/cnn_mathvision.keras` SHA-256: `5d2559c57477cc0646f696a0bff781145bd728a1be5bd630478faaf33e9b3d56`.
   - Match: **100% CONFIRMED**.
5. **Topological Disambiguation Heuristic:** Stage 3.5 heuristic in `src/pipeline.py` kept active and unchanged.

---

## 12. Stage 4 — Comprehensive Post-Promotion Validation

Full validation was executed against the newly promoted production checkpoint:

### 12.1 Automated Test Suite
- **Command:** `python -m pytest`
- **Result:** **`126 passed in 181.89s (0:03:01)`**
- **Failures:** 0
- **Regressions:** 0

### 12.2 Physical Acceptance Cases (9 / 9 Passed)
| Case ID | Input Image | Ground Truth Expression & Value | Pipeline Recognized Expression | Final Solved Answer | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Case A** | `outputs/real_test_case/01_original.png` | `12 + 8 ÷ 4 = 14` | `12+8/4` | `14` | **PASS** |
| **Case B** | `outputs/failure_case_b/01_original.png` | `36 ÷ 6 + 8 = 14` | `36/6+8` | `14` | **PASS** |
| **Case C** | `outputs/failure_case_c/01_original.png` | `45 ÷ 5 + 6 × 2 = 21` | `45/5+6*2` | `21` | **PASS** |
| **Case D** | `outputs/failure_case_d/01_original.png` | `7 × 8 + 12 ÷ 4 = 59` | `7*8+12/4` | `59` | **PASS** |
| **Case E** | `outputs/failure_case_e/01_original.png` | `7 × 8 + 12 ÷ 4 = 59` | `7*8+12/4` | `59` | **PASS** |
| **Case F** | `outputs/failure_case_f/01_original.png` | `(18 + 6) / 4 = 6` | `(18+6)/4` | `6` | **PASS** |
| **Case G** | `outputs/failure_case_g/01_original.png` | `(18 + 6) / 4 = 6` | `(18+6)/4` | `6` | **PASS** |
| **Case H1**| `outputs/failure_case_h/01_original.png` | `23 + 15 = 38` | `23+15` | `38` | **PASS** |
| **Case H2**| `outputs/failure_case_h2/01_original.png` | `23 + 15 = 38` | `23+15` | `38` | **PASS** |

### 12.3 Mathematical Equation Benchmarks (20 / 20 Passed)
- All 20 complex multi-operator and parenthesized equations passed symbolic evaluation (100.0% accuracy).

### 12.4 Held-Out Test Split Metrics (2,850 Samples)
Evaluated on `models/test_split.npz` (SHA-256: `14cef4cef2ae6abc712ed1f00a32ddcb55e8a01efe151bbab85a2dc0efc82f79`):

| Metric | Original Production Baseline (Config A) | Promoted Production Target (Config D: V4 + Heuristic) | Net Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 94.18% (2,684 / 2,850) | **94.35% (2,689 / 2,850)** | **+0.18% (+5 samples)** |
| **Macro F1** | 93.70% | **94.08%** | **+0.38%** |
| **Weighted F1** | 94.29% | **94.45%** | **+0.16%** |
| **`+` $\to$ `div` False Conversions** | 0 | **0** | **0 (Zero regression)** |
| **`div` $\to$ `+` False Conversions** | 0 | **0** | **0 (Zero regression)** |
| **Physical Cases Pass Rate** | 88.9% (8/9, Case E failed) | **100.0% (9/9, Case B & E pass)** | **+11.1% (+1 case)** |
| **Average Latency** | 118.5 ms/sample | **119.3 ms/sample** | +0.8 ms |
| **P95 Latency** | 127.4 ms/sample | **128.2 ms/sample** | +0.8 ms |

---

## 13. Final Production Architecture & Verification Summary

| Component | Final Promoted Production State | Cryptographic SHA-256 / Verification Status |
| :--- | :--- | :--- |
| **Active Production CNN** | **Model V4 CNN** (`models/cnn_mathvision.keras`) | `5d2559c57477cc0646f696a0bff781145bd728a1be5bd630478faaf33e9b3d56` |
| **Pipeline Implementation** | **Stage 3.5 Topological Heuristic Integrated** (`src/pipeline.py`) | Active, fully verified |
| **Test Suite Alignment** | **Aligned Case C Alias Normalization** (`tests/test_pipeline.py`) | Active, fully verified |
| **Pre-Implementation Backup** | Preserved in `models/cnn_mathvision.keras.pre_phase5_backup` | `65d10826b59133896efa12a7c3acb9f1475b1486d74acf1abbee9b15d1bb8cce` |
| **Held-Out Test Dataset** | Preserved in `models/test_split.npz` | `14cef4cef2ae6abc712ed1f00a32ddcb55e8a01efe151bbab85a2dc0efc82f79` |
| **Test Suite Pass Rate** | **126 / 126 Passed (100.0%)** | 0 failures, 0 regressions |
| **Physical Acceptance Pass Rate** | **9 / 9 Passed (100.0%)** | 0 failures |
| **Equation Benchmark Pass Rate** | **20 / 20 Passed (100.0%)** | 0 failures |
| **Rollback Status** | **NOT REQUIRED** | All validation gates passed with zero defects |

**Conclusion:** MathVision Phase 5 Authorized Production Integration is **100% COMPLETE AND SUCCESSFUL**. Model V4 CNN + Stage 3.5 Topological Disambiguation Heuristic is now the active, verified production system.
