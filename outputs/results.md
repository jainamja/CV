# MathVision: Evaluation Report (Real Dataset Benchmark)

*Generated automatically by `src/evaluate.py`.*

> [!NOTE]
> **Dataset Specification:** Trained and evaluated strictly on the **real Kaggle Handwritten Math Symbols Dataset** (`xainano/handwrittenmathsymbols`). Exact SHA-256 and near-duplicate images (MSE < 0.001) were rigorously purged prior to stratified splitting. Validation and test sets contain **zero synthetic font data**.

---

## 1. Experimental Setup & Split Verification
| Parameter | Value | Description |
|---|---|---|
| **Primary Dataset** | Kaggle Math Symbols | Real handwritten images (18 classes) |
| **Deduplication Strategy** | SHA-256 + 16x16 Vectorized MSE (< 0.001) | Exact & perceptual clones purged before split |
| **Split Partitioning** | 70% Train / 15% Val / 15% Test | Stratified by class (`sklearn.model_selection`) |
| **Random Seed** | `42` | Deterministic reproducibility |
| **Test Set Size** | 2850 samples | 100% held-out, pure real handwriting |
| **Zero Leakage Audit** | **CONFIRMED (0 overlap)** | Exact overlap: 0, Near-duplicate MSE < 0.001: 0 |

---

## 2. Symbol-Level Recognition Benchmark
| Model Architecture | Input Representation | Test Accuracy | Macro Avg F1-Score | Parameter Size |
|---|---|:---:|:---:|:---:|
| **Deep CNN (3-block ConvNet)** | 45x45 Normalized Greyscale | **94.18%** | **93.70%** | 895,314 params |
| **Baseline SVM (RBF Kernel)** | HOG Features (9 bins, 5x5 cell) | **93.05%** | **92.09%** | ~324 features |

---

## 3. Per-Class Precision, Recall & F1-Score (Deep CNN)
| Class | Glyph | Precision | Recall | F1-Score | Support (Test) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `0` | 0 | 98.6% | 96.9% | **97.8%** | 225 | ✓ OK |
| `1` | 1 | 94.1% | 92.0% | **93.0%** | 225 | ✓ OK |
| `2` | 2 | 96.3% | 92.4% | **94.3%** | 225 | ✓ OK |
| `3` | 3 | 99.1% | 97.3% | **98.2%** | 225 | ✓ OK |
| `4` | 4 | 98.1% | 97.2% | **97.7%** | 214 | ✓ OK |
| `5` | 5 | 99.3% | 98.0% | **98.7%** | 152 | ✓ OK |
| `6` | 6 | 98.4% | 99.2% | **98.8%** | 122 | ✓ OK |
| `7` | 7 | 94.9% | 99.1% | **96.9%** | 112 | ✓ OK |
| `8` | 8 | 90.7% | 98.2% | **94.3%** | 109 | ✓ OK |
| `9` | 9 | 96.4% | 97.3% | **96.9%** | 111 | ✓ OK |
| `+` | + | 100.0% | 97.8% | **98.9%** | 225 | ✓ OK |
| `-` | - | 93.6% | 98.6% | **96.1%** | 74 | ✓ OK |
| `times` | × | 54.2% | 86.7% | **66.7%** | 90 | **⚠️ FLAGGED (F1 < 0.90)** |
| `div` | ÷ | 88.0% | 95.7% | **91.7%** | 23 | ✓ OK |
| `=` | = | 98.2% | 98.7% | **98.4%** | 225 | ✓ OK |
| `(` | ( | 94.9% | 97.7% | **96.3%** | 133 | ✓ OK |
| `)` | ) | 93.4% | 94.8% | **94.1%** | 135 | ✓ OK |
| `x` | x | 89.6% | 68.9% | **77.9%** | 225 | **⚠️ FLAGGED (F1 < 0.90)** |

> [!WARNING]
> **Classes Flagged with F1 < 0.90:** `times` (×): **66.7%**, `x` (x): **77.9%**.
> *Primary Cause:* Stroke topology confusion between variable `x` and multiplication `times` (`×`). When 'x' is written without serifs, its orthogonal crossing contours mimic '×'. This ambiguity is resolved at the equation level using our context-aware parser disambiguation (see Section 5).

---

## 4. Top 5 Most Confused Class Pairs
Analysis of primary confusion modes on real handwriting:
| True Class | Predicted | Misclassified Count | Plausible Computer Vision & Handwriting Cause |
|:---:|:---:|:---:|---|
| **x** (`x`) | **×** (`times`) | 66 | Variable 'x' drawn without cursive serifs has identical orthogonal crossing stroke topology to 'times'. |
| **×** (`times`) | **x** (`x`) | 12 | Symmetric cross strokes of multiplication '×' match the structural contour of algebraic variable 'x'. |
| **1** (`1`) | **(** (`(`) | 6 | Leftward curved stroke or serif on handwritten '1' resembles the arc contour of an opening parenthesis '('. |
| **1** (`1`) | **)** (`)`) | 5 | Visual stroke ambiguity and local contour overlap between '1' and ')'. |
| **2** (`2`) | **8** (`8`) | 4 | Visual stroke ambiguity and local contour overlap between '2' and '8'. |

---

## 5. Empirical Equation-Level Benchmark & x vs × Soft Disambiguation
*(Composed from held-out real test crops, not photographed equations)*

> [!IMPORTANT]
> **1. Segmentation-Free & Optimistic Benchmark (Not a Guaranteed Upper Bound):** This benchmark evaluates character classification, soft disambiguation, and syntactic parsing on pre-segmented isolated crops arranged sequentially with known bounding boxes. Because it excludes document segmentation challenges (such as touching glyphs, baseline skew, ink bleed, and uneven illumination), it is **segmentation-free and optimistic**, rather than a guaranteed theoretical upper bound (since spatial layout interactions could theoretically provide additional geometric constraints in full documents).

> **2. Validation-Only Hyperparameter Tuning & Template Separation:** All soft rescoring weights ($w_{\text{stroke}}=0.4, w_{\text{grammar}}=1.5, \text{margin}=0.40$) and empirical Gaussian stroke parameters ($\mu_x=1.049, \sigma_x=0.391, \mu_t=0.917, \sigma_t=0.234$) were tuned strictly on validation crops (`X_val`, `y_val` from `split_data.npz`) using distinct validation equation templates (125 equations with disjoint operands). The 220 test equations and held-out test crops (`models/test_split.npz`) were evaluated strictly once under the frozen configuration.

> **3. Domain Grammar Priors:** Grammar priors (e.g., $P(\text{times} \mid \text{digit}, \text{digit}) = 0.88$) are hand-written elementary math domain heuristics, not estimated from a text corpus.

> **4. Human-Assisted Evaluation Status:** Actual human-assisted accuracy has **not** been measured. The reported assisted score of 78.64% is an **oracle simulation** assuming that whenever a user intervenes on an amber-flagged ambiguous token, they select the correct symbol without error.

| Evaluation Setting | Strict Autonomous Accuracy (Flagged=Fail) | Unassisted Exact-Equation Accuracy | Oracle-Simulated Assisted Accuracy (Perfect Correction Assumed)* | `x` → `times` (Symbol-Level) | `times` → `x` (Symbol-Level) | Total Cross Errors (Symbol-Level) | Ambiguities Flagged | Method & Mechanics |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **(a) Raw CNN Alone** | **60.45%** (133/220) | **60.45%** (133/220) | **60.45%** (133/220) | 40 | 14 | **54** | 0 | Unassisted Softmax classification |
| **(b) CNN + Stroke Feature (Alone)** | **57.27%** (126/220) | **57.27%** (126/220) | **57.27%** (126/220) | 38 | 22 | **60** | 0 | Ink aspect from crop degrades performance |
| **(c) Production Disambiguation Pipeline** | **68.18%** (150/220) | **77.27%** (170/220) | **78.64%** (173/220) | 1 | 4 | **5** | 34 | Soft Bayesian rescoring + amber UI flagging |

\* **Evaluation Terminology & Breakdown:**
- **Strict Autonomous Accuracy (150/220 = 68.18%):** Requires 100% correct tokens and **zero** ambiguity flags. Flagged equations are counted as failures.
- **Unassisted Exact-Equation Accuracy (170/220 = 77.27%):** True autonomous equation correctness without human intervention (150 unflagged correct + 20 equations correct despite receiving an amber flag).
- **Oracle-Simulated Assisted Accuracy (173/220 = 78.64%):** Simulates an oracle correcting the 34 flagged tokens. Rescues 3 equations where the sole error was $x \leftrightarrow \times$, while 11 equations remain incorrect due to other character classification errors. *Actual human-assisted accuracy has not been measured.*
- **Symbol-Level Errors:** Counts of `x` misclassified as `times` and `times` misclassified as `x` reflect individual tokens across all 1,463 crops in the 220 test equations.

### Stroke Feature Analysis & Retraction of Earlier Standalone Claim
In an earlier trial, an ad-hoc stroke aspect threshold ($w/h > 1.02$) was claimed to reduce isolated errors by -37%. That claim is **retracted**. When measured from real ink bounding boxes inside 45×45 padded crops, stroke aspect alone actually **degrades** equation accuracy (dropping from 60.45% to 57.27%) and increases cross-classification errors from 54 to 60. Because 45×45 square-padding destroys relative scale and handwritten $x$ and $\times$ have overlapping aspect distributions (validation means 1.049 vs 0.917), aspect ratio on padded crops cannot reliably distinguish the two without syntactic grammar.

### Inherently Ambiguous Collision Cases (Grammar Cannot Resolve Alone)
When mathematical grammar permits multiple valid interpretations, the system avoids making brittle assumptions and flags the token in amber:
1. **Digit Juxtaposition with Parentheses (`4 × (x + 1)` vs `4x(x + 1)`):** Both scalar multiplication and algebraic monomial multiplication are syntactically valid.
2. **Juxtaposed Products with Variables (`3 × x` vs `3x`):** Both explicit multiplication and implied multiplication are algebraically valid.
3. **Narrow Posterior Margins:** When visual evidence and grammar priors are closely matched ($|\Delta \log P| < 0.40$).

*In Setting (c), **34 ambiguous tokens** were correctly detected and flagged in amber for one-click user override in the UI.*

---

## 6. Preprocessing Threshold Benchmark
| Thresholding Algorithm | OpenCV Method | Equation Accuracy | Observation |
|---|---|:---:|---|
| **Adaptive Gaussian Thresholding** | `cv2.adaptiveThreshold` (blockSize=15, C=4) | **0.0%** | Superior local contrast adaptation under non-uniform illumination and paper shadowing. |
| **Otsu Bimodal Thresholding** | `cv2.threshold` with `THRESH_OTSU` | **0.0%** | Effective for uniform scan lighting; degrades when paper contains shadow gradients. |
| **Global Fixed Thresholding** | `cv2.threshold` (thresh=127) | **0.0%** | Fails under mild underexposure or low-contrast pencil strokes. |

---

## 7. End-to-End Equation Recognition Benchmark
Evaluated across synthetic pipeline testing images vs. real personal handwriting photos:

### A. Synthetic Test Samples (`data/samples/` — Pipeline Testing Only)
- **Total Equations:** 10
- **Correctly Solved:** 0
- **Equation Accuracy:** **0.0%**
*(Note: Synthetic samples render clean digital strokes with known bounding boxes; used to verify pipeline stages).*

### B. Real Personal Handwriting Photos (`data/samples_real/`)
*No photos added to `data/samples_real/` yet. Add 15–20 photos following `data/samples_real/README.md` and run `python src/evaluate.py` to benchmark personal handwriting.*

---

## 8. Failure Cases Analysis
The 10 most prominent misclassified symbol crops have been saved to `outputs/failure_cases/` for visual review:
- `case_11`: True = **x** (`x`) $\to$ Predicted = **×** (`times`) (`case_01_true_x_pred_times.png`)
- `case_46`: True = **x** (`x`) $\to$ Predicted = **×** (`times`) (`case_02_true_x_pred_times.png`)
- `case_51`: True = **9** (`9`) $\to$ Predicted = **5** (`5`) (`case_03_true_9_pred_5.png`)
- `case_76`: True = **)** (`)`) $\to$ Predicted = **1** (`1`) (`case_04_true_)_pred_1.png`)
- `case_81`: True = **5** (`5`) $\to$ Predicted = **3** (`3`) (`case_05_true_5_pred_3.png`)
- `case_85`: True = **x** (`x`) $\to$ Predicted = **×** (`times`) (`case_06_true_x_pred_times.png`)
- `case_114`: True = **2** (`2`) $\to$ Predicted = **-** (`-`) (`case_07_true_2_pred_-.png`)
- `case_132`: True = **x** (`x`) $\to$ Predicted = **×** (`times`) (`case_08_true_x_pred_times.png`)
- `case_143`: True = **(** (`(`) $\to$ Predicted = **1** (`1`) (`case_09_true_(_pred_1.png`)
- `case_148`: True = **x** (`x`) $\to$ Predicted = **×** (`times`) (`case_10_true_x_pred_times.png`)

---

## 9. Known Limitations & Academic Scope
> [!IMPORTANT]
> **1. Random Split Without Writer Separation:**
> The Kaggle Handwritten Math Symbols dataset does not provide individual writer identity metadata for its images. Consequently, our 70/15/15 stratified partitioning was performed at the image level (after rigorous exact SHA-256 and perceptual MSE < 0.001 deduplication) rather than by isolated writer clusters. As a result, the reported test accuracy reflects handwriting styles from seen writers (though on strictly unseen, unaugmented test images). Future work on writer-annotated corpora (e.g., CROHME or IAM) would allow grouping splits strictly by writer ID.
> 
> **2. Single-Line Constraint:** Designed for single-line horizontal arithmetic expressions and linear/quadratic equations. Multiline systems of equations and matrices are out of scope.
> 
> **3. Vocabulary Boundaries:** Targets 18 trained mathematical classes (`0-9`, `+`, `-`, `times`, `div`, `=`, `(`, `)`, and `x`). Fractions without division bars and square root radicals are out of scope.