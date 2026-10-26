# MathVision: Experimental Evaluation & Benchmark Report

## 1. Symbol Recognition Benchmark (Held-out Test Split)

| Model Architecture | Feature Representation | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **Deep CNN (Ours)** | Learned Hierarchical Convolutions | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Baseline HOG + SVM** | Handcrafted HOG Gradients (L2-Hys) | 99.81% | 99.82% | 99.81% | 99.81% |

### Key Viva Insights:
- **Performance Delta:** The Deep CNN outperforms the handcrafted HOG+SVM baseline by **+0.19%** in accuracy.
- **Topological Invariance:** The CNN's max-pooling and spatial dropout layers confer superior invariance to stroke thickness variations and slight handwriting deformities compared to rigid HOG cell blocks.

---

## 2. Preprocessing Thresholding Comparison

| Preprocessing Method | Thresholding Type | Theoretical Assumption | End-to-End Equation Accuracy |
|:---|:---|:---|:---:|
| **Adaptive Gaussian (Selected)** | Dynamic Local Neighborhood | Accommodates shadows, gradients, and uneven lighting | **80.0%** |
| **Otsu's Thresholding** | Global Bimodal Minimization | Assumes uniform bimodal intensity histogram | 80.0% |
| **Fixed Global Thresholding** | Static Cutoff ($T=127$) | Highly sensitive to ambient lighting conditions | 0.0% |

---

## 3. End-to-End Equation-Level Accuracy

- **Total Test Sample Equations:** 10
- **Completely Solved Equations:** 8
- **End-to-End Equation Recognition Rate:** **80.00%**

### Per-Equation Sample Breakdown:

| Image File | Expected Canonical Equation | Predicted Equation | Status |
|:---|:---|:---|:---:|
| `sample_01.png` | `2+3*4=14` | `2+3*4=14` | **PASS** |
| `sample_02.png` | `3*x+5=20` | `3*x+5=20` | **PASS** |
| `sample_03.png` | `(8-2)/2=3` | `(8-2)*(2=3` | **FAIL** |
| `sample_04.png` | `x**2-5*x+6=0` | `x**2-5*x+6=0` | **PASS** |
| `sample_05.png` | `7-2*x=1` | `7-2*x=1` | **PASS** |
| `sample_06.png` | `4*x+8=24` | `4*x+8=24` | **PASS** |
| `sample_07.png` | `15/(2+1)=5` | `15*((2+1)=5` | **FAIL** |
| `sample_08.png` | `x+9=12` | `x+9=12` | **PASS** |
| `sample_09.png` | `6*7-10=32` | `6*7-10=32` | **PASS** |
| `sample_10.png` | `2*x-7=3` | `2*x-7=3` | **PASS** |

---
*Report generated automatically by MathVision Evaluation Engine.*
