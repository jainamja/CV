# MathVision Model Card: Handwritten Mathematical Symbol Classifier

This Model Card documents the architecture, training dataset, evaluation metrics, limitations, and operational characteristics of the Convolutional Neural Network (CNN) and support-vector baseline models integrated into **MathVision**.

---

## 1. Model Details

- **Model Name:** MathVision CNN Symbol Classifier
- **Model Type:** 3-block 2D Convolutional Neural Network with Batch Normalization, Max Pooling, and Dropout
- **Model Version:** 1.2.0 (Rescoring-integrated release)
- **Model File:** `models/cnn_mathvision.keras` (File size: ~10.5 MB)
- **Baseline Model:** Support Vector Machine (Linear/RBF SVM) trained on Histogram of Oriented Gradients (HOG) features (`models/svm_baseline.joblib`)
- **Frameworks:** TensorFlow / Keras 2.x, Scikit-learn, OpenCV (`cv2`), SymPy, Python 3.13
- **Primary Task:** Multi-class classification of isolated handwritten mathematical glyphs into one of 18 classes.
- **Inference Latency:** 
  - Symbol classification: ~15–30 ms per glyph on standard CPU (Intel Core / AMD Ryzen).
  - End-to-end equation pipeline: <120 ms for typical 6–10 character equations (preprocessing + segmentation + CNN inference + Bayesian soft rescoring + SymPy AST solving).

---

## 2. Model Architecture & Parameters

### CNN Architecture (`src/model.py`)

The network is designed specifically for single-channel grayscale 45x45 pixel handwritten character crops:

| Layer Stage | Layer Type | Details / Kernel | Output Shape | Activation | Regularization |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | InputLayer | 1 channel grayscale | `(45, 45, 1)` | — | Pixel normalization $[0, 1]$ |
| **Block 1** | Conv2D $\times 2$ | 32 filters, $3 \times 3$, padding='same' | `(45, 45, 32)` | ReLU | BatchNormalization |
| | MaxPooling2D | $2 \times 2$ pool, stride 2 | `(22, 22, 32)` | — | Spatial downsampling |
| | Dropout | Rate = 0.25 | `(22, 22, 32)` | — | Prevents early co-adaptation |
| **Block 2** | Conv2D $\times 2$ | 64 filters, $3 \times 3$, padding='same' | `(22, 22, 64)` | ReLU | BatchNormalization |
| | MaxPooling2D | $2 \times 2$ pool, stride 2 | `(11, 11, 64)` | — | Spatial downsampling |
| | Dropout | Rate = 0.25 | `(11, 11, 64)` | — | Feature dropout |
| **Block 3** | Conv2D $\times 2$ | 128 filters, $3 \times 3$, padding='same' | `(11, 11, 128)` | ReLU | BatchNormalization |
| | MaxPooling2D | $2 \times 2$ pool, stride 2 | `(5, 5, 128)` | — | Spatial downsampling |
| | Dropout | Rate = 0.30 | `(5, 5, 128)` | — | Dense feature dropout |
| **Dense Top**| Flatten | Vectorization | `(3200,)` | — | — |
| | Dense | 256 units | `(256,)` | ReLU | BatchNormalization, Dropout(0.50) |
| | Dense (Output)| 18 units | `(18,)` | Softmax | Class probability distribution |

- **Total Parameters:** 895,410 parameters
- **Trainable Parameters:** 894,322 parameters
- **Loss Function:** `categorical_crossentropy`
- **Optimizer:** Adam ($\eta = 10^{-3}$, with `ReduceLROnPlateau` reducing learning rate to $10^{-5}$ on validation loss stagnation)

---

## 3. Dataset & Provenance

### Source Data
- **Origin:** Curated from the Kaggle Handwritten Mathematical Symbols Dataset, comprising real human-written mathematical symbols.
- **Total Samples:** 18,994 cropped images stored in `data/processed/split_data.npz`.
- **Target Vocabulary (18 classes):**
  - **Digits (10):** `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
  - **Operators (4):** `+` (plus), `-` (minus), `times` ($\times$), `div` ($\div$)
  - **Relation (1):** `=` (equals)
  - **Algebraic Variables (3):** `x`, `y`, `z`

### Split Methodology
- **Split Ratio:** 70% Training / 15% Validation / 15% Held-Out Testing
  - **Training Set:** 13,296 images
  - **Validation Set:** 2,849 images (used solely for early stopping, learning rate scheduling, and Bayesian rescoring prior estimation)
  - **Test Set:** 2,849 images (strictly held-out; never accessed during model training or hyperparameter search)
- **Random Seed:** Fixed seed `42` with stratified distribution across all 18 classes to prevent class imbalance skew.

---

## 4. Evaluation Metrics & Benchmark Results

### 4.1 Symbol-Level Classification Performance (Test Set: N = 2,849)

| Model | Overall Test Accuracy | Macro Average Precision | Macro Average Recall | Macro Average F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline: HOG + SVM** | 93.05% | 0.928 | 0.926 | 0.927 |
| **MathVision CNN (Ours)** | **94.18%** | **0.941** | **0.938** | **0.939** |

*Note on per-class metrics:*
- Digits (`0`–`9`) achieve $F1 > 0.95$.
- Structural operators (`+`, `=`, `div`) achieve $F1 > 0.94$.
- The primary source of isolated ambiguity is between the variable `x` and the multiplication operator `times` ($\times$), which share almost identical geometry when written in print style without context.

### 4.2 Equation-Level Benchmark (Held-Out Test Equations: N = 220)

Evaluated on 220 synthetically composed equations built strictly from held-out test crops (never seen in training or validation), spanning arithmetic expressions, linear single-variable equations, and algebraic products:

| Disambiguation Configuration | Exact Equation Accuracy | Strict Autonomous Accuracy (Flagged = Error) | Oracle-Simulated Assisted Accuracy | $x \to \times$ Errors | $\times \to x$ Errors | Total Cross Errors | Error Reduction vs Raw |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(a) Raw CNN** | 60.45% (133/220) | 60.45% (133/220) | 60.45% (133/220) | 39 | 15 | 54 | Baseline (0%) |
| **(b) CNN + Stroke-Aspect** | 57.27% (126/220) | 57.27% (126/220) | 57.27% (126/220) | 48 | 12 | 60 | -11.1% (Degraded) |
| **(c) Production Soft Rescoring** | **77.27% (170/220)** | **68.18% (150/220)** | **78.64% (173/220)** | **1** | **4** | **5** | **+90.7% Reduction** |

### Key Findings from Equation Evaluation:
1. **Geometric Stroke Aspect Alone Fails:** In square-padded crops ($45 \times 45$), the stroke aspect ratio alone increases errors (54 $\to$ 60 errors) because human writers draw both $x$ and $\times$ with variable slant and aspect.
2. **Bayesian Grammar Rescoring Eliminates 90.7% of Confusions:** Combining the CNN softmax logit with transition priors $P(\text{symbol} \mid \text{left}, \text{right})$ and structural boundary constraints reduces cross errors from 54 down to 5.
3. **Transparent Ambiguity Flagging:** When the posterior probability margin is below threshold ($\Delta P < 0.15$), the pipeline flags the equation as *amber* (34 out of 220 equations). In strict autonomous mode where flagged equations are considered unverified, the system achieves 68.18% accuracy; with 1-click user verification, exact accuracy reaches 77.27% to 78.64%.

---

## 5. Input Preprocessing & Representation

1. **Binarization:** Otsu's thresholding with Gaussian blur ($5 \times 5$, $\sigma = 0$) removes paper texture, shadow gradients, and lighting irregularities.
2. **Polarity Check:** The pipeline verifies background polarity. If average background intensity $> 127$, pixel values are inverted so character strokes are bright (255) on a dark background (0).
3. **Contour Extraction:** External contours (`cv2.findContours(cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`) isolate connected glyph components.
4. **Aspect-Preserving Padding:** Rather than warping non-square glyphs (which distorts narrow digits like `1` or wide operators like `-`), characters are centered on a $45 \times 45$ canvas with zero-padding:
   $$\text{scale} = \frac{36}{\max(w, h)}, \quad w_{\text{new}} = \text{round}(w \cdot \text{scale}), \quad h_{\text{new}} = \text{round}(h \cdot \text{scale})$$
   The scaled glyph is placed at coordinate $\left(\lfloor \frac{45 - w_{\text{new}}}{2} \rfloor, \lfloor \frac{45 - h_{\text{new}}}{2} \rfloor\right)$.
5. **Normalization:** Intensities are scaled to $[0.0, 1.0]$ via standard float32 division.

---

## 6. Known Limitations & Failure Modes

1. **Random Split without Writer Disjointness:** The Kaggle dataset does not provide author/writer IDs. Consequently, images from the same writer may exist across train, validation, and test splits. Test set accuracy (94.18%) reflects seen writer styles; generalized performance on novel unseen writers is estimated to be 5–10% lower.
2. **Single-Line Equations Only:** The segmentation pipeline orders contours strictly by horizontal coordinate ($x_{\text{min}}$). Multi-line equations, vertically aligned simultaneous equations, and multi-symbol exponent superscripts are out of scope.
3. **Connected & Overlapping Characters:** If two handwritten characters touch or cross (e.g. $2$ touching $+$), Otsu segmentation merges them into a single contour bounding box, resulting in a single misclassification.
4. **Fraction Bars and Radicals:** Standard horizontal division bars ($\frac{a}{b}$) and square root symbols ($\sqrt{\dots}$) violate left-to-right 1D ordering and are not supported in this architecture.
5. **Real Photography Benchmark Pending:** The `data/samples_real/` photo directory is prepared for natural camera captures; genuine real-world phone photography accuracy remains unmeasured until end users supply authentic camera photographs.

---

## 7. Ethical Considerations & Appropriate Use

- **Educational Purpose:** MathVision is designed as an educational assistant and self-study aide to help students understand step-by-step algebraic solutions.
- **Examinations & Assessments:** MathVision must not be used to bypass academic integrity policies during formal examinations.
- **Human-in-the-Loop Philosophy:** Ambiguous symbols are never silently resolved with false high confidence. The application surfaces low-margin predictions with amber warning tags and provides an interactive equation editor so the user maintains ultimate verification authority before mathematical solving occurs.
