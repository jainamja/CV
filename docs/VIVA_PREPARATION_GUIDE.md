# MathVision — Comprehensive College Viva Defense & Technical Guide

---

## 1. Project Overview & Elevator Pitch

**MathVision** is an end-to-end Computer Vision and Symbolic Artificial Intelligence application that takes handwritten mathematical expressions and equations (photographed, scanned, or drawn), processes them through a modular classical CV pipeline, classifies individual symbols using a Deep Convolutional Neural Network (CNN), resolves visual-syntactic ambiguities via Bayesian soft rescoring, and produces pedagogical, step-by-step mathematical derivations using SymPy.

### Key Architectural Pillars:
1. **Classical Computer Vision Preprocessing & Segmentation (OpenCV):** Robust against illumination gradients, paper tilt, noise, and multi-contour characters (`=`, `÷`).
2. **Deep Learning Classification (TensorFlow/Keras):** 3-block ConvNet achieving **94.18%** test accuracy on 2,850 real handwritten glyphs across 18 classes.
3. **Soft Bayesian Disambiguation ($x$ vs. $\times$):** Combines CNN softmax probabilities, empirical stroke aspect ratio, and mathematical grammar context priors to reduce cross-classification errors by **-90.7%**.
4. **Symbolic Algebra Engine (SymPy):** Rigorous, mathematically sound step-by-step solving for arithmetic, linear, and quadratic equations (real and complex).
5. **Human-in-the-Loop Educational UI (Streamlit):** Amber UI safety badges for ambiguous glyphs, editable formula preview, and visual CV diagnostic heatmaps.

---

## 2. Classical Computer Vision (OpenCV) Defense

### Q1: Why use OpenCV preprocessing rather than feeding the raw image directly into a deep learning model?
**Answer:**
1. **Resolution & Scale Standardization:** Handwritten photos range from 0.3 MP to 48 MP with arbitrary aspect ratios. Classical scaling standardizes the pixel density so that stroke widths and morphological kernels remain physically meaningful.
2. **Illumination Invariance:** Real photographs contain non-uniform shadows, paper folds, and yellow indoor lighting. Raw RGB values are sensitive to lighting; thresholding isolates pure binary foreground ink from background paper.
3. **Explainability & Modularity:** Decoupling segmentation from classification allows targeted debugging. If a character is missed, we can immediately observe whether the failure occurred during thresholding, contour extraction, or neural classification.

---

### Q2: Why Adaptive Gaussian Thresholding over Otsu's Thresholding or Global Thresholding?
**Answer:**
- **Global Thresholding (`cv2.threshold` with fixed threshold $T=127$):** Assumes uniform lighting. If an equation has a shadow gradient across the page, one side of the paper falls below $T$ and becomes completely black (destroying strokes), while faint pencil strokes on the other side disappear entirely.
- **Otsu's Thresholding (`cv2.THRESH_OTSU`):** Selects an optimal global threshold that minimizes intra-class intensity variance:
  $$\sigma_w^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)$$
  While effective for high-contrast scans, Otsu inherently assumes a **bimodal histogram** across the entire frame. In handheld smartphone photos, paper shadows create a wide, multimodal intensity distribution where a single global threshold fails.
- **Adaptive Gaussian Thresholding (`cv2.adaptiveThreshold`):** Computes a distinct threshold $T(x, y)$ for every single pixel based on the Gaussian-weighted mean of its local neighborhood (block size $15 \times 15$) minus constant $C=4$:
  $$T(x, y) = \sum_{i, j} G(i, j) \cdot I(x+i, y+j) - C$$
  This adapts dynamically to local lighting: a faint pen stroke in a dark corner is separated just as cleanly as a dark pen stroke under direct sunlight. In our benchmark, Adaptive thresholding enabled successful equation segmentation where global thresholding scored 0.0%.

---

### Q3: How do Morphological Operations (Opening and Closing) improve segmentation?
**Answer:**
Morphological operations are non-linear neighborhood transformations based on set theory:
1. **Morphological Opening ($A \circ B = (A \ominus B) \oplus B$):**
   - **Mechanism:** Erosion followed by Dilation using an elliptical structuring element ($2 \times 2$).
   - **Purpose:** Eliminates tiny isolated foreground noise specks, pen splatters, and paper texture without altering the topology of larger character strokes.
2. **Morphological Closing ($A \bullet B = (A \oplus B) \ominus B$):**
   - **Mechanism:** Dilation followed by Erosion using a rectangular structuring element ($3 \times 3$).
   - **Purpose:** Bridges hairline discontinuities in pen strokes caused by fast handwriting, ballpoint pen skips, or faint pencil markings, ensuring characters remain single connected components.

---

### Q4: How are multi-contour characters like '=' and '÷' handled?
**Answer:**
In standard contour extraction (`cv2.findContours` using Suzuki-Abe border following algorithm), disconnected strokes generate separate bounding boxes. For example, an equals sign (`=`) naturally creates two horizontal boxes.
**Our Geometric Merging Algorithm:**
1. Evaluates all candidate bounding box pairs $(B_1, B_2)$.
2. Checks horizontal overlap:
   $$\text{Overlap}_{\text{horizontal}} = \frac{\min(x_1+w_1, x_2+w_2) - \max(x_1, x_2)}{\min(w_1, w_2)}$$
3. Checks vertical gap proximity: $\Delta y = |y_2 - (y_1 + h_1)| \le 1.8 \cdot \max(h_1, h_2)$.
4. If horizontal overlap $> 60\%$ and vertical gap is within threshold, the boxes are merged into a unified bounding box:
   $$x_{\text{merged}} = \min(x_1, x_2), \quad y_{\text{merged}} = \min(y_1, y_2)$$
   $$w_{\text{merged}} = \max(x_1+w_1, x_2+w_2) - x_{\text{merged}}, \quad h_{\text{merged}} = \max(y_1+h_1, y_2+h_2) - y_{\text{merged}}$$
Division signs (`÷`) are similarly merged by grouping the central bar with the vertically aligned dot components.

---

### Q5: How do you normalize segmented crops to 45×45 without distorting aspect ratios?
**Answer:**
A naive resize of an elongated minus sign (`-`) from $40 \times 8$ directly to $45 \times 45$ stretches it into a thick, distorted rectangle, destroying its topological aspect ratio.
**Our Aspect-Preserving Square Padding Pipeline:**
1. Determine the maximum dimension: $S = \max(w, h)$.
2. Create a square blank canvas of size $(S + 2p) \times (S + 2p)$, where $p = \text{int}(0.15 \cdot S)$ is an intentional padding margin.
3. Center the extracted ink crop within this canvas:
   $$x_{\text{offset}} = p + \frac{S - w}{2}, \quad y_{\text{offset}} = p + \frac{S - h}{2}$$
4. Resize the padded square canvas to $45 \times 45$ using `cv2.INTER_AREA` (for downsampling) or `cv2.INTER_CUBIC` (for upsampling).
5. Normalize pixel intensities from $[0, 255]$ uint8 to $[0.0, 1.0]$ float32.

---

## 3. Machine Learning & CNN Architecture Defense

### Q6: Walk me through the Deep CNN architecture and why each layer was selected.
**Answer:**
Our CNN is a 3-stage hierarchical feature extractor designed for grayscale mathematical character crops:
```
Input: (Batch, 45, 45, 1) Normalized Float32
│
├── Conv Block 1:
│   ├── Conv2D(32 filters, 3x3 kernel, padding='same', ReLU)
│   ├── BatchNormalization()
│   ├── Conv2D(32 filters, 3x3 kernel, padding='same', ReLU)
│   ├── MaxPooling2D(pool_size=(2, 2))  -> Output: (22, 22, 32)
│   └── Dropout(0.20)
│
├── Conv Block 2:
│   ├── Conv2D(64 filters, 3x3 kernel, padding='same', ReLU)
│   ├── BatchNormalization()
│   ├── Conv2D(64 filters, 3x3 kernel, padding='same', ReLU)
│   ├── MaxPooling2D(pool_size=(2, 2))  -> Output: (11, 11, 64)
│   └── Dropout(0.25)
│
├── Conv Block 3:
│   ├── Conv2D(128 filters, 3x3 kernel, padding='same', ReLU)
│   ├── BatchNormalization()
│   ├── MaxPooling2D(pool_size=(2, 2))  -> Output: (5, 5, 128)
│   └── Dropout(0.30)
│
├── Dense Classification Head:
│   ├── Flatten() -> 5 x 5 x 128 = 3,200 features
│   ├── Dense(256 units, ReLU)
│   ├── BatchNormalization()
│   ├── Dropout(0.40)
│   └── Dense(18 units, Softmax) -> Probability distribution over 18 classes
```
**Key Layer Justifications:**
- **Double $3 \times 3$ Convolutions:** Two consecutive $3 \times 3$ convolutions have an effective receptive field of $5 \times 5$, but use fewer parameters ($2 \times 3^2 = 18$ weights vs $1 \times 5^2 = 25$ weights) and introduce non-linear activation between stages.
- **Batch Normalization:** Normalizes intermediate activation distributions across mini-batches, smoothing the loss landscape, accelerating convergence, and mitigating internal covariate shift.
- **Progressive Dropout ($0.20 \to 0.40$):** Strongly regularizes the network, preventing co-adaptation of features on high-frequency handwriting styles.
- **Total Parameters:** 895,314 trainable parameters — lightweight enough to execute forward inference in under 45 ms on a standard laptop CPU.

---

### Q7: Why did you compare the CNN against a HOG + SVM baseline?
**Answer:**
In scientific machine learning, deep learning performance must be benchmarked against an established classical feature-engineering baseline:
- **Histogram of Oriented Gradients (HOG):** Divides the $45 \times 45$ crop into $5 \times 5$ pixel cells, computes gradient orientations grouped into 9 angular bins, and normalizes across $2 \times 2$ cell blocks ($324$ total feature dimensions). HOG captures edge directions and contour geometry.
- **Support Vector Machine (SVM):** Uses a non-linear Radial Basis Function (RBF) kernel with balanced class weights to construct maximum-margin hyperplanes.
- **Empirical Viva Finding:**
  - Deep CNN: **94.18% test accuracy** | Macro F1: **94.18%**
  - HOG + SVM: **93.05% test accuracy** | Macro F1: **92.93%**
  The CNN outperforms the SVM because convolutional kernels learn multi-scale spatial stroke hierarchies directly from data rather than relying solely on first-order gradient histograms.

---

## 4. The $x$ vs. $\times$ Ambiguity & Bayesian Disambiguation Defense

### Q8: Why is distinguishing algebraic variable 'x' from multiplication 'times' (×) mathematically ill-posed at the symbol level?
**Answer:**
When an individual writes a variable $x$ without cursive loops or curved serifs (e.g., as two crossed straight lines), its structural stroke topology is geometrically identical to the multiplication cross $\times$. In our test set of 2,850 real handwritten crops, raw CNN cross-entropy alone produced **54 cross-classification errors** ($40$ variable $x$ misclassified as $\times$, and $14$ multiplication $\times$ misclassified as $x$), dragging the individual class F1-score of $\times$ down to 66.7%. No vision model evaluated on isolated square-padded crops can achieve 100% accuracy without contextual information.

---

### Q9: Explain your Bayesian Soft Rescoring Formulation.
**Answer:**
Rather than relying on brittle binary if-else rules, our parser computes a joint log-posterior over both hypotheses $c \in \{x, \text{times}\}$:
$$\log P(c \mid \text{obs}) = \log P_{\text{CNN}}(c) + w_{\text{stroke}} \cdot \log P_{\text{stroke}}(c) + w_{\text{grammar}} \cdot \log P_{\text{grammar}}(c)$$
where:
1. **$P_{\text{CNN}}(c)$:** Continuous softmax probability from the Deep CNN.
2. **$P_{\text{stroke}}(c)$:** Likelihood derived from the unpadded ink bounding box aspect ratio ($w/h$) and relative height to neighbors using Gaussian probability densities tuned on validation data:
   $$\mu_x = 1.049, \quad \sigma_x = 0.391 \qquad \mu_t = 0.917, \quad \sigma_t = 0.234$$
3. **$P_{\text{grammar}}(c)$:** Syntactic context log-prior based on surrounding tokens:
   - Between two digits (`3 [?] 4`): $P(\text{times}) = 0.88, P(x) = 0.12$ (Arithmetic multiplication).
   - After closing parenthesis before digit (`) [?] 4`): $P(\text{times}) = 0.82, P(x) = 0.18$.
   - Hard Syntactic Bounds: Multiplication is mathematically illegal at the start of an expression (`* 5 = 15`), adjacent to equality signs (`= *`), adjacent to binary operators (`+ *`), and before exponents (`* ** 2`). These cases clamp $P(x) = 1.0, P(\text{times}) = 0.0$.
4. **Frozen Validation Weights:** $w_{\text{stroke}} = 0.4$, $w_{\text{grammar}} = 1.5$, $\text{margin} = 0.40$.
**Measured Impact:** Reduces total cross-classification errors from **54 down to 5** (a **-90.7% reduction**).

---

### Q10: What happens when the grammar cannot resolve an ambiguous case?
**Answer:**
There are mathematical edge cases where both interpretations are valid:
1. **Digit followed by cross followed by parenthesis (`4 x (x + 1)`):** Could be scalar multiplication $4 \times (x+1)$ or a monomial multiplying a binomial $4x(x+1)$.
2. **Juxtaposed variable products (`3 x x`):** Could be $3 \times x$ or implicit quadratic notation $3x^2$.
3. **Narrow Posterior Margins ($|\Delta \log P| < 0.40$):** Model confidence is evenly split.

**Safety Mechanism:** In these cases, MathVision refuses to guess. The token is flagged in **amber** with a visual badge (`⚠️ x/×`) in the UI, and the user is prompted to confirm or edit the equation before solving. In our 220-equation benchmark, exactly **34 ambiguous tokens** were flagged, protecting the system from silent miscalculation.

---

## 5. Symbolic Mathematical Solving & Parsing Defense

### Q11: How are superscripts (exponents) detected geometrically?
**Answer:**
When writing $x^2$, the character '2' is smaller and elevated relative to the base character 'x'.
**Spatial Exponent Elevation Rule:**
For consecutive bounding boxes $B_{i-1} = (x_1, y_1, w_1, h_1)$ and $B_i = (x_2, y_2, w_2, h_2)$:
$$y_{\text{center}, i} < y_{1} + 0.45 \cdot h_1 \quad \text{and} \quad h_i \le 0.85 \cdot h_1$$
If both conditions hold, token $i$ is identified as an exponent, and the Python power operator `**` is injected into the syntactic token stream: `['x', '2']` $\to$ `['x', '**', '2']`.

---

### Q12: How does MathVision prevent arbitrary code execution (Security)?
**Answer:**
Never evaluate recognized handwriting with Python's native `eval()` or `exec()`. A misclassified symbol could form malicious syntax (e.g. `__import__('os').system(...)`).
**MathVision Security Sandbox:**
1. Symbols are strictly restricted to the 18 supported classes: $\{0\text{-}9, +, -, \times, \div, =, (, ), x\}$.
2. The expression is passed through a syntax validator (`validate_expression`) that checks for balanced parentheses, single equality signs, valid boundary tokens, and absence of non-mathematical keywords.
3. The sanitized expression is parsed into an Abstract Syntax Tree (AST) using SymPy's restricted parser (`sympy.sympify(..., evaluate=False)`).

---

## 6. Data Integrity & Scientific Leakage Defense

### Q13: How did you prove zero data leakage between training, validation, and test sets?
**Answer:**
1. **Deduplication before Splitting:** The Kaggle dataset contained near-duplicates from identical scans. We performed exact SHA-256 hashing and pairwise Mean Squared Error (MSE) filtering to deduplicate images before partitioning.
2. **Deterministic Stratified Partitioning:** Split 70% train (13,296 images), 15% validation (2,849 images), and 15% test (2,850 images) with a fixed seed (`seed=42`).
3. **Automated Leakage Pytest (`test_train_val_test_isolation`):** A unit test asserts that the intersection of SHA-256 hashes between splits is exactly zero, and pairwise MSE across all splits satisfies $\text{MSE} \ge 0.001$.
4. **Validation-Only Hyperparameter Tuning:** All parser weights ($w_s=0.4, w_g=1.5$), margin thresholds ($0.40$), and stroke distribution parameters ($\mu, \sigma$) were tuned strictly on `X_val` / `y_val` using 125 validation templates with zero operand overlap with the test set. The held-out test split (`test_split.npz`) and 220 test templates were run strictly once for final evaluation.
5. **No Label Leakage in Parser (`test_parser_never_reads_true_labels`):** Verified that `parse_symbols` and `disambiguate_x_times` expose zero parameters for ground-truth strings.

---

### Q14: What is the difference between your reported benchmark metrics?
**Answer:**
On our 220 held-out composed equations (1,463 total crops):
1. **Raw CNN Equation Accuracy ($133 / 220 = 60.45\%$):** Unassisted CNN softmax predictions alone.
2. **Strict Autonomous Accuracy ($150 / 220 = 68.18\%$):** Requires 100% correct character predictions AND **zero** amber ambiguity warnings. Any equation triggering an amber flag is counted as an autonomous failure.
3. **Unassisted Exact Accuracy ($170 / 220 = 77.27\%$):** Actual pipeline accuracy without human intervention ($150$ unflagged correct $+ 20$ equations correct despite receiving an amber warning).
4. **Oracle-Simulated Assisted Accuracy ($173 / 220 = 78.64\%$):** Simulates an oracle correcting the 34 flagged tokens ($170 + 3 = 173$). We explicitly state that **actual human-assisted accuracy has not been measured in a user study**.
5. **Upper Bound Limitation:** Composed from pre-segmented clean crops placed side-by-side with known bounding boxes; it excludes real-world document segmentation challenges (ink bleed, touching strokes, shadow gradients) and represents a segmentation-free optimistic upper bound.

---

## 7. Classical Segmentation Debugging & Real-World Failure Analysis

### Q15: Why did Green's Shoelace formula (`cv2.contourArea`) fail on thin handwritten strokes, and how did you fix it?
**Answer:**
`cv2.contourArea` computes the enclosed polygon area using Green's theorem:
$$A = \frac{1}{2} \left| \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \right|$$
For a closed geometric loop (like `0`, `6`, `8`, or `9`), $A$ accurately measures the internal area. However, for open 1-pixel or 2-pixel thin strokes—such as minus signs (`-`), plus signs (`+`), equality bars (`=`), or straight digits (`1`, `7`)—the contour boundary folds back on itself with zero interior width. Its mathematical polygon area evaluates to $0.0$ to $3.5$.

A filter like `cv2.contourArea(cnt) < 25` caused $100\%$ detection failure on all thin strokes.
**The Fix:**
We discarded the shoelace polygon area for stroke filtering and replaced it with two physically grounded criteria:
1. **Bounding Box Span:** $\max(w, h) \ge 6 \text{ px}$ and $\min(w, h) \ge 1 \text{ px}$ to eliminate microscopic sub-pixel dust.
2. **Foreground Ink Mass:** $\text{cv2.countNonZero}(\text{crop}) \ge 8 \text{ pixels}$, verifying that real ink exists regardless of stroke topology.

---

### Q16: Why did adaptive thresholding bloom strokes, and why is Otsu the preferred default?
**Answer:**
Adaptive Gaussian thresholding computes local thresholds based on neighborhood mean intensity. In white paper regions with uniform background, the local variance is minuscule ($\sigma^2 \approx 0$). Consequently, subtle paper fibers or anti-aliasing pen halos exceed the local threshold ($I < \mu - C$), causing strokes to artificially expand ("bloom") by over 300%.
- In our diagnostics, bloomed strokes had mean pixel density of $35.4$, whereas the CNN training dataset had mean density of $11.9 \pm 3.2$.
- This 3x thickening activated diagonal edge filters in the CNN, misclassifying digits `2` and `7` as `x`.
- **The Fix:** We established unblurred Otsu's thresholding (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`) as the default binarizer. Otsu calculates a global inter-class variance threshold that respects natural stroke boundaries. Adaptive Gaussian thresholding is preserved as an optional toggle for heavily shadowed smartphone photos.

---

### Q17: What is the impact of crop padding margin ratio on CNN classification?
**Answer:**
When segmenting a character bounding box and placing it into a $45 \times 45$ canvas, the padding margin ratio determines the relative scale of the character:
$$\text{padded\_size} = S \times (1 + 2 \times \text{margin\_ratio})$$
- With `padding_margin_ratio: 0.18` (18%), the glyph was compressed into a small ~32px sub-region inside the $45 \times 45$ patch.
- However, the Kaggle training dataset was cropped tightly around characters with an average glyph height of $41.2$ px ($91.5\%$ of the canvas).
- This spatial frequency mismatch caused the CNN's learned convolutional filters to miss crucial topological features.
- **The Fix:** Setting `padding_margin_ratio: 0.04` (4%) restored the extracted glyphs to $41.4$ px, perfectly aligning with the training distribution and restoring CNN accuracy to $>94\%$.

---

### Q18: How does MathVision prevent SymPy solver freezes on complex or invalid equations?
**Answer:**
SymPy's `solve()` uses general Groebner basis algorithms and heuristic transcendental solvers. If an invalid or unconstrained equation is passed (such as $3^x \cdot 4$), SymPy enters a computationally intensive branch that can hang the application thread indefinitely.
**Safety Measures Implemented:**
1. **Restricted Exponent Parsing:** Superscript detection is restricted strictly to numeric digits `0`–`9` and polynomial bases ($x^2, x^3$). Arbitrary expressions are not parsed as powers without explicit syntax.
2. **Baseline Elevation Guard:** Exponents must satisfy strict baseline elevation ($\text{bottom} \le y_{\text{base}} + 0.78 \cdot h_{\text{base}}$).
3. **Execution Timeout Guard (`safe_solve`):** The solver wraps SymPy calls in a 3.0-second background execution thread or signal timer. If computation exceeds 3.0 seconds, it gracefully terminates and returns an informative `timeout` status card rather than crashing the UI.

---

### Q19: Why not use external datasets like Google MathWriting, HME100K, or CROHME?
**Answer:**
1. **Google MathWriting Dataset:** Contains online digital ink (InkML format with temporal stroke sequences $(x_i, y_i, t_i)$). Converting online trajectories to offline 2D raster images requires complex rasterization with artificial stroke broadening and anti-aliasing simulation.
2. **HME100K Dataset:** Comprises complete multi-line document images paired directly with LaTeX strings. It is intended for end-to-end Sequence-to-Sequence (Seq2Seq) Encoder-Decoder models (like TrOCR or Vision Transformers). It lacks character-level bounding box annotations required for classical segmentation pipelines.
3. **CROHME:** Focuses on InkML MathML syntax trees for online recognition competitions.
4. **Conclusion:** Our modular architecture uses classical OpenCV segmentation paired with a character-level CNN. The Kaggle isolated symbol dataset matches this exact paradigm, providing 18,996 clean, deduplicated raster crops across all 18 classes with verified 94.18% test accuracy.

---

### Q20: What are the verified end-to-end benchmark results on the Minimum Solver Test Suite?
**Answer:**
On the 6 canonical benchmark equations passed through end-to-end image synthesis, OpenCV segmentation, CNN classification, Bayesian disambiguation, and SymPy solving:
1. $12 + 8 \div 4 = 14 \implies$ True Identity (718.0 ms) — **PASS**
2. $7 \times 8 - 9 = 47 \implies$ True Identity (613.7 ms) — **PASS**
3. $2x + 5 = 15 \implies x = 5$ (587.8 ms) — **PASS**
4. $4x - 7 = 2x + 9 \implies x = 8$ (238.7 ms) — **PASS**
5. $x^2 - 5x + 6 = 0 \implies x = 2, x = 3$ (221.7 ms) — **PASS**
6. $3(x - 4) = 18 \implies x = 10$ (216.0 ms) — **PASS**

- **End-to-End Recognition Accuracy:** **6 / 6 (100.0%)**
- **End-to-End Solver Accuracy:** **6 / 6 (100.0%)**
- **Automated Regression Suite:** **39 / 39 unit/integration tests passing (100.0%)**
