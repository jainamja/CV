# MathVision — Live Viva Demonstration Walkthrough

**Target Audience:** College Viva Examiners, Project Reviewers, and Evaluators  
**Application URL:** `http://localhost:8501`  
**Execution Command:** `python -m streamlit run app.py`

---

## 1. Pre-Flight Demonstration Checklist

Before calling the examiner to your screen:
1. Open PowerShell in `C:\Users\Jainam\Downloads\CV Project`.
2. Verify Python environment:
   ```powershell
   python -c "import streamlit, cv2, tensorflow, sympy; print('All core libraries ready!')"
   ```
3. Run the automated test suite to verify 100% integrity:
   ```powershell
   python -m pytest tests/
   ```
   *(Expected: `39 passed in ~22s`)*
4. Launch the Streamlit application:
   ```powershell
   python -m streamlit run app.py
   ```
5. Open your browser at `http://localhost:8501`. Confirm the clean light theme loads with title **MathVision**.

---

## 2. 8-Minute Live Viva Presentation Script

### Step 1: The Elevator Pitch (60 Seconds)
> *"Good morning, Professors. MathVision is an end-to-end Computer Vision and Symbolic Artificial Intelligence system designed to bridge physical handwriting with algorithmic mathematical derivation.*  
> *Unlike conventional solutions that wrap proprietary black-box cloud OCR, MathVision implements character localization, contour filtering, and spatial segmentation completely from first principles using classical OpenCV. It classifies glyphs using a custom 3-block Deep CNN, resolves mathematical symbol ambiguities using Bayesian contextual grammar rescoring, and computes pedagogical step-by-step solutions using SymPy."*

---

### Step 2: Live Equation Demonstration (2 Minutes)
1. In the sidebar or top input selector, select **"Preloaded Sample Images"**.
2. Select **Sample 1: `2x + 5 = 15`** (or upload `scratch/phase5_eq_3.png`).
3. Point out how quickly the system processes the image (**< 600 ms**).
4. Direct the examiner's attention to the main view:
   - **Recognized Expression:** Displayed as formatted LaTeX: $2x + 5 = 15$.
   - **Step-by-Step Derivation:**
     - Step 1: Subtract 5 from both sides: $2x = 10$.
     - Step 2: Divide both sides by 2: $x = 5$.
     - Final Answer: **$x = 5$** highlighted in a green success card.

---

### Step 3: Explain the Classical CV Pipeline (2 Minutes)
Click on **Tab 2: "Pipeline Steps"**:
> *"Here is our transparent 6-stage classical computer vision pipeline:*
> 1. ***Original & Aspect-Preserving Resize:** Downscaled to standard pixel dimensions to guarantee uniform morphological kernel behavior.*
> 2. ***Binary Thresholding:** We use unblurred Otsu's thresholding (`cv2.THRESH_OTSU`) by default to eliminate lighting shadows without stroke blooming, with an adaptive Gaussian fallback for complex illumination.*
> 3. ***Morphological Filtering:** $2\times 2$ elliptical opening eliminates isolated noise specks; $2\times 2$ closing bridges micro-discontinuities in ballpoint pen strokes.*
> 4. ***Suzuki-Abe Contour Extraction:** `cv2.findContours(..., cv2.RETR_EXTERNAL)` isolates individual connected components.*
> 5. ***Physical Noise Filtering:** Instead of polygon shoelace area (`cv2.contourArea`) which fails on thin 1-pixel strokes, we enforce physical bounding box dimensions ($\max(w, h) \ge 6$) and ink pixel count ($\ge 8$).*
> 6. ***Multi-Stroke Merging:** Vertical proximity and horizontal overlap clustering merge bipartite symbols like '=' and '÷' into single semantic entities."*

---

### Step 4: Explain Neural Classification & Bayesian Rescoring (2 Minutes)
Click on **Tab 3: "Detected Symbols"**:
> *"Each character is extracted with an aspect-preserving square padding (4% margin) to match our Kaggle training dataset of 18,996 images:*
> - *Notice the confidence scores underneath each glyph (e.g. 99.8% on '2', 98.4% on 'x').*
> - *Notice how the system distinguishes variable $x$ from multiplication $\times$:*
>   *At the isolated stroke level, two crossed lines are geometrically identical. MathVision uses a **Bayesian soft-rescoring log-posterior**:*
>   $$\log P(c \mid \text{obs}) = \log P_{\text{CNN}}(c) + 0.4 \log P_{\text{stroke}}(c) + 1.5 \log P_{\text{grammar}}(c)$$
>   *Between two numbers ($3 \times 4$), grammar favors multiplication; adjacent to coefficients ($2x$), grammar favors variable $x$. This reduced cross-classification errors by **-90.7%**."*

---

### Step 5: Live Real-Time Telemetry & Diagnostics (1 Minute)
Click the expander **"🔍 Real-Time Pipeline Diagnostics"**:
> *"For full engineering transparency, our telemetry expander exposes:*
> - *Exact bounding box coordinates and aspect ratios $(w/h)$.*
> - *Non-zero ink pixel counts per segmented symbol.*
> - *Per-symbol CNN top-1 confidence scores.*
> - *Sub-millisecond execution timing breakdown: Preprocessing (2.4ms), Segmentation (3.8ms), Inference (18.1ms), Parsing (1.2ms), Solver (12.4ms)."*

---

### Step 6: Interactive Canvas or Camera Demo (1 Minute)
1. Select **"Interactive Canvas"** from the sidebar.
2. Use your mouse to draw an equation, for example: `7 * 8 - 9 = 47`.
3. Click **"Solve Equation"**.
4. Show how the system handles live handwriting and proves the identity: LHS = 47, RHS = 47.
5. Click **"Download Solution ZIP"** at the bottom to demonstrate that all intermediate stages and markdown reports can be exported for educational records.

---

## 3. Examiner Q&A Quick Reference

| Potential Examiner Question | Crisp 15-Second Answer |
|---|---|
| *"Why not use Tesseract or EasyOCR?"* | Tesseract is trained for horizontal printed text, not multi-line 2D mathematical syntax or superscripts. Building our own pipeline gives us full algorithmic control over segmentation, stroke merging, and mathematical AST grammar parsing. |
| *"What is your overall test accuracy?"* | The 3-block Deep CNN achieves **94.18% test accuracy** on 2,850 held-out handwritten crops. On our 6 canonical benchmark equations, the end-to-end pipeline achieves **100% recognition and solver accuracy**. |
| *"How do you handle superscripts and powers ($x^2$)?"* | Using spatial baseline elevation: when a bounding box's bottom is higher than $y + 0.78h$ of the preceding base and its height is smaller, our parser automatically injects the power operator `**`. |
| *"Can a user maliciously inject code?"* | No. We never use Python `eval()`. The token vocabulary is strictly constrained to 18 mathematical symbols, validated by an AST syntax guard, and solved in a sandboxed SymPy context with a 3.0-second execution timeout. |

---

## 4. Emergency Live Troubleshooting

- **Issue:** Web camera fails to open.  
  **Action:** Seamlessly switch to "Interactive Canvas" or upload from `scratch/phase5_eq_1.png` to `scratch/phase5_eq_6.png`.
- **Issue:** Hand-drawn characters touch each other on canvas.  
  **Action:** Remind the examiner that classical character segmentation requires standard handwriting character spacing. Clear canvas and write symbols with clear 10px gaps.
- **Issue:** Streamlit server stops.  
  **Action:** Open PowerShell, run `python -m streamlit run app.py --server.port 8501`.
