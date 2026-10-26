# Real Handwritten Equation Test Set (`data/samples_real/`)

This directory is designated for **real physical photos** of your personal handwriting on paper.

## Instructions for Capturing Test Photos
1. **Paper & Ink**: Use plain white paper (unlined or lightly lined) with dark blue or black pen/marker.
2. **Format**: Single-line mathematical equations or arithmetic expressions.
3. **Capture Conditions**:
   - Take 15–20 photos using your smartphone or webcam.
   - Vary the angles slightly (e.g., $0^\circ$ to $15^\circ$ tilt) and lighting conditions (natural daylight, indoor lamp, mild shadow).
   - Ensure the entire paper sheet or expression is visible within the frame.
4. **Filename & Labeling**:
   - Save your photos as `real_01.png`, `real_02.png`, etc., or `.jpg`.
   - Add a corresponding row to `data/samples_real/labels.csv` with the ground-truth expression.

## Example `labels.csv` Template:
```csv
filename,expression,canonical_expression,category
real_01.jpg,2*x + 5 = 15,2*x + 5 = 15,linear
real_02.jpg,3*(x - 4) = 18,3*(x - 4) = 18,linear
real_03.jpg,x^2 - 5*x + 6 = 0,x**2 - 5*x + 6 = 0,quadratic
real_04.jpg,(12 + 8) / 4 = 5,(12 + 8) / 4 = 5,arithmetic
real_05.jpg,7 * 8 - 9 = 47,7 * 8 - 9 = 47,arithmetic
```

## How the Evaluation Suite Uses This:
`src/evaluate.py` benchmarks `data/samples_real/` independently from synthetic samples. For every test image, it reports:
- Symbol-level recognition accuracy.
- Equation-level exact-match accuracy.
- Stage-isolated failure diagnosis:
  - **Segmentation Failure**: Multi-part symbols split or overlapping glyphs merged improperly.
  - **Classification Failure**: Glyph contour correctly isolated, but misclassified by the CNN.
  - **Parsing Failure**: Glyphs classified correctly, but syntax error or malformed expression generated.
