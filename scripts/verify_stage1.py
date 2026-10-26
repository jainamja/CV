"""
=============================================================================
MathVision: Stage 1 Verification & Intermediate Image Generator
=============================================================================

Runs the complete Stage 1 CV pipeline (preprocess + segment) on sample
equation images, saves all intermediate states into outputs/stage1_intermediates/,
and prints diagnostic metrics.
=============================================================================
"""

import os
import sys
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_pipeline
from src.segment import segment_symbols, draw_bounding_boxes


def run_stage1_verification(
    sample_path: str = "data/samples/sample_01.png",
    output_dir: str = "outputs/stage1_intermediates"
) -> None:
    """Executes Stage 1 pipeline and exports every intermediate artifact."""
    print("=" * 70)
    print(f"[Stage 1 Verification] Processing: {sample_path}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Run Preprocessing Pipeline
    prep = preprocess_pipeline(sample_path)
    print(f" -> Loaded Original Image: {prep['original'].shape}")
    print(f" -> Resized Image Shape: {prep['resized'].shape}")
    print(f" -> Paper quadrilateral detected: {prep['paper_detected']}")

    # Save Preprocessing Intermediate Images
    cv2.imwrite(os.path.join(output_dir, "01_original.png"), prep["original"])
    cv2.imwrite(os.path.join(output_dir, "02_resized.png"), prep["resized"])
    cv2.imwrite(os.path.join(output_dir, "03_grayscale.png"), prep["gray"])
    cv2.imwrite(os.path.join(output_dir, "04_blurred.png"), prep["blurred"])

    # Threshold Comparison Grid (Global vs Otsu vs Adaptive)
    comp = prep["threshold_comparison"]
    cv2.imwrite(os.path.join(output_dir, "05_thresh_global.png"), comp["global"])
    cv2.imwrite(os.path.join(output_dir, "05_thresh_otsu.png"), comp["otsu"])
    cv2.imwrite(os.path.join(output_dir, "05_thresh_adaptive.png"), comp["adaptive"])

    # Create a side-by-side threshold comparison montage
    h, w = comp["adaptive"].shape[:2]
    montage = np.zeros((h, w * 3), dtype=np.uint8)
    montage[:, 0:w] = comp["global"]
    montage[:, w:w * 2] = comp["otsu"]
    montage[:, w * 2:w * 3] = comp["adaptive"]

    # Annotate montage with text headers
    montage_bgr = cv2.cvtColor(montage, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(montage_bgr, "Global (Fixed 127)", (15, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(montage_bgr, "Otsu (Bimodal)", (w + 15, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(montage_bgr, "Adaptive Gaussian (Local)", (w * 2 + 15, 30), font, 0.7, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "05_threshold_comparison_montage.png"), montage_bgr)

    # Save Morphology output
    cv2.imwrite(os.path.join(output_dir, "06_cleaned_morphology.png"), prep["cleaned"])
    cv2.imwrite(os.path.join(output_dir, "07_deskewed.png"), prep["deskewed"])

    # 2. Run Segmentation Pipeline
    seg = segment_symbols(prep["cleaned"], prep["resized"])
    boxes = seg["boxes"]
    crops = seg["symbol_crops"]

    print(f" -> Symbols Segmented: {len(boxes)}")
    for i, (bx, by, bw, bh) in enumerate(boxes):
        print(f"    [Symbol #{i + 1}] Box: (x={bx}, y={by}, w={bw}, h={bh}) | Aspect Ratio w/h={bw / float(bh):.2f}")

    # Save Annotated Bounding Box Visual
    cv2.imwrite(os.path.join(output_dir, "08_segmented_bounding_boxes.png"), seg["boxed_image"])

    # Save Individual 45x45 Character Crops
    crops_dir = os.path.join(output_dir, "09_symbol_crops")
    os.makedirs(crops_dir, exist_ok=True)

    # Also build a single horizontal strip of all 45x45 crops for easy review
    if crops:
        strip = np.hstack([np.uint8(c * 255) for c in crops])
        cv2.imwrite(os.path.join(output_dir, "09_crops_horizontal_strip.png"), strip)

    for idx, crop in enumerate(crops):
        crop_uint8 = np.uint8(crop * 255)
        crop_path = os.path.join(crops_dir, f"crop_{idx + 1:02d}.png")
        cv2.imwrite(crop_path, crop_uint8)

    print(f"\n[Stage 1 Verification] All intermediate artifacts successfully saved to:")
    print(f" -> {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    test_sample = "data/samples/sample_01.png"
    if len(sys.argv) > 1:
        test_sample = sys.argv[1]
    run_stage1_verification(test_sample)
