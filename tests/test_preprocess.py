"""
=============================================================================
Unit Tests for MathVision Preprocessing Pipeline
=============================================================================
"""

import numpy as np
import pytest
from src.preprocess import (
    resize_preserve_aspect,
    to_grayscale,
    apply_gaussian_blur,
    apply_adaptive_threshold,
    compare_thresholds,
    apply_morphology,
    preprocess_pipeline,
    PREPROCESS_CONFIG,
)


def test_resize_preserve_aspect_downscales():
    """Verify aspect ratio is preserved when downscaling large images."""
    img = np.zeros((1200, 2000, 3), dtype=np.uint8)
    resized = resize_preserve_aspect(img, max_width=1000)

    assert resized.shape[1] == 1000
    expected_height = int(round(1000 * (1200 / 2000)))
    assert resized.shape[0] == expected_height


def test_resize_preserve_aspect_no_upscale():
    """Verify images smaller than max_width are not unnecessarily rescaled."""
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    resized = resize_preserve_aspect(img, max_width=800)

    assert resized.shape == (200, 400, 3)


def test_to_grayscale():
    """Verify BGR to single-channel 2D conversion."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    gray = to_grayscale(img)

    assert gray.ndim == 2
    assert gray.shape == (100, 100)


def test_apply_adaptive_threshold_binary_values():
    """Verify adaptive threshold produces strictly binary 0 and 255 values."""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    blurred = apply_gaussian_blur(img)
    thresh = apply_adaptive_threshold(blurred)

    unique_vals = set(np.unique(thresh))
    assert unique_vals.issubset({0, 255})


def test_compare_thresholds_returns_all_keys():
    """Verify comparison function outputs global, otsu, and adaptive maps."""
    img = np.random.randint(50, 200, (80, 80), dtype=np.uint8)
    comp = compare_thresholds(img)

    assert "global" in comp
    assert "otsu" in comp
    assert "adaptive" in comp
    assert comp["global"].shape == (80, 80)
    assert comp["otsu"].shape == (80, 80)
    assert comp["adaptive"].shape == (80, 80)


def test_preprocess_pipeline_runs_on_synthetic_array():
    """Verify the full pipeline dict output contains all intermediate keys."""
    # Create white canvas with black simulated stroke
    img = np.ones((200, 500, 3), dtype=np.uint8) * 255
    img[90:110, 100:400] = 0 # Horizontal black bar (ink)

    results = preprocess_pipeline(img)

    required_keys = [
        "original", "resized", "paper_detected", "gray", "blurred",
        "threshold_comparison", "binary", "cleaned", "deskewed"
    ]
    for key in required_keys:
        assert key in results
    assert results["binary"].ndim == 2


def test_flatten_illumination_removes_shadow_gradient():
    """Verify that morphological background division flattens lighting gradients."""
    from src.preprocess import flatten_illumination
    # Create an image with an artificial heavy gradient from left (dark) to right (bright)
    x = np.linspace(80, 220, 300, dtype=np.float32)
    grad = np.tile(x, (150, 1)).astype(np.uint8)

    # Draw a simulated stroke across the gradient
    grad[60:90, 50:70] = 30  # Dark stroke in darker region
    grad[60:90, 230:250] = 30 # Dark stroke in brighter region

    norm = flatten_illumination(grad)
    # Background across both regions should now be normalized close to 255
    assert norm[20, 20] > 230
    assert norm[20, 280] > 230
    # Both strokes must remain dark foreground
    assert norm[75, 60] < 100
    assert norm[75, 240] < 100


def test_preprocess_real_test_case_preserves_strokes():
    """Verify that 01_original.png preserves handwritten strokes without dropouts."""
    import os, cv2
    img_path = "outputs/real_test_case/01_original.png"
    if os.path.exists(img_path):
        res = preprocess_pipeline(img_path)
        # Binarized strokes must be present
        fg_count = cv2.countNonZero(res["binary"])
        assert fg_count > 1000, f"Expected > 1000 foreground stroke pixels, got {fg_count}"
        assert res["original"].shape[:2] == (283, 504)
        assert res["binary"].shape[1] == 504
        assert res["binary"].shape[0] >= 100


def test_suppress_horizontal_lines_removes_notebook_lines_and_preserves_operators():
    """
    CRITICAL REGRESSION TEST:
    Verify that suppress_horizontal_lines removes long background scanner/notebook lines
    while strictly preserving short mathematical operators: minus (-), plus (+),
    division (÷), and equals (=).
    """
    import cv2
    from src.preprocess import suppress_horizontal_lines

    canvas = np.zeros((150, 400), dtype=np.uint8)

    # 1. Background notebook separator line (length = 360 px across 400 px width)
    canvas[120:123, 20:380] = 255

    # 2. Minus operator '-' (length = 24 px)
    canvas[50:53, 50:74] = 255

    # 3. Plus operator '+' (horizontal bar length = 26 px, vertical bar height = 24 px)
    canvas[40:64, 120:123] = 255
    canvas[51:54, 108:134] = 255

    # 4. Division operator '÷' (bar length = 24 px, 2 dots)
    canvas[51:54, 180:204] = 255
    canvas[36:40, 190:194] = 255
    canvas[64:68, 190:194] = 255

    # 5. Equals operator '=' (two bars length = 28 px)
    canvas[48:51, 250:278] = 255
    canvas[58:61, 250:278] = 255

    cleaned, line_mask, stats = suppress_horizontal_lines(canvas)

    # Assert long notebook line was identified in line_mask and removed from cleaned
    assert stats["lines_detected"] is True
    assert stats["pixels_removed"] > 200
    assert np.count_nonzero(cleaned[120:123, 50:350]) == 0, "Notebook line must be eliminated"

    # Assert operators are strictly preserved
    # Minus:
    assert np.count_nonzero(cleaned[50:53, 50:74]) == np.count_nonzero(canvas[50:53, 50:74]), "Minus operator must be preserved"
    # Plus:
    assert np.count_nonzero(cleaned[40:64, 120:123]) > 0, "Plus vertical stem must be preserved"
    assert np.count_nonzero(cleaned[51:54, 108:134]) > 0, "Plus horizontal cross must be preserved"
    # Division:
    assert np.count_nonzero(cleaned[51:54, 180:204]) > 0, "Division bar must be preserved"
    assert np.count_nonzero(cleaned[36:40, 190:194]) > 0, "Division top dot must be preserved"
    assert np.count_nonzero(cleaned[64:68, 190:194]) > 0, "Division bot dot must be preserved"
    # Equals:
    assert np.count_nonzero(cleaned[48:51, 250:278]) > 0, "Equals top bar must be preserved"
    assert np.count_nonzero(cleaned[58:61, 250:278]) > 0, "Equals bottom bar must be preserved"


def test_detect_writing_region_bounds_excludes_distant_page_boundary():
    """
    CRITICAL REGRESSION TEST:
    Verify that detect_writing_region_bounds identifies the primary equation text line
    and excludes distant page/notebook boundaries below a large empty gap.
    """
    from src.preprocess import detect_writing_region_bounds

    canvas = np.zeros((200, 300), dtype=np.uint8)

    # Simulated equation strokes at y in [30, 65]
    canvas[30:65, 30:50] = 255
    canvas[30:65, 70:90] = 255
    canvas[30:65, 110:130] = 255

    # Distant page boundary line remnant at y in [160, 163] (separated by 95px gap)
    canvas[160:163, 10:80] = 255

    y_top, y_bot = detect_writing_region_bounds(canvas)

    # Assert equation rows are completely retained
    assert y_top <= 30
    assert y_bot >= 65
    # Assert distant boundary line is excluded
    assert y_bot < 160, f"Page boundary at y=160 must be excluded, got y_bot={y_bot}"



