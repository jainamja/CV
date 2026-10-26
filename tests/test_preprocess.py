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
