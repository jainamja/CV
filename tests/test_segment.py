"""
=============================================================================
Unit Tests for MathVision Segmentation Pipeline
=============================================================================
"""

import numpy as np
import pytest
from src.segment import (
    find_symbol_contours,
    merge_bounding_boxes,
    sort_boxes_left_to_right,
    normalize_symbol_crop,
    segment_symbols,
    SEGMENT_CONFIG,
)


def test_merge_equals_symbol_into_single_box():
    """
    CRITICAL REQUIREMENT:
    Verify that an '=' sign consisting of two parallel horizontal bars
    merges into exactly one unified bounding box.
    """
    canvas = np.zeros((100, 100), dtype=np.uint8)

    # Top horizontal bar of '=': y in [35, 42], x in [20, 80]
    canvas[35:42, 20:80] = 255
    # Bottom horizontal bar of '=': y in [52, 59], x in [20, 80]
    canvas[52:59, 20:80] = 255

    raw_boxes = find_symbol_contours(canvas)
    assert len(raw_boxes) == 2, f"Expected 2 raw stroke contours, got {len(raw_boxes)}"

    merged_boxes = merge_bounding_boxes(raw_boxes)
    assert len(merged_boxes) == 1, f"Expected 1 merged box for '=', got {len(merged_boxes)}"

    x, y, w, h = merged_boxes[0]
    assert x <= 20
    assert (x + w) >= 80
    assert y <= 35
    assert (y + h) >= 59


def test_merge_division_symbol_into_single_box():
    """
    CRITICAL REQUIREMENT:
    Verify that a '÷' sign consisting of a bar and two vertical dots
    merges into exactly one unified bounding box.
    """
    canvas = np.zeros((120, 100), dtype=np.uint8)

    # Top dot: y in [20, 28], x in [45, 55]
    canvas[20:28, 45:55] = 255
    # Center horizontal bar: y in [55, 62], x in [25, 75]
    canvas[55:62, 25:75] = 255
    # Bottom dot: y in [88, 96], x in [45, 55]
    canvas[88:96, 45:55] = 255

    raw_boxes = find_symbol_contours(canvas)
    assert len(raw_boxes) == 3, f"Expected 3 raw stroke contours, got {len(raw_boxes)}"

    merged_boxes = merge_bounding_boxes(raw_boxes)
    assert len(merged_boxes) == 1, f"Expected 1 merged box for '÷', got {len(merged_boxes)}"


def test_sort_boxes_left_to_right():
    """Verify bounding boxes are strictly ordered left-to-right by x-coordinate."""
    boxes = [
        (300, 50, 40, 40),
        (50, 50, 30, 40),
        (150, 50, 35, 40),
    ]
    sorted_boxes = sort_boxes_left_to_right(boxes)
    x_coords = [b[0] for b in sorted_boxes]
    assert x_coords == [50, 150, 300]


def test_normalize_symbol_crop_shape_and_range():
    """Verify crops are padded, centered, and reshaped to 45x45 in [0.0, 1.0]."""
    canvas = np.zeros((100, 100), dtype=np.uint8)
    # Simulated vertical digit '1'
    canvas[20:80, 45:55] = 255

    box = (45, 20, 10, 60)
    crop = normalize_symbol_crop(canvas, box)

    assert crop.shape == (45, 45)
    assert crop.dtype == np.float32
    assert crop.min() >= 0.0
    assert crop.max() <= 1.0
    # Foreground ink must be present
    assert crop.max() > 0.5


def test_segment_symbols_end_to_end():
    """Verify segment_symbols pipeline on a multi-digit equation canvas."""
    canvas = np.zeros((120, 400), dtype=np.uint8)

    # Symbol '1': x in [40, 50]
    canvas[30:90, 40:50] = 255

    # Symbol '+': vertical x in [110, 120], horizontal x in [95, 135]
    canvas[40:80, 110:120] = 255
    canvas[55:65, 95:135] = 255

    # Symbol '2': x in [170, 200]
    canvas[30:90, 170:200] = 255

    res = segment_symbols(canvas)
    assert res["count"] == 3
    assert len(res["boxes"]) == 3
    assert len(res["symbol_crops"]) == 3


def test_find_symbol_contours_detects_thin_handwritten_strokes():
    """
    REGRESSION TEST:
    Handwritten strokes from gel pens, ballpoint pens, or pencil lines are often
    1-2 pixels wide. Their enclosed polygon contour area (cv2.contourArea) is ~0.
    Verify that thin 1-pixel strokes (such as minus signs or 1s) are detected
    and not discarded by the noise filter.
    """
    canvas = np.zeros((100, 100), dtype=np.uint8)
    # Thin 1-pixel tall horizontal minus sign '-' of width 25
    canvas[50:51, 30:55] = 255
    # Thin 1-pixel wide vertical line '1' of height 35
    canvas[20:55, 75:76] = 255

    boxes = find_symbol_contours(canvas)
    assert len(boxes) == 2, f"Expected 2 thin stroke boxes, got {len(boxes)}: {boxes}"


def test_find_symbol_contours_rejects_single_pixel_noise():
    """Verify that 1x1 and 2x2 dust specks are rejected as noise."""
    canvas = np.zeros((100, 100), dtype=np.uint8)
    # Tiny 1x1 and 2x2 noise specks
    canvas[10, 10] = 255
    canvas[20:22, 20:22] = 255

    boxes = find_symbol_contours(canvas)
    assert len(boxes) == 0, f"Expected 0 noise boxes, got {len(boxes)}"
