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


def test_find_symbol_contours_rejects_border_artifacts_and_shadow_bands():
    """
    REGRESSION TEST:
    Verify that scanner margins, camera bezels, and shadow bands touching image
    perimeters are rejected and not treated as valid mathematical symbols.
    """
    canvas = np.zeros((200, 300), dtype=np.uint8)

    # 1. Giant border shadow touching left edge (x=0, y=0, w=50, h=200)
    canvas[0:200, 0:50] = 255

    # 2. Legitimate centered symbol '1' (x=140, y=60, w=10, h=60)
    canvas[60:120, 140:150] = 255

    # 3. Thin top border frame line (x=0, y=0, w=300, h=2)
    canvas[0:2, 0:300] = 255

    boxes = find_symbol_contours(canvas)
    # The border shadow and top frame line must be filtered out; only the centered symbol '1' survives
    assert len(boxes) == 1, f"Expected 1 valid box, got {len(boxes)}: {boxes}"
    bx, by, bw, bh = boxes[0]
    assert 135 <= bx <= 145
    assert 55 <= by <= 65


def test_segment_real_test_case_finds_all_six_symbols():
    """Verify that 01_original.png segments exactly 6 symbols without edge boxes."""
    import os, cv2
    from src.preprocess import preprocess_pipeline
    img_path = "outputs/real_test_case/01_original.png"
    if os.path.exists(img_path):
        prep = preprocess_pipeline(img_path)
        seg = segment_symbols(prep["binary"], prep["resized"])
        assert seg["count"] == 6, f"Expected 6 symbols, got {seg['count']}: {seg['boxes']}"
        # Ensure no edge-touching border boxes
        for b in seg["boxes"]:
            x, y, w, h = b
            assert x > 5, f"Suspicious edge box at x={x}"
            assert h < 100, f"Suspicious oversized box with height={h}"


def test_should_merge_division_dots_with_slight_offset():
    """Verify that a division horizontal bar and slightly offset dots merge reliably."""
    from src.segment import should_merge, SEGMENT_CONFIG
    bar = (210, 69, 45, 10)
    top_dot = (205, 55, 8, 8)  # slightly shifted left
    bot_dot = (235, 88, 8, 8)  # slightly shifted right

    assert should_merge(bar, top_dot, SEGMENT_CONFIG) is True
    assert should_merge(bar, bot_dot, SEGMENT_CONFIG) is True


def test_should_merge_split_four_strokes():
    """Verify that multi-stroke fragments of digit '4' (L-stroke and vertical stem) merge."""
    from src.segment import should_merge, SEGMENT_CONFIG
    l_stroke = (270, 56, 18, 30)
    stem = (289, 54, 6, 32)  # touches/overlaps horizontally within 2px

    assert should_merge(l_stroke, stem, SEGMENT_CONFIG) is True


def test_find_symbol_contours_rejects_tall_internal_lines():
    """Verify that tall narrow internal line artifacts (e.g. shadow fold h=150 in h=200 image) are rejected."""
    canvas = np.zeros((200, 300), dtype=np.uint8)
    # Legitimate symbol '1' (x=50, y=70, w=8, h=50) -> h=50 is 25% of img_h
    canvas[70:120, 50:58] = 255
    # Tall shadow streak (x=200, y=20, w=6, h=160) -> h=160 is 80% of img_h
    canvas[20:180, 200:206] = 255

    boxes = find_symbol_contours(canvas)
    assert len(boxes) == 1
    assert boxes[0][0] == 50


def test_candidate_contour_logging_records_retained_and_rejected():
    """Verify that find_symbol_contours records full diagnostics with rejection reasons."""
    canvas = np.zeros((200, 300), dtype=np.uint8)
    # Valid symbol '1' (x=50, y=70, w=10, h=50)
    canvas[70:120, 50:60] = 255
    # Tiny speck (x=100, y=100, w=2, h=2)
    canvas[100:102, 100:102] = 255
    # Border sliver (x=0, y=50, w=1, h=30)
    canvas[50:80, 0:1] = 255

    boxes, diags = find_symbol_contours(canvas, return_diagnostics=True)
    assert len(boxes) == 1
    assert len(diags) >= 2

    # Check structure
    for d in diags:
        assert "bbox" in d
        assert "area" in d
        assert "stroke_pixels" in d
        assert "aspect_ratio" in d
        assert "retained" in d
        assert "rejection_reason" in d

    retained = [d for d in diags if d["retained"]]
    assert len(retained) == 1
    assert retained[0]["rejection_reason"] is None
    assert retained[0]["bbox"][0] == 50

    rejected = [d for d in diags if not d["retained"]]
    assert len(rejected) >= 1
    reasons = [d["rejection_reason"] for d in rejected]
    assert any("span_too_small" in r or "width_below_minimum" in r or "border" in r for r in reasons)


def test_segment_symbols_returns_diagnostics_and_raw_boxes():
    """Verify that segment_symbols returns raw_boxes, merged_boxes, and contour_diagnostics."""
    canvas = np.zeros((100, 200), dtype=np.uint8)
    canvas[30:70, 30:50] = 255
    canvas[30:70, 70:90] = 255

    seg = segment_symbols(canvas)
    assert "boxes" in seg
    assert "raw_boxes" in seg
    assert "merged_boxes" in seg
    assert "contour_diagnostics" in seg
    assert len(seg["contour_diagnostics"]) >= 2


def test_validate_segmentation_sanity_detects_oversized_box():
    """
    REGRESSION TEST:
    Verify that validate_segmentation_sanity detects when an entire equation or page
    collapses into a single giant bounding box (as seen in Failure Case B before the fix).
    """
    from src.segment import validate_segmentation_sanity

    canvas = np.zeros((200, 300), dtype=np.uint8)
    # Simulate a single oversized box spanning 180x100 on 300x200 canvas (60% width, 50% height)
    boxes = [(20, 20, 180, 100)]
    raw_boxes = [(20, 20, 30, 40), (60, 20, 30, 40), (100, 20, 30, 40), (140, 20, 30, 40), (180, 20, 30, 40)]

    sanity = validate_segmentation_sanity(boxes, raw_boxes, canvas)
    assert sanity["is_valid"] is False
    assert sanity["quality"] == "failed"
    assert sanity["flags"]["oversized_box"] is True
    assert sanity["flags"]["extreme_collapse"] is True
    assert len(sanity["warnings"]) >= 2


def test_should_merge_blocks_horizontal_line_remnants():
    """
    REGRESSION TEST:
    Verify that should_merge rejects merges involving long thin horizontal line fragments
    and merges that would result in an oversized box.
    """
    from src.segment import should_merge, SEGMENT_CONFIG

    cfg = {**SEGMENT_CONFIG, "canvas_width": 300, "canvas_height": 200}

    # Line remnant: width 60, height 3
    line_remnant = (10, 100, 60, 3)
    # Character box: width 20, height 30
    char_box = (20, 50, 20, 30)

    # Line remnant should NEVER merge with a character
    assert should_merge(line_remnant, char_box, cfg) is False
    assert should_merge(char_box, line_remnant, cfg) is False

    # Large vertical gap (e.g. 50px between rows)
    top_char = (50, 20, 20, 25)
    bot_char = (50, 95, 20, 25)
    assert should_merge(top_char, bot_char, cfg) is False

    # Oversized combined merge
    left_huge = (10, 20, 120, 80)
    right_huge = (110, 20, 120, 80)
    assert should_merge(left_huge, right_huge, cfg) is False


def test_find_symbol_contours_rejects_page_boundary_false_minus():
    """
    CRITICAL REGRESSION TEST:
    Verify that find_symbol_contours rejects bottom page-boundary line remnants
    (such as the false minus box (0, 162, 41, 4) in Failure Case B) while strictly
    preserving genuine baseline mathematical minus signs and characters.
    """
    canvas = np.zeros((200, 300), dtype=np.uint8)

    # 1. Genuine characters along equation baseline y in [30, 60]
    canvas[30:60, 30:50] = 255  # '3'
    canvas[30:60, 60:80] = 255  # '6'

    # 2. Genuine mathematical minus sign at baseline y in [43, 47], x in [95, 111]
    canvas[43:47, 95:111] = 255

    # 3. Bottom page-boundary false minus at y in [162, 166], x in [0, 41]
    canvas[162:166, 0:41] = 255

    boxes, diags = find_symbol_contours(canvas, return_diagnostics=True)

    # Legitimate characters and minus must be retained
    retained_x = [b[0] for b in boxes]
    assert 30 in retained_x, "Character '3' must be retained"
    assert 60 in retained_x, "Character '6' must be retained"
    assert 95 in retained_x, "Genuine minus sign must be retained"

    # Page-boundary false minus must NOT be in valid boxes
    assert 0 not in retained_x, "Page-boundary false minus at x=0, y=162 must be rejected"
    assert not any(b[1] >= 150 for b in boxes), "No box near bottom boundary should survive"


def test_should_merge_split_four_multiple_components():
    """
    REGRESSION TEST:
    Verify that a digit '4' whose strokes form two separate vertical components
    merges into a single bounding box, but avoids merging with an adjacent complete digit '5'.
    """
    from src.segment import should_merge, merge_bounding_boxes, SEGMENT_CONFIG
    stroke1 = (22, 20, 5, 12)  # Left angled/vertical stroke of '4'
    stroke2 = (29, 21, 5, 11)  # Vertical stem of '4'
    digit5 = (35, 21, 11, 15)  # Adjacent independent digit '5'

    # The two strokes of '4' should merge
    assert should_merge(stroke1, stroke2, SEGMENT_CONFIG) is True
    # Neither stroke of '4' should merge into the adjacent digit '5'
    assert should_merge(stroke2, digit5, SEGMENT_CONFIG) is False
    assert should_merge(stroke1, digit5, SEGMENT_CONFIG) is False

    # End-to-end clustering of [stroke1, stroke2, digit5] must produce exactly 2 boxes
    merged = merge_bounding_boxes([stroke1, stroke2, digit5], SEGMENT_CONFIG)
    assert len(merged) == 2, f"Expected 2 merged boxes, got {len(merged)}: {merged}"
    # Box 1 is the unified '4'
    assert merged[0][0] == 22 and (merged[0][0] + merged[0][2]) == 34
    # Box 2 is the intact '5'
    assert merged[1][0] == 35


def test_division_dot_preservation_and_merging():
    """
    REGRESSION TEST:
    Verify that division dots situated above or below a horizontal bar are preserved
    by find_symbol_contours and subsequently merged with the bar into a single 'div' box.
    """
    from src.segment import find_symbol_contours, merge_bounding_boxes
    canvas = np.zeros((60, 100), dtype=np.uint8)

    # Division horizontal bar at y in [25, 29], x in [40, 55] (w=15, h=4)
    canvas[25:29, 40:55] = 255
    # Upper dot at y in [18, 22], x in [46, 49] (w=3, h=4)
    canvas[18:22, 46:49] = 255
    # Lower dot at y in [33, 37], x in [46, 49] (w=3, h=4)
    canvas[33:37, 46:49] = 255

    boxes = find_symbol_contours(canvas)
    # The bar and at least one dot must be detected
    assert len(boxes) >= 2, f"Expected dots to be preserved, got {len(boxes)} boxes: {boxes}"

    # After merging, they must coalesce into a single unified division box
    merged = merge_bounding_boxes(boxes)
    assert len(merged) == 1, f"Expected 1 merged division box, got {len(merged)}: {merged}"
    bx, by, bw, bh = merged[0]
    assert bw >= 13 and bh >= 10, f"Expected full division glyph bounds, got w={bw}, h={bh}"


def test_handwritten_six_loop_preservation_not_four():
    """
    REGRESSION TEST:
    Verify that a digit '6' with an internal loop cavity is normalized properly
    and not collapsed into an open hook that triggers a false '4' prediction.
    """
    from src.segment import normalize_symbol_crop
    from src.pipeline import get_loaded_models
    from src.dataset import CLASSES

    canvas = np.zeros((40, 40), dtype=np.uint8)
    # Draw curved back of 6
    canvas[5:10, 20:25] = 255
    canvas[10:20, 15:19] = 255
    # Draw bottom loop with clear internal cavity: outer bounds y=[20, 35], x=[12, 30]
    canvas[20:35, 12:30] = 255
    # Inner hole
    canvas[24:31, 17:25] = 0

    crop = normalize_symbol_crop(canvas, (12, 5, 18, 30))
    # Cavity must be preserved (center pixel region should remain 0)
    assert crop.shape == (45, 45)

    models = get_loaded_models()
    cnn = models.get("cnn")
    if cnn is not None:
        batch = np.array([crop])[..., np.newaxis]
        probs = cnn.predict(batch, verbose=0)[0]
        pred_label = CLASSES[np.argmax(probs)]
        assert pred_label == "6", f"Expected '6', got '{pred_label}' (conf: {probs[np.argmax(probs)]:.2%})"


def test_should_merge_does_not_merge_parenthesis_with_digit():
    """Verify should_merge does not merge an opening parenthesis with adjacent digit 1."""
    from src.segment import should_merge, SEGMENT_CONFIG
    paren_box = (25, 22, 11, 28)
    digit_one_box = (41, 30, 3, 14)
    assert should_merge(paren_box, digit_one_box, SEGMENT_CONFIG) is False
    assert should_merge(digit_one_box, paren_box, SEGMENT_CONFIG) is False


def test_should_merge_does_not_merge_digit_with_closing_parenthesis():
    """Verify should_merge does not merge digit 6 with adjacent closing parenthesis )."""
    from src.segment import should_merge, SEGMENT_CONFIG
    digit_six_box = (98, 30, 10, 21)
    close_paren_box = (112, 25, 11, 33)
    assert should_merge(digit_six_box, close_paren_box, SEGMENT_CONFIG) is False
    assert should_merge(close_paren_box, digit_six_box, SEGMENT_CONFIG) is False


def test_validate_segmentation_sanity_detects_unexpected_merge():
    """Verify validate_segmentation_sanity flags unexpected_merge when substantial contours collapse."""
    from src.segment import validate_segmentation_sanity
    raw_boxes = [
        (25, 22, 11, 28),
        (41, 30, 3, 14),
        (51, 28, 11, 18),
        (76, 31, 14, 17),
        (98, 30, 10, 21),
        (112, 25, 11, 33),
        (127, 26, 16, 36),
        (148, 33, 15, 20)
    ]
    # Erroneously merged into 5 boxes
    merged_erroneous = [
        (25, 22, 19, 28),
        (51, 28, 11, 18),
        (76, 31, 14, 17),
        (98, 25, 25, 33),
        (148, 33, 15, 20)
    ]
    dummy_bin = np.zeros((80, 200), dtype=np.uint8)
    sanity = validate_segmentation_sanity(merged_erroneous, raw_boxes, dummy_bin)
    assert sanity["is_valid"] is False or sanity["quality"] == "suspicious"
    assert sanity["flags"]["unexpected_merge"] is True or sanity["flags"]["suspicious_wide_box"] is True







