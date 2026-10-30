"""
=============================================================================
MathVision: Character Segmentation Module (Classical Computer Vision)
=============================================================================

This module isolates, merges, and normalizes individual mathematical symbols
from the preprocessed binary equation image.

Key Challenges in Mathematical Symbol Segmentation:
    1. Multi-component Symbols: Characters like '=', '÷', 'i', 'j', and ':'
       are composed of disconnected strokes that produce separate contours.
       A naive contour detector would isolate each stroke as a distinct character.
       We implement a geometric bounding-box merging algorithm based on horizontal
       overlap and vertical proximity.
    2. Aspect Ratio Distortion: Symbols vary dramatically in shape (e.g., '-' is
       wide and short; '1' and '(' are tall and narrow). Naive resizing to 45x45
       would warp and distort topological features. We pad crops to square bounding
       boxes with an intentional margin before resizing.
    3. Spatial Coordinate Preservation: The vertical positioning (y-min, y-center,
       and height) relative to neighboring symbols is critical for detecting
       superscripts/exponents (e.g. x^2 vs x2) in the syntax parser.
=============================================================================
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import cv2
from skimage.morphology import skeletonize


# =============================================================================
# SEGMENTATION CONFIGURATION
# Tunable parameters to eliminate magic numbers.
# =============================================================================
SEGMENT_CONFIG: Dict[str, Any] = {
    # 1. Noise Rejection
    # Rejects minuscule specks (e.g. paper grain, dust)
    "min_contour_area": 10,         # Absolute minimum pixel area
    "min_height_ratio": 0.02,       # Min height relative to total image height
    "min_width_pixels": 1,          # Allow thin 1-pixel strokes (gel pens / pencil lines)
    "min_stroke_pixels": 8,         # Minimum foreground pixels to reject dust specks

    # 2. Multi-Part Symbol Merging (Vital for '=', '÷')
    # Horizontal overlap ratio: overlap_x / min(w1, w2)
    "merge_min_h_overlap_ratio": 0.35, # At least 35% horizontal alignment
    # Vertical gap allowance
    "merge_max_v_gap_ratio": 2.5,      # Allow vertical gap up to 2.5x the taller box height
    "merge_min_v_gap_pixels": 30,      # Minimum absolute vertical gap allowance (essential for thin '=' lines)

    # 3. Square Padding & Normalization
    "target_size": (45, 45),        # CNN standard input dimension (height, width)
    "padding_margin_ratio": 0.04,   # Margin padding (4%) to match Kaggle character scale (~41px/45px)
    "interpolation_down": cv2.INTER_AREA,
    "interpolation_up": cv2.INTER_CUBIC,

    # 4. Visualization
    "box_color": (0, 220, 130),     # Neon Emerald Green in BGR
    "box_thickness": 2,
    "font_scale": 0.55,
    "text_color": (255, 255, 255),  # White
    "badge_bg_color": (30, 30, 30), # Dark slate badge background
}


def find_symbol_contours(
    binary_img: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    return_diagnostics: bool = False
) -> Union[List[Tuple[int, int, int, int]], Tuple[List[Tuple[int, int, int, int]], List[Dict[str, Any]]]]:
    """
    Extracts raw bounding boxes for all candidate symbols using external contours.

    Viva Explanation:
    cv2.RETR_EXTERNAL retrieves only the outermost parent contours, ignoring holes
    inside characters (e.g., inside '0', '8', '6', '4'). This prevents interior
    counter-loops from generating redundant sub-boxes. cv2.CHAIN_APPROX_SIMPLE
    compresses horizontal, vertical, and diagonal contour segments into end-points,
    conserving computational overhead.

    Args:
        binary_img: Single-channel binary image (ink = 255, background = 0).
        config: Tunable settings dictionary.
        return_diagnostics: If True, returns (valid_boxes, candidate_diagnostics).

    Returns:
        List of bounding boxes, or (boxes, diagnostics) if return_diagnostics=True.
    """
    cfg = config or SEGMENT_CONFIG
    img_h, img_w = binary_img.shape[:2]

    # Find external contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ([], []) if return_diagnostics else []

    min_area = cfg.get("min_contour_area", 10)
    min_stroke_pixels = cfg.get("min_stroke_pixels", 8)
    min_h = max(3, int(img_h * cfg.get("min_height_ratio", 0.02)))
    min_w = cfg.get("min_width_pixels", 1)

    # Pre-extract all raw bounding boxes to identify candidate horizontal operator bars (e.g. for division '÷')
    all_raw_boxes = [cv2.boundingRect(c) for c in contours]
    cand_bars = [
        b for b in all_raw_boxes
        if b[2] >= 8 and (float(b[2]) / max(1, b[3])) >= 1.4 and b[3] <= 14
    ]

    valid_boxes: List[Tuple[int, int, int, int]] = []
    diagnostics: List[Dict[str, Any]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox = (x, y, w, h)
        box_area = int(w * h)
        aspect_ratio = round(float(w) / float(h), 2) if h > 0 else 0.0
        stroke_pixels = int(cv2.countNonZero(binary_img[y:y+h, x:x+w]))

        # Principled division dot preservation:
        # A division dot (which may have max(w, h) in [2, 4] and >= 3 stroke pixels)
        # must be preserved if it is vertically aligned with an adjacent horizontal bar.
        is_candidate_div_dot = False
        if min(w, h) >= 1 and stroke_pixels >= 3 and w <= 8 and h <= 8:
            dot_cx = x + 0.5 * w
            for bar_x, bar_y, bar_w, bar_h in cand_bars:
                bar_cx = bar_x + 0.5 * bar_w
                # Check horizontal alignment: center of dot must align near center of horizontal bar
                if abs(dot_cx - bar_cx) <= max(4, 0.30 * bar_w):
                    # Check vertical proximity
                    v_gap = max(0, max(y, bar_y) - min(y + h, bar_y + bar_h))
                    if v_gap <= max(14, int(1.2 * bar_w)) and v_gap <= 24:
                        if w <= 0.75 * bar_w:
                            is_candidate_div_dot = True
                            break

        # Noise filter: reject minuscule specks that are too tiny in span or foreground pixel count.
        # Note on cv2.contourArea: Open/thin handwritten strokes (1-2 px thick) have degenerate
        # polygon contours whose enclosed shoelace polygon area (cv2.contourArea) is ~0.
        # We reject minuscule noise using bounding box span and actual ink pixel count (cv2.countNonZero).
        if not is_candidate_div_dot:
            if max(w, h) < 5 or min(w, h) < 1:
                if return_diagnostics:
                    diagnostics.append({
                        "bbox": bbox,
                        "area": box_area,
                        "stroke_pixels": stroke_pixels,
                        "aspect_ratio": aspect_ratio,
                        "retained": False,
                        "rejection_reason": "span_too_small"
                    })
                continue

            if stroke_pixels < min_stroke_pixels:
                if return_diagnostics:
                    diagnostics.append({
                        "bbox": bbox,
                        "area": box_area,
                        "stroke_pixels": stroke_pixels,
                        "aspect_ratio": aspect_ratio,
                        "retained": False,
                        "rejection_reason": "insufficient_stroke_pixels"
                    })
                continue

        # Reject long horizontal divider lines/rules across the image (e.g. notebook rules or scanner separators)
        if w > 0.65 * img_w and w > 4 * h:
            if return_diagnostics:
                diagnostics.append({
                    "bbox": bbox,
                    "area": box_area,
                    "stroke_pixels": stroke_pixels,
                    "aspect_ratio": aspect_ratio,
                    "retained": False,
                    "rejection_reason": "horizontal_divider_line"
                })
            continue

        # Reject long thin horizontal background line remnants
        is_border_zone = (x <= 2 or (x + w) >= img_w - 2 or y <= 2 or (y + h) >= img_h - 2)
        if (w >= max(45, int(0.30 * img_w)) and (float(w) / max(1, h)) >= 7.0 and h <= 5) or \
           (is_border_zone and (float(w) / max(1, h)) >= 6.0 and h <= 5 and w >= 35):
            if return_diagnostics:
                diagnostics.append({
                    "bbox": bbox,
                    "area": box_area,
                    "stroke_pixels": stroke_pixels,
                    "aspect_ratio": aspect_ratio,
                    "retained": False,
                    "rejection_reason": "horizontal_line_fragment"
                })
            continue

        # Border artifact filter: reject background noise, frame borders, and shadow bands
        # that touch the image perimeter and span unrealistic dimensions.
        is_left_border = x <= 2
        is_top_border = y <= 2
        is_right_border = (x + w) >= img_w - 2
        is_bottom_border = (y + h) >= img_h - 2
        touches_border = is_left_border or is_top_border or is_right_border or is_bottom_border

        if touches_border:
            # Reject 1-2px perimeter edge slivers from margin cropping / edge cutoff
            if ((is_left_border or is_right_border) and w <= 2) or ((is_top_border or is_bottom_border) and h <= 2):
                if return_diagnostics:
                    diagnostics.append({
                        "bbox": bbox,
                        "area": box_area,
                        "stroke_pixels": stroke_pixels,
                        "aspect_ratio": aspect_ratio,
                        "retained": False,
                        "rejection_reason": "border_sliver_edge"
                    })
                continue

            # Reject if it spans too large of the image height or width (shadow band or border frame)
            if h > 0.65 * img_h or w > 0.70 * img_w:
                if return_diagnostics:
                    diagnostics.append({
                        "bbox": bbox,
                        "area": box_area,
                        "stroke_pixels": stroke_pixels,
                        "aspect_ratio": aspect_ratio,
                        "retained": False,
                        "rejection_reason": "border_span_oversized"
                    })
                continue
            # Reject thin border lines running along the boundary (frames/scanners)
            if (is_left_border or is_right_border) and w <= 5 and h > 0.50 * img_h:
                if return_diagnostics:
                    diagnostics.append({
                        "bbox": bbox,
                        "area": box_area,
                        "stroke_pixels": stroke_pixels,
                        "aspect_ratio": aspect_ratio,
                        "retained": False,
                        "rejection_reason": "border_vertical_streak"
                    })
                continue
            if (is_top_border or is_bottom_border) and h <= 5 and w > 0.50 * img_w:
                if return_diagnostics:
                    diagnostics.append({
                        "bbox": bbox,
                        "area": box_area,
                        "stroke_pixels": stroke_pixels,
                        "aspect_ratio": aspect_ratio,
                        "retained": False,
                        "rejection_reason": "border_horizontal_streak"
                    })
                continue
            # Reject large corner shadow blobs
            if (w * h) > 0.15 * (img_w * img_h):
                if return_diagnostics:
                    diagnostics.append({
                        "bbox": bbox,
                        "area": box_area,
                        "stroke_pixels": stroke_pixels,
                        "aspect_ratio": aspect_ratio,
                        "retained": False,
                        "rejection_reason": "border_corner_blob"
                    })
                continue

        # Reject tall narrow internal line artifacts (shadow folds, margin lines, scanner streaks)
        if h > 0.60 * img_h and w <= 12:
            if return_diagnostics:
                diagnostics.append({
                    "bbox": bbox,
                    "area": box_area,
                    "stroke_pixels": stroke_pixels,
                    "aspect_ratio": aspect_ratio,
                    "retained": False,
                    "rejection_reason": "tall_narrow_line_artifact"
                })
            continue

        if return_diagnostics:
            diagnostics.append({
                "bbox": bbox,
                "area": box_area,
                "stroke_pixels": stroke_pixels,
                "aspect_ratio": aspect_ratio,
                "retained": True,
                "rejection_reason": None
            })
        valid_boxes.append(bbox)

    # Positional baseline check: reject horizontal slivers situated far from the writing baseline
    primary_boxes = [b for b in valid_boxes if b[2] * b[3] >= 50 and b[3] >= 10]
    if len(primary_boxes) >= 2:
        med_y = float(np.median([b[1] + b[3] / 2.0 for b in primary_boxes]))
        med_h = float(np.median([b[3] for b in primary_boxes]))
        filtered_boxes = []
        for b in valid_boxes:
            bx, by, bw, bh = b
            dist = abs(by + bh / 2.0 - med_y)
            if dist > 2.2 * med_h and (float(bw) / max(1, bh)) >= 4.0 and bh <= 6:
                if return_diagnostics:
                    for d in diagnostics:
                        if d["bbox"] == b and d["retained"]:
                            d["retained"] = False
                            d["rejection_reason"] = "vertical_outlier_line_remnant"
                continue
            filtered_boxes.append(b)
        valid_boxes = filtered_boxes

    if return_diagnostics:
        return valid_boxes, diagnostics
    return valid_boxes


def should_merge(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
    config: Dict[str, Any]
) -> bool:
    """
    Evaluates whether two bounding boxes represent fragments of a single symbol.

    Viva Explanation:
    Characters like '=' consist of two parallel horizontal bars.
    Characters like '÷' consist of a central bar and two vertical dots.
    Multi-stroke characters like '4', 'x', and '+' consist of intersecting/adjacent strokes.
    We evaluate:
        1. Division Symbol Alignment: If one box is a horizontal bar (w >= 1.6*h)
           and the other is a compact dot/stroke centered horizontally within the bar's
           span and vertically close (v_gap <= 45px or 1.8*bw), merge them.
        2. Multi-Stroke Character Fragments: If two strokes have significant vertical
           overlap (>= 40% of min height) and touch or have <= 2px hairline separation in x,
           merge them (e.g. fractured digit '4' or disconnected crossbars).
        3. Parallel Stroke Alignment: Standard horizontal overlap ratio >= min_h_overlap
           and vertical distance <= max_gap_allowed.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Guard 0: Never merge horizontal line remnants
    for b in [box1, box2]:
        if b[2] >= 40 and (float(b[2]) / max(1, b[3])) >= 7.0 and b[3] <= 6:
            return False

    # Guard 1: Never merge components if it would create an oversized box spanning the whole writing region
    if "canvas_width" in config and "canvas_height" in config:
        cw = config["canvas_width"]
        ch = config["canvas_height"]
        new_w = max(x1 + w1, x2 + w2) - min(x1, x2)
        new_h = max(y1 + h1, y2 + h2) - min(y1, y2)
        if new_w > 0.45 * cw and new_h > 0.35 * ch:
            return False

    # Guard 2: Absolute vertical distance cap (characters in an equation never have v_gap > 28)
    v_top = max(y1, y2)
    v_bottom_of_upper = min(y1 + h1, y2 + h2)
    v_gap = max(0, v_top - v_bottom_of_upper)
    if v_gap > 28:
        return False

    # 1. Specialized Division Symbol (÷) Dot & Bar Clustering
    # A division glyph consists of a horizontal bar with an upper dot and lower dot.
    # The dot's horizontal center is aligned with the bar, and vertically proximate.
    for glyph_b, dot_b in [(box1, box2), (box2, box1)]:
        gx, gy, gw, gh = glyph_b
        dx, dy, dw, dh = dot_b
        if gw >= 8 and (dw <= 0.65 * gw or (dw <= 0.85 * gw and gh <= 1.4 * dh)):
            if dh <= max(15, int(0.85 * gw)):
                dot_cx = dx + 0.5 * dw
                bar_cx = gx + 0.5 * gw
                if abs(dot_cx - bar_cx) <= max(4, int(0.30 * gw)):
                    y_top = max(gy, dy)
                    y_bot_upper = min(gy + gh, dy + dh)
                    v_gap = max(0, y_top - y_bot_upper)
                    if v_gap <= max(18, min(35, int(1.2 * gw))):
                        return True

    # 2. Multi-Stroke Character Fragments (e.g. split digit '4', '+', etc.)
    # When strokes overlap vertically and have <= 2px hairline fracture:
    # Guard: Do not merge two already complete, substantial glyphs that happen to be closely spaced
    v_top = max(y1, y2)
    v_bottom = min(y1 + h1, y2 + h2)
    v_overlap = max(0, v_bottom - v_top)
    min_h = min(h1, h2)
    max_h = max(h1, h2)
    h_gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    combined_w = max(x1 + w1, x2 + w2) - min(x1, x2)

    # 2. Fractured stroke hairline connection (<= 2px gap)
    # Guard: Avoid merging adjacent independent digits or parentheses.
    # Do not merge if both are complete glyphs (w >= 9 and h >= 12).
    is_full1 = (w1 >= 9 and h1 >= 12)
    is_full2 = (w2 >= 9 and h2 >= 12)
    # Guard: Do not merge a tall parenthesis (h >= 22, ar <= 0.45) with an adjacent vertical stroke (w <= 6)
    is_paren1 = (h1 >= 22 and (w1 / float(max(1, h1))) <= 0.45 and w2 <= 6)
    is_paren2 = (h2 >= 22 and (w2 / float(max(1, h2))) <= 0.45 and w1 <= 6)
    # Guard: Do not merge a narrow stroke on the left (w <= 7) with a wide complete glyph on the right (w >= 10, ar >= 0.55)
    # (e.g. digit '1' or stem of '4' before an independent digit like '5', '2', or '8')
    left_b, right_b = (box1, box2) if x1 <= x2 else (box2, box1)
    is_left_stem_right_glyph = (left_b[2] <= 7 and right_b[2] >= 10 and (right_b[2] / float(max(1, right_b[3]))) >= 0.55)
    if not (is_full1 and is_full2) and not (is_paren1 or is_paren2) and not is_left_stem_right_glyph:
        if min_h > 0 and (v_overlap / float(min_h)) >= 0.40:
            if h_gap <= 2:
                return True

    # 3. Horizontal Overlap Calculation
    x_left = max(x1, x2)
    x_right = min(x1 + w1, x2 + w2)
    overlap_x = max(0, x_right - x_left)

    min_w = min(w1, w2)
    if min_w == 0:
        return False

    h_overlap_ratio = overlap_x / float(min_w)
    min_h_overlap = config.get("merge_min_h_overlap_ratio", 0.35)
    if h_overlap_ratio < min_h_overlap:
        return False

    # 4. Vertical Proximity Calculation (Parallel Stroke Alignment, e.g. '=')
    y_top = max(y1, y2)
    y_bottom_of_upper = min(y1 + h1, y2 + h2)
    v_gap = max(0, y_top - y_bottom_of_upper)

    # Hairline vertical break in single character (e.g. split digit 2 where v_gap <= 8)
    if v_gap <= 8:
        return True

    # Stacked parallel horizontal bars (e.g. '=')
    if (w1 >= 1.3 * h1) and (w2 >= 1.3 * h2):
        max_v_gap = min(25, max(18, int(1.2 * min_w)))
        return v_gap <= max_v_gap

    return False


def merge_bounding_boxes(
    boxes: List[Tuple[int, int, int, int]],
    config: Optional[Dict[str, Any]] = None
) -> List[Tuple[int, int, int, int]]:
    """
    Iteratively merges multi-part bounding boxes into unified symbol boxes.

    Viva Explanation:
    This algorithm implements connected component transitive clustering over
    bounding boxes. If stroke A merges with stroke B, and stroke B merges with
    stroke C (as in '÷'), all three coalesce into a single bounding box enclosing
    the entire glyph.

    Args:
        boxes: List of candidate (x, y, w, h) tuples.
        config: Tunable settings dictionary.

    Returns:
        List[Tuple[int, int, int, int]]: Merged bounding boxes.
    """
    if not boxes:
        return []

    cfg = config or SEGMENT_CONFIG
    current_boxes = sorted(list(boxes), key=lambda b: b[0])

    merged_any = True
    while merged_any:
        merged_any = False
        new_boxes: List[Tuple[int, int, int, int]] = []
        visited = [False] * len(current_boxes)

        for i in range(len(current_boxes)):
            if visited[i]:
                continue

            bx, by, bw, bh = current_boxes[i]
            visited[i] = True

            for j in range(i + 1, len(current_boxes)):
                if visited[j]:
                    continue

                if should_merge((bx, by, bw, bh), current_boxes[j], cfg):
                    # Merge box j into the current box
                    jx, jy, jw, jh = current_boxes[j]
                    new_x = min(bx, jx)
                    new_y = min(by, jy)
                    new_r = max(bx + bw, jx + jw)
                    new_b = max(by + bh, jy + jh)

                    bx = new_x
                    by = new_y
                    bw = new_r - new_x
                    bh = new_b - new_y
                    visited[j] = True
                    merged_any = True

            new_boxes.append((bx, by, bw, bh))

        current_boxes = new_boxes

    return current_boxes


def sort_boxes_left_to_right(
    boxes: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Sorts bounding boxes in reading order (strictly left-to-right by x-coordinate).

    Viva Explanation:
    Mathematical equations are read from left to right. Sorting the bounding
    rectangles by their horizontal origin (x) guarantees sequential expression
    reconstruction for the syntactic parser.
    """
    return sorted(boxes, key=lambda b: b[0])


def normalize_symbol_crop(
    binary_img: np.ndarray,
    box: Tuple[int, int, int, int],
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Crops, centers, square-pads, resizes, and skeleton-normalizes a symbol to 45x45 float32.

    Viva Explanation:
    1. Square Padding: If a digit '1' (tall and narrow) were directly resized to
       45x45 without padding, it would be horizontally stretched into a wide block.
       Padding preserves the true geometric aspect ratio of strokes.
    2. Border Margin: Convolutional filters near image borders suffer from edge
       artifacts. Providing an intentional ~4-6% border margin ensures the complete
       contour falls well inside the receptive field of first-layer filters.
    3. Stroke Width Normalization (Medial Axis Skeletonization):
       Real camera captures, webcam photos, and thick pens produce strokes that are
       3-8 pixels thick. However, the Kaggle training dataset consists of normalized
       1-pixel strokes (mean stroke half-width ~0.96 px on a 45x45 canvas).
       Thick strokes severely distort learned feature responses (e.g. bloated '2' triggers
       diagonal 'x' kernels; bloated '+' triggers corner '4' filters).
       Applying topological skeletonization (Zhang-Suen / Lee algorithm) directly on
       the 45x45 grid standardizes stroke thickness to 1 pixel, perfectly matching
       the CNN's learned filter distribution while preserving loops, junctions, and endpoints.
    4. Value Scaling: Normalizing to [0.0, 1.0] float32 ensures numerical stability.

    Args:
        binary_img: Full cleaned binary image.
        box: (x, y, w, h) tuple defining symbol boundaries.
        config: Tunable settings dictionary.

    Returns:
        np.ndarray: Normalized 45x45 float32 array in range [0.0, 1.0].
    """
    cfg = config or SEGMENT_CONFIG
    target_h, target_w = cfg.get("target_size", (45, 45))
    margin_ratio = cfg.get("padding_margin_ratio", 0.04)

    img_h, img_w = binary_img.shape[:2]
    x, y, w, h = box

    # Safely clamp crop boundaries to image dimensions
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    crop = binary_img[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]

    if crop_h == 0 or crop_w == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)

    # Determine max dimension to make square
    max_dim = max(crop_h, crop_w)
    margin = int(round(max_dim * margin_ratio))
    square_size = max_dim + 2 * margin

    # Create square canvas with black background
    square_canvas = np.zeros((square_size, square_size), dtype=np.uint8)

    # Compute centering offsets
    offset_y = margin + (max_dim - crop_h) // 2
    offset_x = margin + (max_dim - crop_w) // 2

    square_canvas[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w] = crop

    # Resize to target size (45x45)
    # Using INTER_AREA for downsampling and INTER_LINEAR for upsampling avoids cubic overshoot halos
    interp = cv2.INTER_AREA if square_size > target_h else cv2.INTER_LINEAR
    resized = cv2.resize(square_canvas, (target_w, target_h), interpolation=interp)

    # Topological stroke skeletonization to match training distribution
    # Threshold at 30/255 to reject bilinear interpolation halos while preserving thin waists & loops
    bin_thresh = resized > 30
    if np.count_nonzero(bin_thresh) == 0:
        bin_thresh = resized > 10

    if np.count_nonzero(bin_thresh) > 0:
        skel = skeletonize(bin_thresh)
        normalized = skel.astype(np.float32)
    else:
        normalized = np.zeros((target_h, target_w), dtype=np.float32)

    return normalized


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Overlays numbered bounding boxes and optional prediction labels on the image.

    Viva Explanation:
    This visualization is vital for debugging segmentation errors (under-segmentation
    where two characters merge, or over-segmentation where one character splits).
    In the UI, it provides evaluators with direct visual verification of OpenCV's
    spatial localization.

    Args:
        image: Original or preprocessed BGR image.
        boxes: List of (x, y, w, h) bounding boxes.
        labels: Optional list of predicted class strings.
        confidences: Optional list of confidence floats in [0, 1].
        config: Tunable settings dictionary.

    Returns:
        np.ndarray: Annotated BGR visualization image.
    """
    cfg = config or SEGMENT_CONFIG
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    box_color = cfg.get("box_color", (0, 220, 130))       # Vibrant Green/Teal
    box_thickness = cfg.get("box_thickness", 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, (x, y, w, h) in enumerate(boxes):
        # Draw bounding rectangle
        cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, box_thickness)

        # Build label text: e.g. "#1" or "#1: x (98%)"
        if labels is not None and idx < len(labels):
            lbl_str = labels[idx]
            if confidences is not None and idx < len(confidences):
                badge_text = f"{lbl_str} ({int(confidences[idx] * 100)}%)"
            else:
                badge_text = f"{lbl_str}"
        else:
            badge_text = f"#{idx + 1}"

        # Draw a small solid badge above the box for legibility
        font_scale = 0.45
        (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, 1)
        badge_y1 = max(0, y - th - 6)
        badge_y2 = y
        badge_x1 = x
        badge_x2 = min(vis.shape[1], x + tw + 6)

        # Draw dark badge background
        cv2.rectangle(vis, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 25, 35), -1)
        cv2.rectangle(vis, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, 1)

        # Put badge text
        cv2.putText(
            vis,
            badge_text,
            (badge_x1 + 3, badge_y2 - 3),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return vis


def validate_segmentation_sanity(
    boxes: List[Tuple[int, int, int, int]],
    raw_boxes: List[Tuple[int, int, int, int]],
    binary_img: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluates segmented bounding boxes against mathematical layout sanity criteria.
    Detects abnormal failure modes:
        1. Oversized Box Collapse: A single bounding box encompassing the entire page/writing area.
        2. Dominant Line Artifact: A box whose geometry matches an unsuppressed horizontal background line.
        3. Extreme Component Collapse: Substantial candidate strokes collapsed into <= 1 box.
        4. Crowded Overlap: Bounding boxes that severely overlap (> 50% area overlap).

    Returns:
        Dict[str, Any]:
            - 'is_valid': bool (True if all checks pass)
            - 'quality': 'valid' | 'suspicious' | 'failed'
            - 'warnings': List[str]
            - 'flags': Dict[str, bool]
    """
    img_h, img_w = binary_img.shape[:2]
    warnings: List[str] = []
    flags = {
        "oversized_box": False,
        "line_artifact_box": False,
        "extreme_collapse": False,
        "severe_box_overlap": False,
        "suspicious_wide_box": False,
        "unexpected_merge": False,
    }

    if not boxes:
        return {
            "is_valid": False,
            "quality": "failed",
            "warnings": ["No symbol bounding boxes found."],
            "flags": flags
        }

    # 1. Oversized Box Check (e.g. single box merging the entire equation and page rules)
    if len(boxes) == 1 and (boxes[0][2] > 0.40 * img_w and boxes[0][3] > 0.25 * img_h):
        flags["oversized_box"] = True
        warnings.append(
            f"Oversized single bounding box {boxes[0]} spans {boxes[0][2]}x{boxes[0][3]} px "
            f"({boxes[0][2]/img_w:.1%} width, {boxes[0][3]/img_h:.1%} height), indicating merged expression."
        )

    # 2. Dominant Line Artifact Check
    for idx, (bx, by, bw, bh) in enumerate(boxes):
        aspect = float(bw) / max(1, bh)
        if bw > 0.35 * img_w and aspect >= 6.0 and bh <= 8:
            flags["line_artifact_box"] = True
            warnings.append(
                f"Box #{idx+1} ({bx}, {by}, {bw}, {bh}) has extreme aspect ratio {aspect:.1f} "
                "characteristic of an unsuppressed background or notebook line."
            )

    # 3. Component Collapse Check
    if len(raw_boxes) >= 5 and len(boxes) <= 1:
        flags["extreme_collapse"] = True
        warnings.append(
            f"Extreme component collapse: {len(raw_boxes)} candidate contours coalesced into {len(boxes)} box."
        )

    # 4. Box Overlap Check
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ix1, iy1, iw1, ih1 = boxes[i]
            jx1, jy1, jw1, jh1 = boxes[j]
            ox = max(0, min(ix1 + iw1, jx1 + jw1) - max(ix1, jx1))
            oy = max(0, min(iy1 + ih1, jy1 + jh1) - max(iy1, jy1))
            inter_area = ox * oy
            min_area = min(iw1 * ih1, jw1 * jh1)
            if min_area > 0 and (inter_area / float(min_area)) > 0.50:
                flags["severe_box_overlap"] = True
                warnings.append(f"Boxes #{i+1} and #{j+1} severely overlap ({inter_area/min_area:.1%}).")

    # 5. Suspiciously Wide Box Check (characters in an equation are relatively square or tall)
    if len(boxes) >= 2:
        widths = [b[2] for b in boxes]
        med_w = float(np.median(widths))
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            if bw > max(24.0, 2.2 * med_w) and (float(bw) / max(1, bh)) > 1.3 and bh > 14:
                flags["suspicious_wide_box"] = True
                warnings.append(
                    f"Box #{idx+1} ({bx}, {by}, {bw}, {bh}) is suspiciously wide ({bw}px vs median {med_w:.1f}px), "
                    "suggesting multiple adjacent characters were erroneously merged."
                )

    # 6. Unexpected Component Merging Check
    subst_raw = [b for b in raw_boxes if b[2] >= 3 and b[3] >= 10]
    if len(subst_raw) >= 4 and len(boxes) < len(subst_raw) - 1:
        div_boxes = [b for b in boxes if b[2] >= 8 and (float(b[2]) / max(1, b[3])) >= 0.8]
        allowed_reduction = 2 if len(div_boxes) > 0 else 0
        if (len(subst_raw) - len(boxes)) > max(1, allowed_reduction):
            flags["unexpected_merge"] = True
            warnings.append(
                f"Unexpected contour collapse: {len(subst_raw)} substantial character strokes "
                f"merged into only {len(boxes)} boxes."
            )

    is_valid = not (flags["oversized_box"] or flags["line_artifact_box"] or flags["extreme_collapse"] or flags["severe_box_overlap"] or flags["unexpected_merge"] or flags["suspicious_wide_box"])
    quality = "valid" if is_valid else ("failed" if (flags["oversized_box"] or flags["extreme_collapse"]) else "suspicious")

    return {
        "is_valid": is_valid,
        "quality": quality,
        "warnings": warnings,
        "flags": flags
    }


def segment_symbols(
    binary_img: np.ndarray,
    original_bgr: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes the full segmentation pipeline:
    Detection -> Merging -> Left-to-Right Sorting -> Square Padding -> 45x45 Normalization.

    Args:
        binary_img: Cleaned inverted binary image.
        original_bgr: Optional BGR image for rendering visual bounding box overlay.
        config: Tunable settings dictionary.

    Returns:
        Dict[str, Any]:
            - 'boxes': Sorted (x, y, w, h) bounding boxes.
            - 'symbol_crops': List of 45x45 normalized float32 arrays.
            - 'boxed_image': Annotated visual overlay image (if original_bgr provided).
            - 'count': Number of symbols detected.
    """
    cfg = {**SEGMENT_CONFIG, **(config or {})}
    cfg["canvas_width"] = binary_img.shape[1]
    cfg["canvas_height"] = binary_img.shape[0]

    # 1. Detect external contours with candidate diagnostics
    raw_boxes, contour_diagnostics = find_symbol_contours(binary_img, cfg, return_diagnostics=True)

    # 2. Merge multi-part characters (=, ÷)
    merged_boxes = merge_bounding_boxes(raw_boxes, cfg)

    # 3. Sort reading order (left-to-right)
    sorted_boxes = sort_boxes_left_to_right(merged_boxes)

    # 3.5. Filter out isolated tiny specks and baseline outliers
    if len(sorted_boxes) >= 2:
        centers_y = [b[1] + 0.5 * b[3] for b in sorted_boxes]
        med_y = float(np.median(centers_y))
        med_h = float(np.median([b[3] for b in sorted_boxes]))
        med_w = float(np.median([b[2] for b in sorted_boxes]))
        filtered_boxes = []
        for b in sorted_boxes:
            bx, by, bw, bh = b
            cy = by + 0.5 * bh

            # Outlier in vertical distance (far away from equation line)
            if abs(cy - med_y) > max(40.0, 2.5 * med_h):
                continue
            if max(bw, bh) <= 8 and abs(cy - med_y) > 1.2 * med_h:
                continue

            # Reject isolated tiny degenerate noise specks
            # Degenerate criteria: both width and height are small (< 30% of median height)
            if bh < 0.30 * med_h and bw < 0.30 * med_h:
                # Preservation exemptions for legitimate mathematical components:
                # 1. Decimal point: sits near bottom baseline and is horizontally adjacent to symbols
                is_decimal_candidate = (
                    (by + bh >= med_y + 0.20 * med_h)
                    and any(
                        (ob != b) and (abs((bx + 0.5 * bw) - (ob[0] + 0.5 * ob[2])) <= 2.5 * med_w)
                        for ob in sorted_boxes
                    )
                )
                # 2. Exponent / superscript mark: sits in superscript zone
                is_exponent_candidate = (by + bh <= med_y - 0.25 * med_h)
                # 3. Division dot: aligned horizontally with an adjacent horizontal bar
                is_div_dot_candidate = any(
                    (ob != b)
                    and (ob[2] >= 1.3 * ob[3])
                    and (abs((bx + 0.5 * bw) - (ob[0] + 0.5 * ob[2])) <= max(6, int(0.35 * ob[2])))
                    for ob in sorted_boxes
                )
                # 4. Minus / horizontal bar: wide aspect ratio
                is_bar_candidate = (bw >= 1.8 * bh and bw >= 7)

                if not (is_decimal_candidate or is_exponent_candidate or is_div_dot_candidate or is_bar_candidate):
                    continue

            filtered_boxes.append(b)
        sorted_boxes = filtered_boxes

    # 3.6. Validate segmentation sanity against layout anomalies
    sanity = validate_segmentation_sanity(sorted_boxes, raw_boxes, binary_img, cfg)

    # 4. Extract centered, padded 45x45 crops
    symbol_crops = [
        normalize_symbol_crop(binary_img, box, cfg)
        for box in sorted_boxes
    ]

    # 5. Visual overlay
    boxed_image = None
    if original_bgr is not None:
        boxed_image = draw_bounding_boxes(original_bgr, sorted_boxes, config=cfg)

    return {
        "boxes": sorted_boxes,
        "raw_boxes": raw_boxes,
        "merged_boxes": merged_boxes,
        "contour_diagnostics": contour_diagnostics,
        "symbol_crops": symbol_crops,
        "boxed_image": boxed_image,
        "count": len(sorted_boxes),
        "sanity": sanity
    }
