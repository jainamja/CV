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

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2


# =============================================================================
# SEGMENTATION CONFIGURATION
# Tunable parameters to eliminate magic numbers.
# =============================================================================
SEGMENT_CONFIG: Dict[str, Any] = {
    # 1. Noise Rejection
    # Rejects minuscule specks (e.g. paper grain, dust)
    "min_contour_area": 10,         # Absolute minimum pixel area
    "min_height_ratio": 0.02,       # Min height relative to total image height
    "min_width_pixels": 2,          # Rejects 1-pixel wide vertical noise lines
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
    config: Optional[Dict[str, Any]] = None
) -> List[Tuple[int, int, int, int]]:
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

    Returns:
        List[Tuple[int, int, int, int]]: List of (x, y, w, h) bounding boxes.
    """
    cfg = config or SEGMENT_CONFIG
    img_h, img_w = binary_img.shape[:2]

    # Find external contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    min_area = cfg.get("min_contour_area", 10)
    min_stroke_pixels = cfg.get("min_stroke_pixels", 8)
    min_h = max(3, int(img_h * cfg.get("min_height_ratio", 0.02)))
    min_w = cfg.get("min_width_pixels", 2)

    valid_boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Noise filter: reject minuscule specks that are too tiny in span or foreground pixel count.
        # Note on cv2.contourArea: Open/thin handwritten strokes (1-2 px thick) have degenerate
        # polygon contours whose enclosed shoelace polygon area (cv2.contourArea) is ~0.
        # We reject minuscule noise using bounding box span and actual ink pixel count (cv2.countNonZero).
        if max(w, h) < 6 or min(w, h) < 1:
            continue

        stroke_pixels = cv2.countNonZero(binary_img[y:y+h, x:x+w])
        if stroke_pixels < min_stroke_pixels:
            continue

        valid_boxes.append((x, y, w, h))

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
    Both exhibit:
        1. Substantial horizontal overlap: The x-projections of the two strokes
           overlap for at least 40% of the narrower stroke's width.
        2. Small vertical distance: The vertical gap between the bottom of the upper
           stroke and the top of the lower stroke is smaller than 1.25x the height
           of the taller stroke.
    If both spatial conditions are met, the two boxes are merged.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 1. Horizontal Overlap Calculation
    x_left = max(x1, x2)
    x_right = min(x1 + w1, x2 + w2)
    overlap_x = max(0, x_right - x_left)

    min_w = min(w1, w2)
    if min_w == 0:
        return False

    h_overlap_ratio = overlap_x / float(min_w)
    min_h_overlap = config.get("merge_min_h_overlap_ratio", 0.40)
    if h_overlap_ratio < min_h_overlap:
        return False

    # 2. Vertical Proximity Calculation
    # Find vertical gap between the two boxes
    y_top = max(y1, y2)
    y_bottom_of_upper = min(y1 + h1, y2 + h2)
    v_gap = max(0, y_top - y_bottom_of_upper)

    max_h = max(h1, h2)
    min_w = min(w1, w2)
    ratio_gap = max_h * config.get("merge_max_v_gap_ratio", 2.5)
    width_gap = min_w * 0.80 # For horizontal strokes like '=', vertical gap is proportional to stroke length
    min_pixels_gap = config.get("merge_min_v_gap_pixels", 30)

    max_gap_allowed = max(ratio_gap, width_gap, min_pixels_gap)

    return v_gap <= max_gap_allowed


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
    current_boxes = list(boxes)

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
    Crops, centers, square-pads, and resizes a symbol to 45x45 normalized float32.

    Viva Explanation:
    1. Square Padding: If a digit '1' (tall and narrow) were directly resized to
       45x45 without padding, it would be horizontally stretched into a wide block.
       Padding preserves the true geometric aspect ratio of strokes.
    2. Border Margin: Convolutional filters near image borders suffer from edge
       artifacts. Providing an intentional ~18% border margin ensures the complete
       contour falls well inside the receptive field of first-layer filters.
    3. Normalization: Dividing by 255.0 scales pixel values to [0.0, 1.0], preventing
       vanishing or exploding gradients during CNN backpropagation.

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
    interp = cfg.get("interpolation_down", cv2.INTER_AREA) if square_size > target_h else cfg.get("interpolation_up", cv2.INTER_CUBIC)
    resized = cv2.resize(square_canvas, (target_w, target_h), interpolation=interp)

    # Normalize to [0.0, 1.0] float32
    normalized = resized.astype(np.float32) / 255.0
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

    # 1. Detect external contours
    raw_boxes = find_symbol_contours(binary_img, cfg)

    # 2. Merge multi-part characters (=, ÷)
    merged_boxes = merge_bounding_boxes(raw_boxes, cfg)

    # 3. Sort reading order (left-to-right)
    sorted_boxes = sort_boxes_left_to_right(merged_boxes)

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
        "symbol_crops": symbol_crops,
        "boxed_image": boxed_image,
        "count": len(sorted_boxes)
    }
