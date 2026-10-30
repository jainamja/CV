"""
=============================================================================
MathVision: Preprocessing Module (Classical Computer Vision Pipeline)
=============================================================================

This module is responsible for raw image conditioning prior to character
segmentation. In real-world scenarios (handheld camera photos, webcam feeds,
or scans), handwriting exhibits significant variability in illumination,
orientation, noise, and stroke continuity.

Key Stages:
    1. Aspect-Ratio Preserving Resize: Standardizes scale to ensure morphological
       kernels operate consistently across varying camera resolutions.
    2. Document / Paper Perspective Correction: Identifies document boundaries
       via convex quadrilateral approximation and computes homography matrix
       to rectify perspective tilt.
    3. Grayscale Conversion: Removes color channels to simplify intensity processing.
    4. Gaussian Smoothing: Suppresses high-frequency camera sensor noise.
    5. Inverted Adaptive Thresholding: Accommodates non-uniform lighting/shadows
       and creates binary ink representation (foreground ink = 255, background = 0).
    6. Morphological Operations:
       - Opening: Removes isolated speckles (salt noise).
       - Closing: Bridges minute hairline gaps in pen/pencil strokes.
    7. Orientation Deskewing: Aligns text baselines horizontally using minAreaRect.
    8. Threshold Comparative Analysis: Evaluates Global vs. Otsu vs. Adaptive methods.
=============================================================================
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


# =============================================================================
# DEFAULT CONFIGURATION
# All tunable hyperparameters are centralized here to avoid magic numbers.
# Every parameter is documented with its mathematical / practical purpose.
# =============================================================================
PREPROCESS_CONFIG: Dict[str, Any] = {
    # 1. Resize Settings
    "max_width": 1000,              # Max width in pixels to keep kernel sizes effective
    "interpolation": cv2.INTER_AREA,# Area interpolation avoids moire artifacts during downscaling

    # 2. Perspective Correction Settings
    "perspective_enabled": True,    # Attempt document/sheet quadrilateral detection
    "min_paper_area_ratio": 0.20,   # Paper contour must cover at least 20% of image area
    "approx_poly_epsilon_ratio": 0.02, # Epsilon factor for approxPolyDP (2% of perimeter)

    # 3. Gaussian Blur Settings
    # Light Gaussian kernel removes high-frequency thermal & sensor noise without fattening thin strokes
    "blur_kernel_size": (3, 3),     # Compact kernel to preserve crisp 1-2px stroke widths
    "blur_sigma_x": 0,              # 0 indicates sigma calculated from kernel dimensions

    # 4. Adaptive Threshold Settings
    # Inverted so ink = 255 (white foreground) and paper = 0 (black background).
    # This standardizes input for contour detection and CNN classification.
    "adaptive_max_val": 255,
    "adaptive_method": cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Gaussian weighting around local neighborhood
    "adaptive_type": cv2.THRESH_BINARY_INV,
    "adaptive_block_size": 15,      # Neighborhood size in pixels (must be odd > 1)
    "adaptive_c": 8,                # Higher C value prevents thin strokes from blooming into thick blobs

    # 5. Threshold Selection & Comparison Settings
    "threshold_method": "adaptive", # Default to local adaptive binarization for lighting & margin robustness
    "global_thresh_val": 127,       # Midpoint value for naive binary thresholding

    # 6. Illumination Flattening Settings (Retinex / Morphological Background Subtraction)
    "illumination_flattening_enabled": True, # Removes shadows, vignetting, and ambient gradients
    "illumination_kernel_size": (41, 41),    # Large morphological kernel capturing paper background

    # 7. Morphological Refinement
    # Opening (erosion followed by dilation) eliminates isolated single-pixel salt noise.
    # Closing (dilation followed by erosion) bridges micro gaps caused by dry ink strokes.
    "morph_kernel_open": (1, 1),    # (1, 1) preserves delicate 1px crossbars (e.g. in '4'); noise is rejected by CC analysis
    "morph_kernel_close": (2, 2),   # Compact 2x2 bridges broken strokes without filling small loops (e.g. '6', '8')
    
    # 8. Background Horizontal Line Suppression Settings
    "line_suppression_enabled": True, # Removes notebook rules, table borders, and scanner boundary bars
    "line_min_width_ratio": 0.22,     # Minimum length relative to image width to qualify as background line

    # 9. Deskew Settings
    "deskew_enabled": False,        # Optional orientation rectification
    "max_deskew_angle": 30.0,       # Angles beyond 30 deg are likely non-horizontal layouts
}


def load_image(image_input: Any) -> np.ndarray:
    """
    Loads an image from a file path, byte stream, or validates an existing numpy array.

    Args:
        image_input: File path (str), raw image bytes, or existing np.ndarray.

    Returns:
        np.ndarray: BGR image array.

    Raises:
        ValueError: If the image cannot be read or decoded.
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Failed to read image from path: '{image_input}'")
        return image

    if isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from raw bytes.")
        return image

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.ndim == 3 and image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        return image_input.copy()

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def resize_preserve_aspect(image: np.ndarray, max_width: Optional[int] = None) -> np.ndarray:
    """
    Resizes image to a fixed maximum width while preserving the exact aspect ratio.

    Viva Explanation:
    Normalizing the overall scale ensures that fixed morphological kernels (e.g., 3x3)
    and contour area thresholds operate uniformly regardless of whether the input photo
    was taken with a 48MP smartphone or a 720p webcam.

    Args:
        image: BGR or Grayscale image array.
        max_width: Maximum allowable width in pixels. If None, uses PREPROCESS_CONFIG.

    Returns:
        np.ndarray: Rescaled image.
    """
    if max_width is None:
        max_width = PREPROCESS_CONFIG["max_width"]

    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy()

    aspect_ratio = float(height) / float(width)
    new_width = max_width
    new_height = int(round(new_width * aspect_ratio))

    interpolation = PREPROCESS_CONFIG["interpolation"]
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 quadrilateral coordinates in canonical sequence:
    [top-left, top-right, bottom-right, bottom-left].

    Viva Explanation:
    Perspective warping requires matching 4 source points to 4 destination points.
    - Top-left has the minimum (x + y) sum.
    - Bottom-right has the maximum (x + y) sum.
    - Top-right has the minimum (y - x) difference.
    - Bottom-left has the maximum (y - x) difference.
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates: x + y
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right

    # Difference of coordinates: y - x
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left

    return rect


def correct_perspective(
    image: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, bool]:
    """
    Detects paper boundaries and performs perspective rectification via Homography.

    Viva Explanation:
    Photos taken at an angle distort symbol geometry (keystone effect), turning
    circles into ellipses and slanting vertical strokes. Finding the sheet's convex
    quadrilateral and projecting it to a flat plane restores true spatial proportions.
    If no quadrilateral paper sheet is found (e.g., canvas drawings or tight crops),
    the function skips gracefully without altering the image.

    Args:
        image: BGR input image.
        config: Tunable settings dictionary.

    Returns:
        Tuple[np.ndarray, bool]: Rectified image and boolean flag indicating whether
                                 perspective warp was applied.
    """
    cfg = config or PREPROCESS_CONFIG
    if not cfg.get("perspective_enabled", True):
        return image.copy(), False

    orig_h, orig_w = image.shape[:2]
    total_area = orig_h * orig_w

    # Convert to grayscale and blur to smooth fine texture
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Otsu-guided Canny thresholds
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # Dilate edges to close gaps in paper boundary
    dilated_edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy(), False

    # Sort contours by descending area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    min_area = total_area * cfg.get("min_paper_area_ratio", 0.20)
    eps_ratio = cfg.get("approx_poly_epsilon_ratio", 0.02)

    paper_quad = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            paper_quad = approx
            break

    if paper_quad is None:
        return image.copy(), False

    # Order corners
    src_pts = order_quad_points(paper_quad.reshape(4, 2))
    (tl, tr, br, bl) = src_pts

    # Compute width of new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))

    # Compute height of new image
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))

    if max_w < 50 or max_h < 50:
        return image.copy(), False

    dst_pts = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, transform_matrix, (max_w, max_h), flags=cv2.INTER_LINEAR)
    return warped, True


def crop_black_margins(
    image: np.ndarray,
    threshold: float = 15.0
) -> Tuple[np.ndarray, bool, Tuple[int, int, int, int]]:
    """
    Detects and crops uniform black/dark margin pillarboxes/letterboxes (e.g. from webcams or screenshots).

    Viva Explanation:
    Webcam captures and mobile screenshots frequently contain large pitch-black pillarboxes
    or letterbox padding margins on the borders. Because these dark regions occupy a massive fraction
    (e.g. 50-60%) of the image pixels with intensity ~0, global binarization algorithms (like Otsu)
    break down: the global threshold is pulled down into the ink-intensity range, corrupting the
    paper binarization, completely wiping out faint pencil/pen strokes, and converting the black
    margins into enormous 80,000+ pixel white noise blocks.
    Cropping the black margin borders to the active paper bounding box restores the true bimodal
    distribution between paper (~120-200) and ink (~30-60), enabling clean segmentation of all glyphs.

    Args:
        image: BGR or Grayscale image array.
        threshold: Mean intensity cutoff below which an edge row/col is considered black padding.

    Returns:
        Tuple[np.ndarray, bool, Tuple[int, int, int, int]]:
            - Cropped image (or original copy if no black margins).
            - Boolean flag indicating whether black margins were cropped.
            - (x, y, w, h) bounding rectangle of the retained active region in original coordinates.
    """
    if image is None or image.size == 0:
        return image, False, (0, 0, 0, 0)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    col_means = gray.mean(axis=0)
    row_means = gray.mean(axis=1)

    valid_cols = np.where(col_means > threshold)[0]
    valid_rows = np.where(row_means > threshold)[0]

    img_h, img_w = image.shape[:2]
    if len(valid_cols) == 0 or len(valid_rows) == 0:
        return image.copy(), False, (0, 0, img_w, img_h)

    x1, x2 = int(valid_cols[0]), int(valid_cols[-1]) + 1
    y1, y2 = int(valid_rows[0]), int(valid_rows[-1]) + 1

    # Check if substantial black margin was detected (> 5% of width or height)
    has_left_margin = x1 > int(0.05 * img_w)
    has_right_margin = (img_w - x2) > int(0.05 * img_w)
    has_top_margin = y1 > int(0.05 * img_h)
    has_bottom_margin = (img_h - y2) > int(0.05 * img_h)

    if has_left_margin or has_right_margin or has_top_margin or has_bottom_margin:
        # Apply a conservative 2px inset away from black margin transition boundaries
        # to ensure no 1-pixel boundary transition artifacts remain
        safe_x1 = min(img_w - 1, x1 + 2) if has_left_margin else x1
        safe_x2 = max(0, x2 - 2) if has_right_margin else x2
        safe_y1 = min(img_h - 1, y1 + 2) if has_top_margin else y1
        safe_y2 = max(0, y2 - 2) if has_bottom_margin else y2

        if safe_x2 > safe_x1 + 20 and safe_y2 > safe_y1 + 20:
            crop = image[safe_y1:safe_y2, safe_x1:safe_x2].copy()
            bbox = (safe_x1, safe_y1, safe_x2 - safe_x1, safe_y2 - safe_y1)
            return crop, True, bbox

    return image.copy(), False, (0, 0, img_w, img_h)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts BGR image to single-channel 8-bit Grayscale.

    Viva Explanation:
    Color information (chrominance) does not carry mathematical symbol semantics.
    Discarding color channels reduces memory complexity from 3 to 1 byte per pixel
    and prepares the image for intensity-based gradient and threshold analysis.
    """
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def flatten_illumination(gray: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Normalizes uneven paper background illumination, vignetting, and shadows via morphological division.

    Viva Explanation:
    Handheld camera captures and webcam images inherently suffer from lighting gradients,
    shadows, and vignetting (dark corners). A fixed or naive threshold fails because
    shadows in corners are darker than ink strokes in well-lit areas.
    By applying a large morphological dilation (kernel size 41x41), we filter out thin
    character strokes (< 20px wide) and obtain an estimated smooth paper background illumination
    field B(x, y). Dividing the grayscale image by this background field flattens the paper
    surface to uniform white (255) while preserving thin, faint pencil/pen strokes, enabling
    clean binarization without edge shadows or lost strokes.
    """
    cfg = config or PREPROCESS_CONFIG
    if not cfg.get("illumination_flattening_enabled", True):
        return gray.copy()

    ksize = cfg.get("illumination_kernel_size", (41, 41))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    norm = np.clip((gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 255.0, 0, 255).astype(np.uint8)
    return norm


def apply_gaussian_blur(gray: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Applies 2D Gaussian filter for isotropic noise smoothing.

    Viva Explanation:
    Thresholding without prior smoothing leads to ragged edges and salt-and-pepper noise
    because high-frequency camera noise crosses the threshold boundary.
    Gaussian convolution acts as a low-pass filter, attenuating pixel-level variance.
    """
    cfg = config or PREPROCESS_CONFIG
    ksize = cfg.get("blur_kernel_size", (5, 5))
    sigma_x = cfg.get("blur_sigma_x", 0)
    return cv2.GaussianBlur(gray, ksize, sigma_x)


def apply_adaptive_threshold(blurred: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Computes an inverted binary image using local Gaussian-weighted adaptive thresholding.

    Viva Explanation:
    Why not Global or Otsu thresholding?
    Handwritten photos frequently suffer from shadows, vignetting, or uneven ambient
    lighting. Global thresholding assumes a bimodal intensity histogram across the entire
    image, causing shadowed areas to be classified as ink.
    Adaptive thresholding calculates a dynamic threshold for each local pixel neighborhood:
        T(x, y) = Mean(Gaussian_neighborhood(x, y)) - C
    We use THRESH_BINARY_INV so ink pixels become white (255) and paper becomes black (0).
    Contour detection algorithms in OpenCV look for white foreground objects on black background.
    """
    cfg = config or PREPROCESS_CONFIG
    max_val = cfg.get("adaptive_max_val", 255)
    method = cfg.get("adaptive_method", cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    thresh_type = cfg.get("adaptive_type", cv2.THRESH_BINARY_INV)
    block_size = cfg.get("adaptive_block_size", 15)
    c_val = cfg.get("adaptive_c", 8)

    # Ensure block_size is odd and greater than 1
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    return cv2.adaptiveThreshold(blurred, max_val, method, thresh_type, block_size, c_val)


def compare_thresholds(gray: np.ndarray, config: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """
    Generates Global, Otsu, and Adaptive thresholding results side-by-side.

    Viva Explanation:
    This comparison function demonstrates why adaptive and normalized Otsu thresholding are chosen:
    1. Global Thresholding (fixed threshold T=127): Evaluated on blurred normalized gray.
    2. Otsu's Thresholding: Automatically minimizes intra-class variance assuming a bimodal
       histogram. Computed on illumination-normalized grayscale to preserve crisp, thin stroke width
       and completely eliminate boundary lighting gradients.
    3. Adaptive Gaussian Thresholding: Operates locally, maintaining consistent stroke width.

    Returns:
        Dict[str, np.ndarray]: Mapping of threshold names to inverted binary images.
    """
    cfg = config or PREPROCESS_CONFIG
    norm_gray = flatten_illumination(gray, cfg)
    blurred = apply_gaussian_blur(norm_gray, cfg)

    # 1. Plain Global Thresholding on normalized image
    t_val = cfg.get("global_thresh_val", 127)
    _, global_thresh = cv2.threshold(blurred, t_val, 255, cv2.THRESH_BINARY_INV)

    # 2. Otsu Thresholding (computed on illumination-normalized grayscale to preserve crisp, thin stroke width)
    _, otsu_thresh = cv2.threshold(norm_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Adaptive Gaussian Thresholding
    adaptive_thresh = apply_adaptive_threshold(blurred, cfg)

    return {
        "global": global_thresh,
        "otsu": otsu_thresh,
        "adaptive": adaptive_thresh
    }


def apply_morphology(binary_img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Applies mathematical morphology (Opening then Closing) to clean the binary image.

    Viva Explanation:
    - Morphological Opening = Erosion followed by Dilation (A ∘ B = (A ⊖ B) ⊕ B):
      Removes small foreground specks (salt noise) and pen splatters that are smaller
      than the structuring element, without altering the overall character topology.
    - Morphological Closing = Dilation followed by Erosion (A • B = (A ⊕ B) ⊖ B):
      Bridges tiny discontinuities, pinholes, and broken strokes caused by faint pencil
      markings or rapid pen movements.
    """
    cfg = config or PREPROCESS_CONFIG
    k_open = cfg.get("morph_kernel_open", (1, 1))
    k_close = cfg.get("morph_kernel_close", (2, 2))

    current = binary_img.copy()

    # Step 1: Remove isolated foreground noise (only if kernel > (1, 1))
    if k_open[0] > 1 and k_open[1] > 1:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_open)
        current = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel_open)

    # Step 2: Bridge hairline gaps in strokes
    if k_close[0] > 1 or k_close[1] > 1:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, k_close)
        current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel_close)

    return current


def suppress_horizontal_lines(
    binary_img: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Suppresses long horizontal background lines, notebook rules, and scanner boundary bars
    while strictly preserving legitimate handwritten operators (minus '-', plus '+',
    division '÷', equals '=') and character loops.

    Viva Explanation:
    Ruled notebook paper, table borders, and document scanner edge bands create long,
    continuous horizontal structures. When left unsuppressed, these structures:
    1. Intersect or vertically bridge multiple characters, causing transitive contour
       clustering to coalesce the entire expression into a single oversized bounding box.
    2. Distort learned CNN features, causing merged crops to be erroneously classified
       as '=' with high false confidence.
    To distinguish background lines from legitimate handwritten strokes:
    - Scale-Aware Minimum Length: Mathematical operator strokes ('-', '+', '÷', '=')
      rarely exceed 10-15% of the writing canvas width (typically 15-40px). Background
      rules and page borders span 22-100% of the canvas width (>= 45px).
    - Morphological Filtering: A horizontal structuring element of length L = max(45, 0.22*W)
      isolates lines longer than L, leaving all characters and operators untouched.
    - Connected Component Aspect Analysis: Components with aspect ratio w/h >= 6.0 and
      width >= L are flagged as lines.
    - Vertical Stroke Bridging: Where a genuine vertical stroke (e.g. '|' in '+', '4', '1')
      crossed a notebook line, subtracting the line creates a 1-2px gap. Vertical closing
      bridges this hairline intersection, preserving stroke continuity.

    Args:
        binary_img: Single-channel inverted binary image (ink = 255, background = 0).
        config: Tunable settings dictionary.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - Cleaned binary image with horizontal lines suppressed.
            - Detected horizontal line mask (ink = 255).
            - Diagnostic metadata dictionary.
    """
    if binary_img is None or binary_img.size == 0:
        return binary_img, np.zeros_like(binary_img), {"lines_detected": False, "pixels_removed": 0, "min_line_len": 45}

    cfg = config or PREPROCESS_CONFIG
    img_h, img_w = binary_img.shape[:2]

    min_line_len = max(35, int(img_w * cfg.get("line_min_width_ratio", 0.18)))

    # 1. Morphological horizontal opening isolates continuous horizontal lines >= min_line_len
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len, 1))
    morph_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horiz_kernel)

    # 2. Connected-component candidate lines (captures slightly fragmented or non-uniform lines)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    cc_lines = np.zeros_like(binary_img)
    for i in range(1, num_labels):
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = float(cw) / max(1, ch)
        if cw >= min_line_len and aspect >= 6.0 and ch <= 10:
            cc_lines[labels == i] = 255

        # Border & boundary-zone line remnants (e.g. page/notebook rules near image perimeter)
        touches_border = (cx <= 2 or (cx + cw) >= img_w - 2 or cy <= 2 or (cy + ch) >= img_h - 2)
        is_boundary_zone = (cy >= int(0.70 * img_h) or (cy + ch) <= int(0.18 * img_h))
        if (touches_border or is_boundary_zone) and aspect >= 6.0 and ch <= 5 and cw >= 35:
            cc_lines[labels == i] = 255

    # 3. Combined line mask with conservative vertical dilation (2-3px) to capture line boundaries
    line_mask = cv2.bitwise_or(morph_lines, cc_lines)
    pixels_removed = int(cv2.countNonZero(line_mask))

    if pixels_removed > 0:
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated_mask = cv2.dilate(line_mask, kernel_v)
        # 4. Subtract line mask
        cleaned = cv2.bitwise_and(binary_img, cv2.bitwise_not(dilated_mask))
        # 5. Bridge hairline vertical fractures where vertical character strokes crossed the line
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    else:
        dilated_mask = line_mask
        cleaned = binary_img.copy()

    stats_dict = {
        "lines_detected": pixels_removed > 0,
        "pixels_removed": pixels_removed,
        "min_line_len": min_line_len
    }
    return cleaned, dilated_mask, stats_dict


def detect_equation_writing_region(
    binary_img: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[int, int, bool, Dict[str, Any]]:
    """
    Identifies the vertical span [y1, y2] of the primary equation writing region,
    excluding distant page borders, scanner rules, and table margins.

    Viva Explanation:
    Handwritten mathematical expressions naturally form a coherent horizontal text line
    composed of distinct, character-sized connected components. In contrast, paper/table
    boundaries and dark scanner letterboxes consist of large continuous dark masses or
    isolated border slivers separated from the equation by wide empty gaps (> 1.2x line height).
    Rather than naively counting raw foreground pixels (which biases detection toward dark
    border noise), this algorithm uses connected-component structural filtering:
    1. Identifies plausible character candidates (bounded aspect ratio, character scale).
    2. Clusters character candidates sharing a horizontal baseline into text lines.
    3. Mandates validity checks: the selected writing region must contain >= 2 characters
       and preserve >= 70% of all detected character candidates.
    4. If any check fails or no distant outlier boundaries exist, falls back safely to
       the full uncropped paper canvas (zero silent crop failures).
    """
    if binary_img is None or binary_img.size == 0:
        return 0, 0, False, {"status": "empty_input"}

    img_h, img_w = binary_img.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    char_candidates = []
    outlier_candidates = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        aspect = float(w) / max(1, h)

        # Plausible character criteria
        is_char = (h >= 8 and h <= 0.70 * img_h and w >= 3 and w <= 0.65 * img_w and aspect >= 0.15 and aspect <= 4.0 and area >= 18)
        touches_border = (x <= 1 or (x + w) >= img_w - 1 or y <= 1 or (y + h) >= img_h - 1)
        is_border_blob = touches_border and (h > 0.35 * img_h or w > 0.35 * img_w or aspect > 3.0 or area > 300)

        if is_char and not is_border_blob:
            char_candidates.append({'bbox': (x, y, w, h), 'area': area, 'y_mid': y + h / 2.0, 'y1': y, 'y2': y + h, 'x1': x, 'x2': x + w})
        else:
            outlier_candidates.append({'bbox': (x, y, w, h), 'area': area, 'y1': y, 'y2': y + h})

    if len(char_candidates) < 2:
        return 0, img_h, False, {'status': 'insufficient_chars'}

    # Group characters into text-line clusters
    char_candidates.sort(key=lambda c: c['y_mid'])
    clusters = []
    for c in char_candidates:
        assigned = False
        for cl in clusters:
            cl_med_h = np.median([ch['bbox'][3] for ch in cl])
            cl_med_ymid = np.median([ch['y_mid'] for ch in cl])
            if abs(c['y_mid'] - cl_med_ymid) <= max(18, int(1.2 * cl_med_h)):
                cl.append(c)
                assigned = True
                break
        if not assigned:
            clusters.append([c])

    scored_clusters = []
    for cl in clusters:
        count = len(cl)
        x_span = max(c['x2'] for c in cl) - min(c['x1'] for c in cl)
        tot_area = sum(c['area'] for c in cl)
        score = count * 1000 + x_span + tot_area
        y1 = min(c['y1'] for c in cl)
        y2 = max(c['y2'] for c in cl)
        scored_clusters.append((score, count, y1, y2, cl))

    scored_clusters.sort(key=lambda x: x[0], reverse=True)
    best_score, best_count, best_y1, best_y2, best_chars = scored_clusters[0]

    # Mandatory validity check: primary writing line must have >= 2 characters or significant glyph
    if best_count < 2 and best_chars[0]['area'] < 50:
        return 0, img_h, False, {'status': 'single_small_char'}

    line_h = best_y2 - best_y1
    # Check if there are distant outlier components (e.g. table border below or header rule above)
    has_bottom_outlier = any(o['y1'] >= best_y2 + max(35, int(1.2 * line_h)) for o in outlier_candidates)
    has_top_outlier = any(o['y2'] <= best_y1 - max(35, int(1.2 * line_h)) for o in outlier_candidates)

    if not has_bottom_outlier and not has_top_outlier:
        return 0, img_h, False, {'status': 'no_outlier_boundaries'}

    pad = max(20, int(0.40 * line_h))
    y_top = max(0, int(best_y1 - pad)) if has_top_outlier else 0
    y_bot = min(img_h, int(best_y2 + pad)) if has_bottom_outlier else img_h

    # Priority 2: Ensure writing is preserved in proposed crop
    chars_inside = sum(1 for c in char_candidates if c['y_mid'] >= y_top and c['y_mid'] <= y_bot)
    if chars_inside < 0.70 * len(char_candidates):
        return 0, img_h, False, {'status': 'rejected_dropped_chars'}

    return y_top, y_bot, True, {'status': 'applied', 'best_count': best_count, 'char_span': (int(best_y1), int(best_y2))}


def detect_writing_region_bounds(
    binary_img: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[int, int]:
    """Compatibility wrapper returning (y_top, y_bot) for writing region detection."""
    y1, y2, _, _ = detect_equation_writing_region(binary_img, config)
    return y1, y2


def deskew_image(binary_img: np.ndarray, config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
    """
    Estimates and rectifies orientation tilt of handwritten equations.

    Viva Explanation:
    When writing on unlined paper, equations often slope upward or downward.
    Extracting non-zero ink coordinates and fitting a Minimum Area Bounding Box (cv2.minAreaRect)
    yields the dominant inclination angle θ. Rotating by -θ aligns the equation
    horizontally, improving bounding box separation and exponent elevation detection.

    Args:
        binary_img: Inverted binary image (ink = 255).
        config: Configuration dictionary.

    Returns:
        Tuple[np.ndarray, float]: Deskewed binary image and applied rotation angle in degrees.
    """
    cfg = config or PREPROCESS_CONFIG
    if not cfg.get("deskew_enabled", False):
        return binary_img.copy(), 0.0

    # Extract coordinates of all foreground ink pixels
    pts = cv2.findNonZero(binary_img)
    if pts is None or len(pts) < 50:
        return binary_img.copy(), 0.0

    rect = cv2.minAreaRect(pts)
    angle = rect[-1]

    # OpenCV minAreaRect returns angle in [-90, 0)
    if angle < -45.0:
        angle = -(90.0 + angle)
    else:
        angle = -angle

    max_angle = cfg.get("max_deskew_angle", 30.0)
    if abs(angle) > max_angle or abs(angle) < 0.5:
        # Ignore negligible angles or extreme angles (which are likely vertical artifacts)
        return binary_img.copy(), 0.0

    (h, w) = binary_img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        binary_img,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return deskewed, angle


def preprocess_pipeline(
    image_input: Any,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Orchestrates the complete preprocessing pipeline, capturing every intermediate state.

    Args:
        image_input: Image path, bytes, or numpy BGR array.
        config: Optional parameter override dictionary.

    Returns:
        Dict[str, Any]: Dictionary containing all intermediate stages:
            - 'original': Loaded BGR image.
            - 'resized': Scaled BGR image.
            - 'perspective_warped': Perspective rectified BGR image.
            - 'paper_detected': Boolean indicating if document paper was found.
            - 'gray': Single-channel grayscale image.
            - 'blurred': Gaussian blurred grayscale image.
            - 'threshold_comparison': Dict with 'global', 'otsu', and 'adaptive' binary images.
            - 'binary': Selected inverted binary image (adaptive threshold).
            - 'cleaned': Morphologically opened and closed binary image.
            - 'deskewed': Deskewed binary image.
            - 'skew_angle': Angle rotated during deskewing.
    """
    cfg = {**PREPROCESS_CONFIG, **(config or {})}

    # 1. Ingestion
    original = load_image(image_input)

    # 2. Scale normalization
    resized = resize_preserve_aspect(original, cfg["max_width"])

    # 3. Document perspective rectification
    warped, paper_detected = correct_perspective(resized, cfg)
    working_bgr = warped if paper_detected else resized

    # 3.5. Black/dark margin pillarbox & letterbox auto-cropping
    active_bgr, margin_cropped, margin_bbox = crop_black_margins(working_bgr)

    # 4. Grayscale & Noise reduction
    gray = to_grayscale(active_bgr)
    norm_gray = flatten_illumination(gray, cfg)
    blurred = apply_gaussian_blur(norm_gray, cfg)

    # 5. Threshold comparisons (Global vs Otsu vs Adaptive)
    thresh_comp = compare_thresholds(gray, cfg)
    thresh_method = cfg.get("threshold_method", "adaptive").lower()
    raw_selected = thresh_comp.get(thresh_method, thresh_comp["adaptive"])
    binary = raw_selected

    # 6. Morphological filtering (remove speckles & join strokes)
    cleaned = apply_morphology(binary, cfg)

    # 6.5. Horizontal background-line / notebook rule suppression
    binary_before_line_suppression = cleaned.copy()
    if cfg.get("line_suppression_enabled", True):
        cleaned, horiz_line_mask, line_stats = suppress_horizontal_lines(cleaned, cfg)
    else:
        horiz_line_mask = np.zeros_like(cleaned)
        line_stats = {"lines_detected": False, "pixels_removed": 0, "min_line_len": 45}

    cleaned_comp = {}
    for name, b_img in thresh_comp.items():
        m_img = apply_morphology(b_img, cfg)
        if cfg.get("line_suppression_enabled", True):
            c_img, _, _ = suppress_horizontal_lines(m_img, cfg)
        else:
            c_img = m_img
        cleaned_comp[name] = c_img

    # 6.6. Tighten writing region: exclude distant table/scanner boundary noise outside equation writing line
    y_top, y_bot, writing_crop_applied, writing_diag = detect_equation_writing_region(cleaned, cfg)

    # Visualization of margin crop with detected writing bounds drawn on it
    margin_crop_with_bounds = active_bgr.copy()
    if writing_crop_applied:
        cv2.rectangle(margin_crop_with_bounds, (0, y_top), (active_bgr.shape[1] - 1, y_bot - 1), (0, 220, 130), 2)
        cv2.putText(margin_crop_with_bounds, f"Writing Region: y=[{y_top}:{y_bot}]", (5, max(15, y_top - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 130), 1)

    if writing_crop_applied:
        final_binary = cleaned[y_top:y_bot, :].copy()
        final_resized = active_bgr[y_top:y_bot, :].copy()
        final_active_crop = active_bgr[y_top:y_bot, :].copy()
        final_binary_before = binary_before_line_suppression[y_top:y_bot, :].copy()
        final_horiz_mask = horiz_line_mask[y_top:y_bot, :].copy()
        final_cleaned_comp = {name: c_img[y_top:y_bot, :].copy() for name, c_img in cleaned_comp.items()}
    else:
        final_binary = cleaned
        final_resized = active_bgr
        final_active_crop = active_bgr
        final_binary_before = binary_before_line_suppression
        final_horiz_mask = horiz_line_mask
        final_cleaned_comp = cleaned_comp

    # 7. Orientation deskew
    deskewed, skew_angle = deskew_image(final_binary, cfg)

    return {
        "original": original,
        "resized": final_resized,
        "active_crop": final_active_crop,
        "margin_crop_image": active_bgr,
        "margin_crop_with_bounds": margin_crop_with_bounds,
        "writing_crop_applied": writing_crop_applied,
        "writing_bounds": (int(y_top), int(y_bot)),
        "writing_diag": writing_diag,
        "perspective_warped": warped if paper_detected else None,
        "paper_detected": paper_detected,
        "margin_crop_applied": margin_cropped,
        "margin_crop_bbox": margin_bbox,
        "gray": gray,
        "blurred": blurred,
        "threshold_comparison": thresh_comp,
        "threshold_comparison_cleaned": final_cleaned_comp,
        "binary_before_line_suppression": final_binary_before,
        "horizontal_line_mask": final_horiz_mask,
        "line_suppression_stats": line_stats,
        "binary": final_binary,
        "cleaned": final_binary,
        "deskewed": deskewed,
        "skew_angle": skew_angle
    }
