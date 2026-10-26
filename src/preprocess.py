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

    # 5. Global & Otsu Comparison Settings
    "global_thresh_val": 127,       # Midpoint value for naive binary thresholding

    # 6. Morphological Refinement
    # Opening (erosion followed by dilation) eliminates isolated single-pixel salt noise.
    # Closing (dilation followed by erosion) bridges micro gaps caused by dry ink strokes.
    "morph_kernel_open": (2, 2),    # Compact kernel to remove micro noise without thinning strokes
    "morph_kernel_close": (2, 2),   # Bridges broken handwriting strokes without blooming
    
    # 7. Deskew Settings
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


def apply_gaussian_blur(gray: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Applies 2D Gaussian filter for isotropic noise smoothing.

    Viva Explanation:
    Thresholding without prior smoothing leads to ragged edges and salt-and-pepper noise
    because high-frequency camera noise crosses the threshold threshold boundary.
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
    c_val = cfg.get("adaptive_c", 4)

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
    This comparison function demonstrates why adaptive thresholding is chosen:
    1. Global Thresholding (fixed threshold T=127): Fails when lighting is dim or bright.
    2. Otsu's Thresholding: Automatically minimizes intra-class variance assuming a bimodal
       histogram. Effective under uniform illumination, but fails when lighting gradients
       create non-bimodal distributions across the document.
    3. Adaptive Gaussian Thresholding: Operates locally, maintaining consistent stroke width
       even under sharp shadow lines.

    Returns:
        Dict[str, np.ndarray]: Mapping of threshold names to inverted binary images.
    """
    cfg = config or PREPROCESS_CONFIG
    blurred = apply_gaussian_blur(gray, cfg)

    # 1. Plain Global Thresholding
    t_val = cfg.get("global_thresh_val", 127)
    _, global_thresh = cv2.threshold(blurred, t_val, 255, cv2.THRESH_BINARY_INV)

    # 2. Otsu Thresholding (computed on grayscale to preserve crisp, thin stroke width)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
    k_open = cfg.get("morph_kernel_open", (2, 2))
    k_close = cfg.get("morph_kernel_close", (3, 3))

    # Structuring elements
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, k_close)

    # Step 1: Remove isolated foreground noise
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_open)

    # Step 2: Bridge hairline gaps in strokes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    return closed


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

    # 4. Grayscale & Noise reduction
    gray = to_grayscale(working_bgr)
    blurred = apply_gaussian_blur(gray, cfg)

    # 5. Threshold comparisons (Global vs Otsu vs Adaptive)
    thresh_comp = compare_thresholds(gray, cfg)
    binary = thresh_comp["adaptive"]

    # 6. Morphological filtering (remove speckles & join strokes)
    cleaned = apply_morphology(binary, cfg)

    # 7. Orientation deskew
    deskewed, skew_angle = deskew_image(cleaned, cfg)

    return {
        "original": original,
        "resized": resized,
        "perspective_warped": warped if paper_detected else None,
        "paper_detected": paper_detected,
        "gray": gray,
        "blurred": blurred,
        "threshold_comparison": thresh_comp,
        "binary": binary,
        "cleaned": cleaned,
        "deskewed": deskewed,
        "skew_angle": skew_angle
    }
