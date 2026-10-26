"""
=============================================================================
MathVision: Dataset Ingestion, Deduplication, Splitting & Augmentation
=============================================================================

This module manages the training, validation, and test datasets for the 18
handwritten mathematical symbol classes using real Kaggle handwritten symbols.

Target Classes (18 total):
    - Digits (10): '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    - Operators (5): '+', '-', 'times' (×), 'div' (÷), '='
    - Parentheses (2): '(', ')'
    - Algebraic Variable (1): 'x'

Key Capabilities:
    1. Real Kaggle Dataset Loader: Ingests raw symbol folders from data/raw/
       and maps folder names (e.g., 'plus', 'times', 'X') to canonical labels.
       Raises FileNotFoundError if data/raw/ is missing (no silent fallback).
    2. High-Performance Deduplication: Multi-threaded image reading, exact
       SHA-256 deduplication, and ultra-fast vectorized near-duplicate
       rejection (MSE < 0.001) BEFORE splitting.
    3. Class Imbalance Mitigation: Caps overrepresented classes to
       MAX_SAMPLES_PER_CLASS (default 1500) and computes balanced class weights.
    4. Stratified Train / Val / Test Split: 70% train, 15% validation, 15% test
       using scikit-learn's stratified splitting with a fixed seed (42).
    5. Pure Evaluation Sets: Validation and Test splits contain 100% real,
       unaugmented handwritten images. Zero synthetic data in val/test.
    6. Training Augmentation: On-the-fly affine perturbations (rotation,
       translation, scale) applied strictly to X_train.
    7. Processed Cache: Saves the verified deduplicated split to
       data/processed/split_data.npz for sub-second loading in subsequent runs.
=============================================================================
"""

import os
import random
import hashlib
from typing import Tuple, Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# Configuration Flags & Constants
USE_SYNTHETIC_TRAIN: bool = False      # If True, appends synthetic data ONLY to X_train (never val/test)
MAX_SAMPLES_PER_CLASS: int = 1500     # Cap overrepresented classes to mitigate severe imbalance
RANDOM_STATE: int = 42                 # Seed for reproducibility
NEAR_DUP_MSE_THRESH: float = 0.001    # Mean squared error threshold for near-duplicate rejection
TARGET_SIZE: Tuple[int, int] = (45, 45)
PROCESSED_FILE: str = r"data/processed/split_data.npz"

# Canonical class index mapping (18 classes)
CLASSES: List[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "times", "div", "=", "(", ")", "x"
]

CLASS_TO_IDX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(CLASSES)}

# Visual glyph representations for UI / plotting
LABEL_TO_DISPLAY: Dict[str, str] = {
    "times": "×",
    "div": "÷",
}

# Aliases commonly used in Kaggle folders (e.g., CROHME / Kaggle Math Symbols)
FOLDER_ALIASES: Dict[str, str] = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "+": "+", "add": "+", "plus": "+",
    "-": "-", "sub": "-", "minus": "-", "subtraction": "-",
    "times": "times", "mul": "times", "multiply": "times", "x_cross": "times", "*": "times",
    "div": "div", "divide": "div", "division": "div", "/": "div", "slash": "div",
    "=": "=", "eq": "=", "equal": "=", "equals": "=",
    "(": "(", "lparen": "(", "left_paren": "(", "[": "(",
    ")": ")", "rparen": ")", "right_paren": ")", "]": ")",
    "x": "x", "X": "x", "var_x": "x", "letter_x": "x"
}

# Candidate handwriting fonts available on Windows for optional synthetic train data
SYSTEM_FONTS = [
    r"C:\Windows\Fonts\Inkfree.ttf",
    r"C:\Windows\Fonts\segoepr.ttf",
    r"C:\Windows\Fonts\segoeprb.ttf",
    r"C:\Windows\Fonts\comic.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]


def get_kaggle_download_instructions() -> str:
    """Returns step-by-step instructions for downloading the Kaggle dataset."""
    return """
=============================================================================
MANUAL KAGGLE DATASET DOWNLOAD INSTRUCTIONS
=============================================================================
To run MathVision with real handwritten math symbols:

1. Visit Kaggle:
     https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
     (or https://www.kaggle.com/datasets/sagarkachauhan/handwritten-math-symbols)
2. Download the archive (e.g., archive.zip or data.rar).
3. Extract the symbol folders directly into:
     data/raw/
4. Ensure each class has its own folder inside data/raw/, e.g.:
     data/raw/0/
     data/raw/1/
     ...
     data/raw/9/
     data/raw/+/
     data/raw/-/
     data/raw/times/  (or mul, *)
     data/raw/div/
     data/raw/=/
     data/raw/(/
     data/raw/)/
     data/raw/x/      (or X)
=============================================================================
"""


def _read_and_prep_image(fpath: str, target_size: Tuple[int, int] = TARGET_SIZE) -> Optional[np.ndarray]:
    """Helper to read, invert, resize, and normalize a single image."""
    try:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if np.mean(img) > 127:
            img = 255 - img
        if img.shape != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0
    except Exception:
        return None


def deduplicate_class_samples(
    images: List[np.ndarray],
    mse_thresh: float = NEAR_DUP_MSE_THRESH
) -> Tuple[List[np.ndarray], int, int]:
    """
    Removes exact SHA-256 duplicates and near-duplicate images (MSE < mse_thresh)
    within a single class using vectorized matrix distance calculation.

    Returns:
        (unique_images, exact_dupes_dropped, near_dupes_dropped)
    """
    if not images:
        return [], 0, 0

    # 1. Exact SHA-256 deduplication
    exact_hashes: Set[str] = set()
    exact_unique: List[np.ndarray] = []
    exact_dupes = 0

    for norm_img in images:
        uint8_bytes = (norm_img * 255.0).astype(np.uint8).tobytes()
        sha = hashlib.sha256(uint8_bytes).hexdigest()
        if sha in exact_hashes:
            exact_dupes += 1
            continue
        exact_hashes.add(sha)
        exact_unique.append(norm_img)

    if len(exact_unique) <= 1:
        return exact_unique, exact_dupes, 0

    # 2. Vectorized near-duplicate filtering via 16x16 downsample
    arr = np.array(exact_unique, dtype=np.float32)
    thumbs = np.array([cv2.resize(x, (16, 16), interpolation=cv2.INTER_AREA) for x in arr])
    flat = thumbs.reshape(len(thumbs), -1)  # (N, 256)

    # Vectorized pairwise MSE: (A - B)^2 = A^2 - 2AB + B^2
    norms = np.sum(flat ** 2, axis=1, keepdims=True)
    dists = (norms + norms.T - 2 * np.dot(flat, flat.T)) / 256.0
    dists = np.clip(dists, 0.0, None)
    np.fill_diagonal(dists, 1.0)

    kept_indices: List[int] = []
    discarded_indices: Set[int] = set()

    for i in range(len(arr)):
        if i in discarded_indices:
            continue
        kept_indices.append(i)
        near_j = np.where(dists[i, i+1:] < mse_thresh)[0] + (i + 1)
        discarded_indices.update(near_j)

    near_dupes = len(discarded_indices)
    cleaned_images = [exact_unique[i] for i in kept_indices]
    return cleaned_images, exact_dupes, near_dupes


def load_raw_kaggle_dataset(
    raw_dir: str = "data/raw",
    target_size: Tuple[int, int] = TARGET_SIZE,
    max_per_class: int = MAX_SAMPLES_PER_CLASS,
    random_seed: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, int]]]:
    """
    Scans data/raw/ for symbol folders, normalizes images to 45x45 float32,
    deduplicates (exact + near) within each class using multi-threading and vectorization,
    caps oversized classes, and returns (X, y, dedup_stats).

    Raises:
        FileNotFoundError: If data/raw/ does not exist or has no valid symbol images.
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(
            f"Directory '{raw_dir}' does not exist.\n" + get_kaggle_download_instructions()
        )

    subdirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if not subdirs:
        raise FileNotFoundError(
            f"No class subdirectories found in '{raw_dir}'.\n" + get_kaggle_download_instructions()
        )

    random.seed(random_seed)
    np.random.seed(random_seed)

    all_images: List[np.ndarray] = []
    all_labels: List[int] = []
    dedup_stats: Dict[str, Dict[str, int]] = {}

    print(f"[Dataset] Ingesting real Kaggle dataset from '{raw_dir}' (cap={max_per_class}/class)...")

    for subdir in sorted(subdirs):
        canonical_label = FOLDER_ALIASES.get(subdir.strip(), None)
        if canonical_label is None:
            canonical_label = FOLDER_ALIASES.get(subdir.strip().lower(), None)

        if canonical_label is None or canonical_label not in CLASS_TO_IDX:
            continue

        class_idx = CLASS_TO_IDX[canonical_label]
        class_folder = os.path.join(raw_dir, subdir)
        all_class_files = [
            os.path.join(class_folder, f)
            for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if not all_class_files:
            continue

        # Shuffle deterministically
        random.shuffle(all_class_files)

        # Sample up to max_per_class * 2.2 files to read in parallel
        sample_limit = min(len(all_class_files), int(max_per_class * 2.2))
        files_to_read = all_class_files[:sample_limit]

        with ThreadPoolExecutor(max_workers=8) as executor:
            read_results = list(executor.map(lambda p: _read_and_prep_image(p, target_size), files_to_read))

        valid_raw = [x for x in read_results if x is not None]

        # Deduplicate (exact + near)
        unique_samples, exact_dupes, near_dupes = deduplicate_class_samples(valid_raw)

        # Cap to max_per_class
        if len(unique_samples) > max_per_class:
            selected_samples = unique_samples[:max_per_class]
        else:
            selected_samples = unique_samples

        dedup_stats[canonical_label] = {
            "total_files_in_folder": len(all_class_files),
            "files_scanned": len(files_to_read),
            "exact_dupes_dropped": exact_dupes,
            "near_dupes_dropped": near_dupes,
            "final_unique_used": len(selected_samples)
        }

        print(f"  [{canonical_label:>5}] Unique used: {len(selected_samples):>4} | Exact Dupes: {exact_dupes:>3} | Near Dupes: {near_dupes:>3} (pool: {len(all_class_files)})")

        for img in selected_samples:
            all_images.append(img)
            all_labels.append(class_idx)

    if not all_images:
        raise FileNotFoundError(
            f"No valid images could be loaded from '{raw_dir}'.\n" + get_kaggle_download_instructions()
        )

    X = np.array(all_images, dtype=np.float32)[..., np.newaxis]
    y = np.array(all_labels, dtype=np.int32)
    return X, y, dedup_stats


def generate_synthetic_symbol(
    char: str,
    font_path: str,
    target_size: Tuple[int, int] = TARGET_SIZE,
    rotation_deg: float = 0.0,
    thickness_dilate: int = 0
) -> np.ndarray:
    """
    Generates a single synthetic handwritten glyph on a target canvas.
    Used ONLY when USE_SYNTHETIC_TRAIN is explicitly enabled, and strictly for X_train.
    """
    canvas_size = 80
    pil_img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(pil_img)

    try:
        font_size = random.randint(40, 52)
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    display_char = char
    if char == "times":
        display_char = "×"
    elif char == "div":
        display_char = "÷"

    bbox = draw.textbbox((0, 0), display_char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (canvas_size - tw) // 2 - bbox[0]
    ty = (canvas_size - th) // 2 - bbox[1]

    draw.text((tx, ty), display_char, fill=255, font=font)
    glyph_arr = np.array(pil_img, dtype=np.uint8)

    if thickness_dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness_dilate, thickness_dilate))
        glyph_arr = cv2.dilate(glyph_arr, kernel)

    if abs(rotation_deg) > 0.5:
        center = (canvas_size // 2, canvas_size // 2)
        M = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        glyph_arr = cv2.warpAffine(glyph_arr, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR, borderValue=0)

    pts = cv2.findNonZero(glyph_arr)
    if pts is None:
        return np.zeros(target_size, dtype=np.float32)

    x, y, w, h = cv2.boundingRect(pts)
    crop = glyph_arr[y:y+h, x:x+w]

    max_d = max(w, h)
    margin = int(round(max_d * 0.20))
    sq_size = max_d + 2 * margin
    sq_canvas = np.zeros((sq_size, sq_size), dtype=np.uint8)

    ox = margin + (max_d - w) // 2
    oy = margin + (max_d - h) // 2
    sq_canvas[oy:oy+h, ox:ox+w] = crop

    resized = cv2.resize(sq_canvas, target_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def generate_synthetic_train_boost(
    samples_per_class: int = 100,
    target_size: Tuple[int, int] = TARGET_SIZE,
    seed: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic glyphs strictly for augmenting X_train."""
    random.seed(seed)
    np.random.seed(seed)

    available_fonts = [f for f in SYSTEM_FONTS if os.path.exists(f)]
    if not available_fonts:
        available_fonts = ["default"]

    images_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for class_idx, class_name in enumerate(CLASSES):
        for _ in range(samples_per_class):
            font = random.choice(available_fonts)
            rot = random.uniform(-14.0, 14.0)
            dilate = random.choice([0, 0, 1, 2])

            glyph = generate_synthetic_symbol(
                class_name, font, target_size=target_size,
                rotation_deg=rot, thickness_dilate=dilate
            )
            noise = np.random.normal(0, 0.02, target_size).astype(np.float32)
            glyph = np.clip(glyph + noise, 0.0, 1.0)

            images_list.append(glyph)
            labels_list.append(class_idx)

    X_syn = np.array(images_list, dtype=np.float32)[..., np.newaxis]
    y_syn = np.array(labels_list, dtype=np.int32)
    return X_syn, y_syn


def augment_batch(images: np.ndarray) -> np.ndarray:
    """
    Applies small random affine transformations on-the-fly to a batch of 45x45 images.
    - Rotation: [-10, +10] degrees
    - Translation: [-2, +2] pixels
    - Scaling: [0.90, 1.10]
    Applied ONLY to X_train batches during training.
    """
    augmented = np.empty_like(images)
    h, w = images.shape[1:3]
    center = (w / 2.0, h / 2.0)

    for i in range(len(images)):
        img = images[i, :, :, 0]
        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.90, 1.10)
        tx = random.uniform(-2.0, 2.0)
        ty = random.uniform(-2.0, 2.0)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        aug = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        augmented[i, :, :, 0] = np.clip(aug, 0.0, 1.0)

    return augmented


def check_split_leakage(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    mse_thresh: float = NEAR_DUP_MSE_THRESH
) -> Dict[str, Any]:
    """
    Thoroughly verifies that no exact duplicate or near-duplicate (MSE < mse_thresh on 45x45)
    exists across train, validation, and test splits.
    """
    def to_hash_set(arr: np.ndarray) -> Set[str]:
        return {
            hashlib.sha256((x[..., 0] * 255.0).astype(np.uint8).tobytes()).hexdigest()
            for x in arr
        }

    h_train = to_hash_set(X_train)
    h_val = to_hash_set(X_val)
    h_test = to_hash_set(X_test)

    exact_tr_val = len(h_train & h_val)
    exact_tr_te = len(h_train & h_test)
    exact_va_te = len(h_val & h_test)

    # Vectorized full 45x45 MSE calculation across splits
    flat_tr = X_train[..., 0].reshape(len(X_train), -1)  # (N_tr, 2025)
    norms_tr = np.sum(flat_tr ** 2, axis=1, keepdims=True)

    # Check Val vs Train
    flat_va = X_val[..., 0].reshape(len(X_val), -1)      # (N_va, 2025)
    norms_va = np.sum(flat_va ** 2, axis=1, keepdims=True)
    dists_va_tr = (norms_va + norms_tr.T - 2 * np.dot(flat_va, flat_tr.T)) / 2025.0
    dists_va_tr = np.clip(dists_va_tr, 0.0, None)
    near_val_tr = int(np.sum(dists_va_tr < mse_thresh))

    # Check Test vs Train
    flat_te = X_test[..., 0].reshape(len(X_test), -1)    # (N_te, 2025)
    norms_te = np.sum(flat_te ** 2, axis=1, keepdims=True)
    dists_te_tr = (norms_te + norms_tr.T - 2 * np.dot(flat_te, flat_tr.T)) / 2025.0
    dists_te_tr = np.clip(dists_te_tr, 0.0, None)
    near_te_tr = int(np.sum(dists_te_tr < mse_thresh))

    min_va_mse = float(np.min(dists_va_tr)) if dists_va_tr.size > 0 else 1.0
    min_te_mse = float(np.min(dists_te_tr)) if dists_te_tr.size > 0 else 1.0

    leakage_detected = any([
        exact_tr_val > 0, exact_tr_te > 0, exact_va_te > 0,
        near_val_tr > 0, near_te_tr > 0
    ])

    return {
        "leakage_detected": leakage_detected,
        "exact_train_val_overlap": exact_tr_val,
        "exact_train_test_overlap": exact_tr_te,
        "exact_val_test_overlap": exact_va_te,
        "near_val_train_overlap": near_val_tr,
        "near_test_train_overlap": near_te_tr,
        "min_val_train_mse": min_va_mse,
        "min_test_train_mse": min_te_mse,
    }


def load_dataset(
    raw_dir: str = "data/raw",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
    use_synthetic_train: bool = USE_SYNTHETIC_TRAIN,
    max_per_class: int = MAX_SAMPLES_PER_CLASS,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Primary dataset loading function:
    1. Ingests real Kaggle images from data/raw/ (or loads from data/processed/split_data.npz if present)
    2. Deduplicates exact and near-duplicates BEFORE splitting
    3. Caps oversized classes to max_per_class
    4. Performs stratified 70/15/15 train/val/test split
    5. Computes balanced class weights
    6. Validates zero leakage across splits

    Returns:
        Dict[str, Any] with train, val, and test splits and metadata.
    """
    # Check if precomputed split already exists and rebuild is not forced
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    if os.path.exists(PROCESSED_FILE) and not force_rebuild and not use_synthetic_train:
        print(f"[Dataset] Loading pre-verified split from '{PROCESSED_FILE}'...")
        cached = np.load(PROCESSED_FILE, allow_pickle=True)
        X_train, y_train = cached["X_train"], cached["y_train"]
        X_val, y_val = cached["X_val"], cached["y_val"]
        X_test, y_test = cached["X_test"], cached["y_test"]

        unique_tr, counts_tr = np.unique(y_train, return_counts=True)
        class_counts = {IDX_TO_CLASS[u]: int(c) for u, c in zip(unique_tr, counts_tr)}
        weights = compute_class_weight("balanced", classes=unique_tr, y=y_train)
        class_weights = {int(cls_idx): float(w) for cls_idx, w in zip(unique_tr, weights)}

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "class_names": CLASSES,
            "class_weights": class_weights,
            "source": "kaggle_real",
            "class_counts": class_counts,
            "is_imbalanced": (max(counts_tr) / float(min(counts_tr))) > 1.5,
            "dedup_stats": cached["dedup_stats"].item() if "dedup_stats" in cached else {},
            "leakage_audit": cached["leakage_audit"].item() if "leakage_audit" in cached else {"leakage_detected": False}
        }

    # 1. Ingest, deduplicate, and cap real Kaggle symbols
    X, y, dedup_stats = load_raw_kaggle_dataset(
        raw_dir=raw_dir,
        max_per_class=max_per_class,
        random_seed=random_state
    )

    print(f"[Dataset] Total unique images collected: {len(X)} across {len(np.unique(y))} classes.")

    # 2. Stratified Train / Temp Split (70% train, 30% temp)
    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        stratify=y,
        random_state=random_state
    )

    # 3. Stratified Val / Test Split (50/50 of temp -> 15% val, 15% test of total)
    relative_val_size = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - relative_val_size),
        stratify=y_temp,
        random_state=random_state
    )

    # 4. Optional synthetic augmentation strictly on training split
    if use_synthetic_train:
        print("[Dataset] USE_SYNTHETIC_TRAIN is True: Generating synthetic glyphs for X_train ONLY...")
        X_syn_tr, y_syn_tr = generate_synthetic_train_boost(samples_per_class=100, seed=random_state)
        X_train = np.concatenate([X_train, X_syn_tr], axis=0)
        y_train = np.concatenate([y_train, y_syn_tr], axis=0)
        shuff_idx = np.arange(len(y_train))
        np.random.seed(random_state)
        np.random.shuffle(shuff_idx)
        X_train = X_train[shuff_idx]
        y_train = y_train[shuff_idx]

    # 5. Verify Zero Leakage Across Splits
    leakage_audit = check_split_leakage(X_train, X_val, X_test)
    if leakage_audit["leakage_detected"]:
        raise ValueError(f"DATA LEAKAGE DETECTED across splits! Audit details: {leakage_audit}")

    # 6. Compute Balanced Class Weights
    unique_tr, counts_tr = np.unique(y_train, return_counts=True)
    class_counts = {IDX_TO_CLASS[u]: int(c) for u, c in zip(unique_tr, counts_tr)}
    weights = compute_class_weight("balanced", classes=unique_tr, y=y_train)
    class_weights = {int(cls_idx): float(w) for cls_idx, w in zip(unique_tr, weights)}

    min_c = min(counts_tr)
    max_c = max(counts_tr)
    is_imbalanced = (max_c / float(min_c)) > 1.5 if min_c > 0 else False

    # 7. Save Verified Clean Split to data/processed/
    np.savez_compressed(
        PROCESSED_FILE,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        dedup_stats=dedup_stats,
        leakage_audit=leakage_audit
    )
    print(f"[Dataset] Saved verified clean split to '{PROCESSED_FILE}'.")

    print(f"[Dataset] Split Summary: Train={len(X_train)} (70%), Val={len(X_val)} (15%), Test={len(X_test)} (15%)")
    print(f"[Dataset] Zero Leakage Confirmed: {not leakage_audit['leakage_detected']}")
    print(f"[Dataset] Class Imbalance Flag: {is_imbalanced} (Min train: {min_c}, Max train: {max_c})")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": CLASSES,
        "class_weights": class_weights,
        "source": "kaggle_real",
        "class_counts": class_counts,
        "is_imbalanced": is_imbalanced,
        "dedup_stats": dedup_stats,
        "leakage_audit": leakage_audit
    }


if __name__ == "__main__":
    ds = load_dataset(force_rebuild=True)
    print("\nSummary:")
    print("Classes:", ds["class_names"])
    print("Class Counts (Train):", ds["class_counts"])
    print("Leakage Audit:", ds["leakage_audit"])
