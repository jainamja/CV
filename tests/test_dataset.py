"""
=============================================================================
Unit Tests for MathVision Dataset Module
=============================================================================
"""

import numpy as np
import pytest
import hashlib
from src.dataset import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    generate_synthetic_symbol,
    augment_batch,
    SYSTEM_FONTS,
    load_dataset,
    check_split_leakage,
)


def test_classes_integrity():
    """Verify class mapping consistency."""
    assert len(CLASSES) == 18
    assert len(CLASS_TO_IDX) == 18
    assert len(IDX_TO_CLASS) == 18

    for i, name in enumerate(CLASSES):
        assert CLASS_TO_IDX[name] == i
        assert IDX_TO_CLASS[i] == name


def test_generate_synthetic_symbol():
    """Verify synthetic symbol generator returns normalized 45x45 array."""
    font = SYSTEM_FONTS[0] if SYSTEM_FONTS else "default"
    glyph = generate_synthetic_symbol("x", font, target_size=(45, 45))

    assert glyph.shape == (45, 45)
    assert glyph.dtype == np.float32
    assert glyph.min() >= 0.0
    assert glyph.max() <= 1.0


def test_augment_batch():
    """Verify data augmentation maintains tensor shapes and values."""
    batch = np.random.uniform(0.0, 1.0, (8, 45, 45, 1)).astype(np.float32)
    aug = augment_batch(batch)

    assert aug.shape == (8, 45, 45, 1)
    assert aug.min() >= 0.0
    assert aug.max() <= 1.0


def test_train_val_test_isolation():
    """
    Asserts zero exact hash overlap and zero near-duplicate overlap (MSE < 0.001)
    across all train, validation, and test splits.
    """
    ds = load_dataset()
    X_train, y_train = ds["X_train"], ds["y_train"]
    X_val, y_val = ds["X_val"], ds["y_val"]
    X_test, y_test = ds["X_test"], ds["y_test"]

    # 1. Exact SHA-256 hash sets
    def to_hashes(arr):
        return {
            hashlib.sha256((x[..., 0] * 255.0).astype(np.uint8).tobytes()).hexdigest()
            for x in arr
        }

    h_train = to_hashes(X_train)
    h_val = to_hashes(X_val)
    h_test = to_hashes(X_test)

    # Overlaps must be exactly zero
    assert len(h_train & h_val) == 0, f"Train/Val exact hash leakage detected: {len(h_train & h_val)} images"
    assert len(h_train & h_test) == 0, f"Train/Test exact hash leakage detected: {len(h_train & h_test)} images"
    assert len(h_val & h_test) == 0, f"Val/Test exact hash leakage detected: {len(h_val & h_test)} images"

    # 2. Comprehensive leakage check (exact + near-duplicate MSE < 0.001)
    audit = check_split_leakage(X_train, X_val, X_test, mse_thresh=0.001)
    assert not audit["leakage_detected"], f"Data leakage detected by check_split_leakage: {audit}"
    assert audit["exact_train_val_overlap"] == 0
    assert audit["exact_train_test_overlap"] == 0
    assert audit["exact_val_test_overlap"] == 0
    assert audit["near_val_train_overlap"] == 0
    assert audit["near_test_train_overlap"] == 0
    assert audit["min_val_train_mse"] >= 0.001
    assert audit["min_test_train_mse"] >= 0.001


def test_dataset_real_source_and_classes():
    """Verify dataset loaded is strictly from real Kaggle data across all 18 classes."""
    ds = load_dataset()
    assert ds["source"] == "kaggle_real"
    assert len(ds["class_names"]) == 18
    assert len(ds["class_weights"]) == 18

    # All 18 classes must have samples in train, val, and test
    assert len(np.unique(ds["y_train"])) == 18
    assert len(np.unique(ds["y_val"])) == 18
    assert len(np.unique(ds["y_test"])) == 18
