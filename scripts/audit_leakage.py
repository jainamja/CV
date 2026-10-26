"""
Audit script to inspect train, validation, and test splits for exact or near duplicates.
"""
import sys
import os
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.dataset import load_dataset

def audit():
    ds = load_dataset()
    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_val = ds["X_val"]
    y_val = ds["y_val"]
    X_test = ds["X_test"]
    y_test = ds["y_test"]

    print(f"\n--- DATA AUDIT REPORT ---")
    print(f"Data Source: {ds['source']}")
    print(f"Total Raw / Base samples: {len(X_train) + len(X_val) + len(X_test)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 1. Exact MD5 Hash check
    def hash_array(arr):
        # Convert to float32 bytes rounded to 4 decimals or uint8
        arr_uint8 = np.uint8(np.round(arr * 255.0))
        return hashlib.sha256(arr_uint8.tobytes()).hexdigest()

    train_hashes = {hash_array(img): i for i, img in enumerate(X_train)}
    val_hashes = {hash_array(img): i for i, img in enumerate(X_val)}
    test_hashes = {hash_array(img): i for i, img in enumerate(X_test)}

    train_val_overlap = set(train_hashes.keys()).intersection(set(val_hashes.keys()))
    train_test_overlap = set(train_hashes.keys()).intersection(set(test_hashes.keys()))
    val_test_overlap = set(val_hashes.keys()).intersection(set(test_hashes.keys()))

    print(f"\n--- EXACT DUPLICATE HASH ANALYSIS ---")
    print(f"Unique Train Hashes: {len(train_hashes)} / {len(X_train)}")
    print(f"Unique Val Hashes:   {len(val_hashes)} / {len(X_val)}")
    print(f"Unique Test Hashes:  {len(test_hashes)} / {len(X_test)}")
    print(f"Exact Train-Validation Overlap: {len(train_val_overlap)}")
    print(f"Exact Train-Test Overlap:       {len(train_test_overlap)}")
    print(f"Exact Val-Test Overlap:         {len(val_test_overlap)}")

    # 2. Near-duplicate / Mean Squared Error (MSE) check
    # Check if any test image has MSE < 1e-4 with any train image
    print(f"\n--- NEAR-DUPLICATE (MSE < 0.001) ANALYSIS ---")
    near_dup_count = 0
    # Sample a subset of test images to check against train
    for i in range(min(100, len(X_test))):
        test_img = X_test[i]
        diffs = np.mean((X_train - test_img) ** 2, axis=(1, 2, 3))
        min_mse = np.min(diffs)
        if min_mse < 0.001:
            near_dup_count += 1

    print(f"Near-duplicates found in sampled 100 test images (MSE < 0.001): {near_dup_count} / 100")

if __name__ == "__main__":
    audit()
