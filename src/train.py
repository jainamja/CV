"""
=============================================================================
MathVision: Model Training & Checkpointing Engine
=============================================================================

Trains:
    1. Deep CNN with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
    2. Baseline HOG + RBF SVM model.
    3. Saves best models to models/ and training curves to outputs/.
=============================================================================
"""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.dataset import load_dataset, augment_batch, CLASSES
from src.model import build_cnn_model, extract_hog_dataset, build_svm_baseline


def plot_training_curves(history: keras.callbacks.History, save_path: str = "outputs/training_curves.png") -> None:
    """Plots and saves loss and accuracy curves across training epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history.history["loss"], "o-", color="#EF4444", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history.history["val_loss"], "s--", color="#F59E0B", label="Validation Loss", linewidth=2)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy subplot
    ax2.plot(epochs, history.history["accuracy"], "o-", color="#10B981", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, history.history["val_accuracy"], "s--", color="#06B6D4", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Training] Saved convergence curves to: {os.path.abspath(save_path)}")


def train_models(
    epochs: int = 18,
    batch_size: int = 64,
    models_dir: str = "models",
    outputs_dir: str = "outputs"
) -> None:
    """Orchestrates dataset loading, CNN training, SVM fitting, and artifact saving."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print("=" * 70)
    print("[MathVision] Initiating Model Training Pipeline")
    print("=" * 70)

    # 1. Ingest Dataset
    ds = load_dataset()
    X_train, y_train = ds["X_train"], ds["y_train"]
    X_val, y_val = ds["X_val"], ds["y_val"]
    X_test, y_test = ds["X_test"], ds["y_test"]
    class_weights = ds["class_weights"]

    # Save test dataset split for independent evaluation
    np.savez_compressed(
        os.path.join(models_dir, "test_split.npz"),
        X_test=X_test, y_test=y_test
    )

    # Augment training split for enhanced generalization
    print(f"[Training] Augmenting {len(X_train)} training samples with rotation/scale/translation...")
    X_train_aug = augment_batch(X_train)
    X_train_combined = np.concatenate([X_train, X_train_aug], axis=0)
    y_train_combined = np.concatenate([y_train, y_train], axis=0)

    # Shuffle combined training set
    idx = np.arange(len(y_train_combined))
    np.random.shuffle(idx)
    X_train_combined = X_train_combined[idx]
    y_train_combined = y_train_combined[idx]

    # 2. Build & Train CNN Model
    print("\n" + "-" * 70)
    print("[CNN] Initializing Deep CNN Architecture...")
    print("-" * 70)
    cnn_model = build_cnn_model(input_shape=(45, 45, 1), num_classes=len(CLASSES))

    cnn_save_path = os.path.join(models_dir, "cnn_mathvision.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=cnn_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"[CNN] Training for up to {epochs} epochs (Batch Size: {batch_size})...")
    history = cnn_model.fit(
        X_train_combined,
        y_train_combined,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Save training convergence plot
    plot_training_curves(history, os.path.join(outputs_dir, "training_curves.png"))

    # Ensure best weights are saved
    cnn_model.save(cnn_save_path)
    print(f"[CNN] Best model successfully saved to: {os.path.abspath(cnn_save_path)}")

    # 3. Train Baseline: HOG + RBF SVM
    print("\n" + "-" * 70)
    print("[SVM Baseline] Extracting HOG Features & Fitting Support Vector Classifier...")
    print("-" * 70)

    print(f" -> Extracting HOG features from {len(X_train)} training samples...")
    X_train_hog = extract_hog_dataset(X_train)
    print(f" -> Extracting HOG features from {len(X_val)} validation samples...")
    X_val_hog = extract_hog_dataset(X_val)

    svm_model = build_svm_baseline(c_val=10.0, random_state=42)
    svm_model.fit(X_train_hog, y_train)

    val_acc_svm = svm_model.score(X_val_hog, y_val)
    print(f" -> Baseline HOG+SVM Validation Accuracy: {val_acc_svm * 100:.2f}%")

    svm_save_path = os.path.join(models_dir, "svm_baseline.joblib")
    joblib.dump(svm_model, svm_save_path)
    print(f"[SVM Baseline] Model successfully saved to: {os.path.abspath(svm_save_path)}")

    print("\n" + "=" * 70)
    print("[Training Complete] Both CNN and SVM Baseline are ready for evaluation!")
    print("=" * 70)


if __name__ == "__main__":
    train_models(epochs=15, batch_size=64)
