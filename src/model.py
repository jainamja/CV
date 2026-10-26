"""
=============================================================================
MathVision: Model Architectures (CNN & HOG+SVM Baseline)
=============================================================================

This module defines:
    1. A Deep Convolutional Neural Network (CNN) engineered specifically
       for 45x45 grayscale handwritten math symbol classification.
    2. A classical Machine Learning baseline: Histogram of Oriented Gradients
       (HOG) feature extraction coupled with a Support Vector Machine (SVM)
       with RBF kernel.

Viva Explanation:
Why compare CNN vs HOG+SVM?
In academic computer vision, establishing a classical feature engineering baseline
(HOG + SVM) is the gold standard for validating deep learning utility.
- HOG captures localized edge direction distributions (first-order spatial gradients).
- CNN hierarchically learns optimal low-level features (edges, curves), mid-level
  topological compositions (junctions, loops, intersections), and high-level class
  representations directly from pixel matrices via backpropagation.
=============================================================================
"""

from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skimage.feature import hog
from sklearn.svm import SVC


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (45, 45, 1),
    num_classes: int = 18,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Constructs a 3-block Deep CNN with Batch Normalization, Spatial Max-Pooling,
    and Dropout Regularization.

    Architecture Details:
        Block 1 (Receptive Field ~7x7):
            - Conv2D(32, 3x3, same) + BatchNorm + ReLU
            - Conv2D(32, 3x3) + BatchNorm + ReLU
            - MaxPool2D(2x2) -> Reduces (45x45) to (21x21)
            - Dropout(0.25) -> Reduces co-adaptation of feature detectors

        Block 2:
            - Conv2D(64, 3x3, same) + BatchNorm + ReLU
            - Conv2D(64, 3x3) + BatchNorm + ReLU
            - MaxPool2D(2x2) -> Reduces (21x21) to (9x9)
            - Dropout(0.25)

        Block 3:
            - Conv2D(128, 3x3) + BatchNorm + ReLU -> (7x7)
            - MaxPool2D(2x2) -> (3x3)
            - Dropout(0.30)

        Dense Head:
            - Flatten -> 3 * 3 * 128 = 1152 features
            - Dense(128) + BatchNorm + ReLU
            - Dropout(0.50)
            - Dense(num_classes, softmax) -> Probability distribution over 18 classes

    Returns:
        keras.Model: Compiled Keras Model ready for training.
    """
    model = keras.Sequential([
        # Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"),
        layers.BatchNormalization(name="bn1_1"),
        layers.Conv2D(32, (3, 3), activation="relu", name="conv1_2"),
        layers.BatchNormalization(name="bn1_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Dropout(0.25, name="drop1"),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
        layers.BatchNormalization(name="bn2_1"),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv2_2"),
        layers.BatchNormalization(name="bn2_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Dropout(0.25, name="drop2"),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", name="conv3_1"),
        layers.BatchNormalization(name="bn3_1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),
        layers.Dropout(0.30, name="drop3"),

        # Dense Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.BatchNormalization(name="bn_dense"),
        layers.Dropout(0.50, name="drop_dense"),
        layers.Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name="MathVision_CNN")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def extract_hog_features(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (5, 5),
    cells_per_block: Tuple[int, int] = (2, 2)
) -> np.ndarray:
    """
    Extracts Histogram of Oriented Gradients (HOG) features from a 45x45 symbol.

    Viva Explanation:
    HOG counts occurrences of gradient orientation in localized portions of an image.
    For handwritten glyphs, orientation histograms describe stroke directions (e.g.
    vertical strokes of '1', horizontal strokes of '-', diagonal intersection of 'x')
    invariance to small translations and lighting changes.

    Args:
        image: 2D array (45, 45) or 3D array (45, 45, 1) in [0.0, 1.0].

    Returns:
        np.ndarray: 1D HOG feature vector.
    """
    img_2d = image.squeeze()
    features = hog(
        img_2d,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=False,
        transform_sqrt=True
    )
    return features


def extract_hog_dataset(X: np.ndarray) -> np.ndarray:
    """Extracts HOG feature vectors for an entire dataset split."""
    return np.array([extract_hog_features(img) for img in X], dtype=np.float32)


def build_svm_baseline(c_val: float = 10.0, class_weight: str = "balanced", random_state: int = 42, probability: bool = False) -> SVC:
    """
    Constructs a Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel.

    Viva Explanation:
    The RBF kernel projects the non-linear HOG feature space into a higher-dimensional
    Hilbert space where mathematical symbols become linearly separable.
    class_weight="balanced" adjusts margins inversely proportional to class frequencies.
    """
    return SVC(
        C=c_val,
        kernel="rbf",
        gamma="scale",
        probability=probability,
        class_weight=class_weight,
        random_state=random_state
    )


if __name__ == "__main__":
    cnn = build_cnn_model()
    cnn.summary()
    sample_hog = extract_hog_features(np.zeros((45, 45), dtype=np.float32))
    print(f"\nHOG Feature Vector Dimension: {sample_hog.shape}")
