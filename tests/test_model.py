"""
=============================================================================
Unit Tests for MathVision Model Architecture & Feature Extraction
=============================================================================
"""

import os
import numpy as np
import pytest
from src.model import build_cnn_model, extract_hog_features, build_svm_baseline
from tensorflow import keras


def test_build_cnn_model_shapes():
    """Verify CNN builds with expected input and output tensor dimensions."""
    model = build_cnn_model(input_shape=(45, 45, 1), num_classes=18)
    assert model.input_shape == (None, 45, 45, 1)
    assert model.output_shape == (None, 18)


def test_extract_hog_features_dimension():
    """Verify HOG feature extraction produces 1D non-empty feature vector."""
    dummy_img = np.random.uniform(0.0, 1.0, (45, 45)).astype(np.float32)
    features = extract_hog_features(dummy_img)

    assert features.ndim == 1
    assert len(features) > 100
    assert not np.isnan(features).any()


def test_trained_cnn_prediction():
    """Verify saved CNN model predicts valid probability distributions."""
    model_path = os.path.join("models", "cnn_mathvision.keras")
    if not os.path.exists(model_path):
        pytest.skip("Trained model checkpoint not found.")

    model = keras.models.load_model(model_path)
    sample = np.random.uniform(0.0, 1.0, (1, 45, 45, 1)).astype(np.float32)
    probs = model.predict(sample, verbose=0)

    assert probs.shape == (1, 18)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
    assert np.all(probs >= 0.0)
