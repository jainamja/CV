"""
MathVision Phase 3.2: Permanent Adversarial Regression Test Suite
Validates that topological disambiguation reliably distinguishes division from plus
and that genuine, faint, tilted, or broken plus signs are never converted to division.
"""

import pytest
import numpy as np
import cv2
from tensorflow import keras

from experiments.phase3_1.topological_disambiguation import (
    check_division_topology,
    apply_topological_plus_div_disambiguation
)
from experiments.phase3_1.adversarial_evaluation import generate_adversarial_suite

@pytest.fixture(scope="module")
def v4_model():
    return keras.models.load_model("models/cnn_experimental_v4.keras")

@pytest.fixture(scope="module")
def adversarial_suite():
    return generate_adversarial_suite()


def test_adversarial_genuine_plus_never_converted_to_division(v4_model, adversarial_suite):
    """Zero genuine plus signs in the adversarial suite may be converted to division."""
    for name, cat, crop in adversarial_suite:
        if cat != "plus":
            continue
        
        # Test full disambiguation logic
        inp = np.expand_dims(np.expand_dims(crop, axis=0), axis=-1)
        probs = v4_model.predict(inp, verbose=0)
        sym = "+"
        conf = float(probs[0, 10])  # '+' index
        
        top3_indices = np.argsort(probs[0])[-3:][::-1]
        from src.dataset import CLASSES
        top3 = [(CLASSES[i], float(probs[0, i])) for i in top3_indices]
        
        diags = [{"top3": top3}]
        new_syms, new_confs, new_diags, n_corr = apply_topological_plus_div_disambiguation(
            predicted_symbols=[sym],
            confidences=[conf],
            symbol_crops=[crop],
            symbol_diagnostics=diags,
            probs=probs
        )
        
        assert new_syms[0] == "+", (
            f"Adversarial case '{name}' was falsely converted from '+' to '{new_syms[0]}'!"
        )
        assert n_corr == 0, f"False correction registered on '{name}'"


def test_adversarial_distractors_and_degenerates_fail_safely(v4_model, adversarial_suite):
    """Distractors (equals, times, minus, digits, noise) and degenerates must never trigger division."""
    for name, cat, crop in adversarial_suite:
        if cat not in ("distractor", "degenerate"):
            continue
        
        is_div, reason, _ = check_division_topology(crop)
        # Even if topology triggered on a freak noise pattern, gating must protect it
        inp = np.expand_dims(np.expand_dims(crop, axis=0), axis=-1)
        probs = v4_model.predict(inp, verbose=0)
        from src.dataset import CLASSES
        raw_pred = CLASSES[np.argmax(probs[0])]
        top3_indices = np.argsort(probs[0])[-3:][::-1]
        top3 = [(CLASSES[i], float(probs[0, i])) for i in top3_indices]
        diags = [{"top3": top3}]
        
        new_syms, new_confs, new_diags, n_corr = apply_topological_plus_div_disambiguation(
            predicted_symbols=[raw_pred],
            confidences=[float(np.max(probs[0]))],
            symbol_crops=[crop],
            symbol_diagnostics=diags,
            probs=probs
        )
        
        assert new_syms[0] != "div" or raw_pred == "div", (
            f"Distractor/Degenerate '{name}' was falsely converted to 'div'!"
        )


def test_adversarial_elongated_division_dots_detected(v4_model, adversarial_suite):
    """Division with elongated vertical dots (Case B style) must confirm division topology."""
    case_b_style = next((crop for name, cat, crop in adversarial_suite if "Case B style" in name), None)
    assert case_b_style is not None
    is_div, reason, metrics = check_division_topology(case_b_style)
    assert is_div, f"Elongated division symbol failed topology check: {reason}"
    assert metrics["zeros_below"] >= 2
    assert metrics["num_components"] >= 2
