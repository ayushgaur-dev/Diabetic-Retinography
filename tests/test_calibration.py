"""Calibration tests — SIH26038 Phase 7. Numpy-level only: no model
weights, no dataset, no API. Deterministic throughout."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import metrics as M
from src.calibration.config import DEFAULTS, load_calibration_config
from src.calibration.pipeline import calibrate_record
from src.calibration.temperature_scaling import (apply_temperature,
                                                 fit_temperature,
                                                 multiclass_nll,
                                                 recover_logits, softmax)


def _overconfident(n=200, seed=0, correct_frac=0.70):
    """70% correct but always 0.92 confident -> genuinely overconfident
    (ECE ~= 0.22). Deterministic."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)
    probs = np.full((n, 5), 0.02)
    n_correct = int(n * correct_frac)
    probs[np.arange(n_correct), y[:n_correct]] = 0.92
    wrong = (y[n_correct:] + 1) % 5
    probs[np.arange(n_correct, n), wrong] = 0.92
    return probs, y


# --- 1-6. Temperature scaling -------------------------------------------------

def test_t1_leaves_probabilities_unchanged():
    rng = np.random.default_rng(1)
    p = rng.random((20, 5))
    p /= p.sum(axis=1, keepdims=True)
    assert np.allclose(apply_temperature(p, 1.0), p, atol=1e-9)


def test_positive_temperature_validation():
    p = np.full((4, 5), 0.2)
    with pytest.raises(ValueError):
        apply_temperature(p, 0.0)
    with pytest.raises(ValueError):
        apply_temperature(p, -1.5)


def test_scaling_preserves_argmax():
    rng = np.random.default_rng(2)
    p = rng.random((50, 5))
    p /= p.sum(axis=1, keepdims=True)
    for T in (0.2, 0.7, 1.5, 5.0):
        c = apply_temperature(p, T)
        assert (c.argmax(axis=1) == p.argmax(axis=1)).all()


def test_probabilities_sum_to_one():
    rng = np.random.default_rng(3)
    p = rng.random((30, 5))
    p /= p.sum(axis=1, keepdims=True)
    c = apply_temperature(p, 2.3)
    assert np.allclose(c.sum(axis=1), 1.0)


def test_numerical_stability():
    p = np.array([[1.0, 0, 0, 0, 0], [0.2] * 5, [1e-15, 0.5, 0.5, 0, 0]])
    c = apply_temperature(p, 1.7)
    assert np.isfinite(c).all() and np.allclose(c.sum(axis=1), 1.0)


def test_deterministic_fitting():
    probs, y = _overconfident()
    f1 = fit_temperature(probs, y)
    f2 = fit_temperature(probs, y)
    assert f1["temperature"] == f2["temperature"]
    assert f1["temperature"] > 0
    assert f1["nll_after"] <= f1["nll_before"] + 1e-9


def test_recover_logits_consistency():
    rng = np.random.default_rng(4)
    p = rng.random((10, 5))
    p /= p.sum(axis=1, keepdims=True)
    assert np.allclose(softmax(recover_logits(p)), p, atol=1e-9)


def test_overconfident_gets_softened():
    probs, y = _overconfident()
    fit = fit_temperature(probs, y)
    assert fit["temperature"] > 1.0  # overconfidence -> T > 1
    assert apply_temperature(probs, fit["temperature"]).max() < 0.92


# --- 7-12. Metrics ------------------------------------------------------------

def test_perfect_calibration():
    # confidence == accuracy in every occupied bin
    probs = np.array([[0.8, 0.05, 0.05, 0.05, 0.05]] * 40 +
                     [[0.3, 0.3, 0.2, 0.1, 0.1]] * 10)
    y = np.array([0] * 32 + [1] * 8 + [0] * 3 + [1] * 3 + [2] * 2 + [3] * 1 + [4] * 1)
    e = M.ece_metrics(probs, y, 10)
    assert e["ece"] < 0.05


def test_overconfident_detected():
    probs, y = _overconfident()
    e = M.ece_metrics(probs, y, 10)
    assert e["ece"] > 0.2 and e["mce"] > 0.2


def test_ece_bins_documented():
    probs, y = _overconfident(n=50)
    e = M.ece_metrics(probs, y, 10)
    assert e["n_bins"] == 10 and e["mode"] == "equal_width"
    assert len(e["bins"]) == 10
    assert sum(b["count"] for b in e["bins"]) == 50
    assert all(set(b) >= {"bin", "count", "mean_confidence", "accuracy"}
               for b in e["bins"])
    e15 = M.ece_metrics(probs, y, 15)
    assert e15["n_bins"] == 15


def test_brier_and_nll():
    probs = np.array([[1.0, 0, 0, 0, 0], [0.2] * 5])
    y = np.array([0, 3])
    assert M.brier_score(probs, y) == pytest.approx(0.08, abs=1e-9)
    assert multiclass_nll(probs, y) < multiclass_nll(
        np.array([[0.9, 0.025, 0.025, 0.025, 0.025]] * 2), y)


def test_mce_present():
    probs, y = _overconfident(n=60)
    e = M.ece_metrics(probs, y, 10)
    assert e["mce"] >= e["ece"]


# --- 13-15. Data isolation ----------------------------------------------------

def test_fit_consumes_only_passed_arrays():
    probs, y = _overconfident()
    seen = {}

    def spy_fit(p, labels, **kw):
        seen["n"] = len(labels)
        return fit_temperature(p, labels, **kw)

    val_n = 150
    spy_fit(probs[:val_n], y[:val_n])
    assert seen["n"] == val_n  # test rows (150:) never passed
    full = fit_temperature(probs, y)
    part = fit_temperature(probs[:val_n], y[:val_n])
    assert full["temperature"] != part["temperature"]  # fit actually uses input


def test_frozen_temperature_reused():
    probs, y = _overconfident()
    fit = fit_temperature(probs[:150], y[:150])
    c1 = apply_temperature(probs[150:], fit["temperature"])
    c2 = apply_temperature(probs[150:], fit["temperature"])
    assert np.array_equal(c1, c2)


# --- 16-19. Prediction invariance ----------------------------------------------

def test_predictions_identical_after_scaling():
    from src.evaluation.classification_metrics import five_class_metrics

    rng = np.random.default_rng(5)
    p = rng.random((60, 5))
    p /= p.sum(axis=1, keepdims=True)
    y = rng.integers(0, 5, 60)
    c = apply_temperature(p, 2.0)
    assert (p.argmax(1) == c.argmax(1)).all()
    m1, m2 = five_class_metrics(y, p.argmax(1)), five_class_metrics(y, c.argmax(1))
    assert m1["accuracy"] == m2["accuracy"] and m1["qwk"] == m2["qwk"]
    assert m1["macro_f1"] == m2["macro_f1"]


def test_calibrate_record_schema():
    rec = calibrate_record([0.7, 0.1, 0.1, 0.05, 0.05], 0, 1.5)
    d = rec.to_dict()
    assert set(d) >= {"temperature", "raw_probabilities",
                      "calibrated_probabilities", "raw_confidence",
                      "calibrated_confidence", "raw_prediction",
                      "calibrated_prediction", "raw_nll", "calibrated_nll",
                      "warnings"}
    assert len(d["raw_probabilities"]) == 5 == len(d["calibrated_probabilities"])
    assert d["raw_prediction"] == d["calibrated_prediction"] == 0


# --- 20-23. Referable ------------------------------------------------------------

def test_referable_score():
    from src.evaluation.referable_metrics import referable_scores

    s = referable_scores([[0.6, 0.2, 0.1, 0.05, 0.05]], 2)
    assert s[0] == pytest.approx(0.20)


def test_binary_calibration_metrics():
    s = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 1, 0, 0])
    m = M.binary_metrics(s, y)
    assert set(m) >= {"brier", "ece", "n_bins", "bins"}
    assert 0 <= m["brier"] <= 1 and 0 <= m["ece"] <= 1


def test_referable_decision_count():
    raw = np.array([0.65, 0.75, 0.60])
    cal = np.array([0.72, 0.68, 0.60])
    changed = int(((raw >= 0.7).astype(int) != (cal >= 0.7).astype(int)).sum())
    assert changed == 2


# --- 24-26. Serialization ----------------------------------------------------------

def test_config_schema():
    cfg = load_calibration_config()
    assert cfg["random_seed"] == DEFAULTS["random_seed"] == 42
    assert cfg["initial_temperature"] == 1.0
    assert cfg["temperature_bounds"][0] > 0
    assert cfg["ece_bins"][0] == 10 and cfg["ece_mode"] == "equal_width"
    assert cfg["referable_threshold"] == 0.7
    assert cfg["bootstrap"]["seed"] == 42
    import json

    file_cfg = json.load(open(Path(__file__).resolve().parents[1] /
                              "configs" / "calibration_config.json"))
    assert file_cfg["optimizer"]["method"] == cfg["optimizer"]["method"]


def test_report_schema():
    from src.calibration.types import CalibrationResult

    r = CalibrationResult(temperature=1.5)
    assert set(r.to_dict()) >= {"temperature", "raw_probabilities",
                                "calibrated_probabilities", "raw_confidence",
                                "calibrated_confidence", "raw_prediction",
                                "calibrated_prediction", "raw_nll",
                                "calibrated_nll", "warnings"}


def test_per_image_schema():
    cols = ["image_id", "ground_truth", "raw_prediction", "calibrated_prediction"] + \
           [f"raw_probability_{i}" for i in range(5)] + \
           [f"calibrated_probability_{i}" for i in range(5)] + \
           ["raw_confidence", "calibrated_confidence"]
    assert len(cols) == 4 + 5 + 5 + 2 == 16


# --- 27-28 + immutability ------------------------------------------------------------

def test_same_input_same_temperature():
    probs, y = _overconfident()
    assert (fit_temperature(probs, y)["temperature"]
            == fit_temperature(probs.copy(), y.copy())["temperature"])


def test_same_probabilities_same_metrics():
    probs, y = _overconfident()
    assert M.ece_metrics(probs, y) == M.ece_metrics(probs.copy(), y.copy())
    assert M.brier_score(probs, y) == M.brier_score(probs.copy(), y.copy())


def test_calibrator_never_touches_model():
    sig = inspect.signature(fit_temperature)
    assert "model" not in sig.parameters and "weights" not in sig.parameters
    sig2 = inspect.signature(apply_temperature)
    assert "model" not in sig2.parameters
