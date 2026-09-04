"""Calibration evaluation orchestration (SIH26038 Phase 7). Compares raw vs
temperature-scaled probabilities on FROZEN test data. Never fits here."""

import numpy as np

from src.calibration import metrics as M
from src.calibration.temperature_scaling import (apply_temperature,
                                                 multiclass_nll)


def _rounded(d):
    return {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in d.items()}


def full_comparison(test_probs, test_labels, temperature, cfg):
    """Return raw/calibrated metric comparison dict (test-only)."""
    P = np.asarray(test_probs, dtype=float)
    y = np.asarray(test_labels, dtype=int)
    C = apply_temperature(P, temperature)
    bins_list = cfg["ece_bins"]
    ece_raw = {str(b): M.ece_metrics(P, y, b) for b in bins_list}
    ece_cal = {str(b): M.ece_metrics(C, y, b) for b in bins_list}
    out = {
        "temperature": round(float(temperature), 4),
        "n": len(y),
        "nll": {"raw": round(multiclass_nll(P, y), 4),
                "calibrated": round(multiclass_nll(C, y), 4)},
        "brier": {"raw": round(M.brier_score(P, y), 4),
                  "calibrated": round(M.brier_score(C, y), 4)},
        "ece": {b: {"raw": ece_raw[b]["ece"], "calibrated": ece_cal[b]["ece"],
                   "delta": round(ece_cal[b]["ece"] - ece_raw[b]["ece"], 4)}
                for b in ece_raw},
        "mce": {b: {"raw": ece_raw[b]["mce"], "calibrated": ece_cal[b]["mce"]}
                for b in ece_raw},
        "ece_bins_detail": {"raw": ece_raw[str(bins_list[0])]["bins"],
                            "calibrated": ece_cal[str(bins_list[0])]["bins"]},
        "classwise": {"raw": M.classwise_metrics(P, y),
                      "calibrated": M.classwise_metrics(C, y)},
        "confidence": {"raw": M.confidence_distribution(P, y),
                       "calibrated": M.confidence_distribution(C, y)},
    }
    # prediction invariance (multiclass argmax must be identical)
    same = bool((P.argmax(axis=1) == C.argmax(axis=1)).all())
    out["prediction_invariance"] = {
        "multiclass_identical": same,
        "n_changed": int((P.argmax(axis=1) != C.argmax(axis=1)).sum()),
    }
    if not same:
        out["prediction_invariance"]["WARNING"] = \
            "Argmax changed — investigate before trusting calibration."
    return out, P, C


def referable_comparison(test_probs, test_labels, temperature, cfg):
    """Binary referable calibration at the FROZEN Phase 5 threshold."""
    pgm = cfg["referable_grade_min"]
    thr = cfg.get("referable_threshold", 0.7)
    P = np.asarray(test_probs, dtype=float)
    y = (np.asarray(test_labels, dtype=int) >= pgm).astype(int)
    C = apply_temperature(P, temperature)
    s_raw, s_cal = P[:, pgm:].sum(axis=1), C[:, pgm:].sum(axis=1)
    d_raw, d_cal = (s_raw >= thr).astype(int), (s_cal >= thr).astype(int)
    changed = int((d_raw != d_cal).sum())
    from src.evaluation.referable_metrics import referable_metrics as rm

    return {
        "threshold": thr, "frozen_from": "Phase 5 validation (unchanged)",
        "binary_brier": {"raw": M.binary_metrics(s_raw, y)["brier"],
                         "calibrated": M.binary_metrics(s_cal, y)["brier"]},
        "binary_ece": {"raw": M.binary_metrics(s_raw, y)["ece"],
                       "calibrated": M.binary_metrics(s_cal, y)["ece"]},
        "binary_bins": {"raw": M.binary_metrics(s_raw, y, 10)["bins"],
                        "calibrated": M.binary_metrics(s_cal, y, 10)["bins"]},
        "decisions": {"raw": rm(y, d_raw), "calibrated": rm(y, d_cal)},
        "changed_referable_decisions": changed,
        "changed_fraction": round(changed / len(y), 4),
    }


def bootstrap_calibration(test_probs, test_labels, temperature, cfg):
    """Deterministic bootstrap CIs for ECE/Brier/NLL/MCE (raw + calibrated)."""
    P = np.asarray(test_probs, dtype=float)
    y = np.asarray(test_labels, dtype=int)
    C = apply_temperature(P, temperature)

    def raw_fn(pb, yb):
        return {"ece": M.ece_metrics(pb, yb, 10)["ece"],
                "brier": M.brier_score(pb, yb),
                "nll": multiclass_nll(pb, yb),
                "mce": M.ece_metrics(pb, yb, 10)["mce"]}

    def cal_fn(pb, yb):
        Cc = apply_temperature(pb, temperature)
        return {"ece": M.ece_metrics(Cc, yb, 10)["ece"],
                "brier": M.brier_score(Cc, yb),
                "nll": multiclass_nll(Cc, yb),
                "mce": M.ece_metrics(Cc, yb, 10)["mce"]}

    n, seed = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"]
    return {"raw": M.bootstrap_cis(raw_fn, P, y, n, seed),
            "calibrated": M.bootstrap_cis(cal_fn, P, y, n, seed),
            "n_resamples": n, "seed": seed}
