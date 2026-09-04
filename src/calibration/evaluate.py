"""Calibration evaluation CLI (SIH26038 Phase 7, offline, no API).

Flow: Phase 5 split (+artifact verification) -> fresh VAL inference ->
fit T (validation ONLY) -> FREEZE -> test probs (verified cache or fresh)
-> raw-vs-calibrated comparison, referable analysis, bootstrap, figures.

Usage: python -m src.calibration.evaluate --data-root <APTOS> [--out DIR] [--no-bootstrap]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.calibration import inference as CI  # noqa: E402
from src.calibration.config import load_calibration_config  # noqa: E402
from src.calibration.dataset import (load_phase5_splits,  # noqa: E402
                                     verify_against_phase5_artifact)
from src.calibration.evaluator import (bootstrap_calibration, full_comparison,  # noqa: E402
                                       referable_comparison)
from src.calibration.temperature_scaling import (apply_temperature,  # noqa: E402
                                                 fit_temperature)
from src.calibration.visualization import (save_confidence_shift,  # noqa: E402
                                           save_reliability)

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 7 calibration evaluation")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "calibration"))
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args(argv)

    t_all = time.perf_counter()
    cfg = load_calibration_config()
    splits, ecfg = load_phase5_splits(args.data_root)
    artifact = REPO_ROOT / "reports" / "grading" / "per_image.csv"
    split_check = verify_against_phase5_artifact(splits, artifact)
    model_path = str(REPO_ROOT / cfg["model_artifact"])
    size = 224  # frozen preprocessing, Phase 1/5 procedure

    # --- validation: fresh inference + fit (test untouched) ---
    val_recs = CI.fresh_records(splits["val"], model_path, size, 32, split="val")
    P_val = np.array([r.probabilities for r in val_recs])
    y_val = np.array([r.ground_truth for r in val_recs])
    fit = fit_temperature(P_val, y_val,
                          initial_temperature=cfg["initial_temperature"],
                          logT_bounds=tuple(cfg["optimizer"]["logT_bounds"]),
                          tolerance=cfg["optimizer"]["tolerance"],
                          max_iterations=cfg["optimizer"]["max_iterations"])
    T = fit["temperature"]
    lo, hi = cfg["temperature_bounds"]
    assert lo <= T <= hi, f"Fitted T={T} outside bounds {lo, hi}."

    # --- test: verified cache preferred, fresh fallback ---
    try:
        P_test, y_test, ids, cache_info = CI.verified_cached_test_probs(
            artifact, splits, model_path, size)
    except ValueError as e:
        print(f"Cache rejected ({e}); running fresh test inference.")
        test_recs = CI.fresh_records(splits["test"], model_path, size, 32, split="test")
        P_test = np.array([r.probabilities for r in test_recs])
        y_test = np.array([r.ground_truth for r in test_recs])
        ids = [r.image_id for r in test_recs]
        cache_info = {"source": "fresh inference (cache rejected)"}

    comp, P, C = full_comparison(P_test, y_test, T, cfg)
    ref = referable_comparison(P_test, y_test, T, cfg)
    summary = {
        "temperature": round(T, 4),
        "fit": {**fit, "data": "validation only (549 images)",
                "test_labels_used": False},
        "split_verification": {**split_check, "procedure": "Phase 5 notebook split"},
        "test_source": cache_info,
        "model": {"artifact": cfg["model_artifact"], "frozen": True,
                  "weights_hash": _weights_hash(model_path)},
        **comp,
        "referable_calibration": ref,
        "runtime_s": round(time.perf_counter() - t_all, 1),
    }
    if not args.no_bootstrap:
        summary["bootstrap"] = bootstrap_calibration(P_test, y_test, T, cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize(summary), f, indent=2)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "ground_truth", "raw_prediction",
                    "calibrated_prediction"] +
                   [f"raw_probability_{i}" for i in range(5)] +
                   [f"calibrated_probability_{i}" for i in range(5)] +
                   ["raw_confidence", "calibrated_confidence"])
        for i, _id in enumerate(ids):
            rp, cp = int(P[i].argmax()), int(C[i].argmax())
            w.writerow([_id, int(y_test[i]), rp, cp] +
                       [round(float(v), 4) for v in P[i]] +
                       [round(float(v), 4) for v in C[i]] +
                       [round(float(P[i].max()), 4), round(float(C[i].max()), 4)])
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    save_reliability(comp["ece_bins_detail"]["raw"], "Reliability — raw softmax (test)",
                     figs / "reliability_raw.png")
    save_reliability(comp["ece_bins_detail"]["calibrated"],
                     "Reliability — temperature-scaled (test)",
                     figs / "reliability_calibrated.png")
    save_reliability(ref["binary_bins"]["raw"], "Referable reliability — raw (test)",
                     figs / "reliability_referable_raw.png",
                     xlabel="Mean referable score")
    save_reliability(ref["binary_bins"]["calibrated"],
                     "Referable reliability — calibrated (test)",
                     figs / "reliability_referable_calibrated.png",
                     xlabel="Mean referable score")
    save_confidence_shift(P.max(axis=1), C.max(axis=1),
                          figs / "confidence_shift.png")
    print(f"T={T:.4f} val NLL {fit['nll_before']} -> {fit['nll_after']}")
    print(f"test ECE {comp['ece']['10']['raw']} -> {comp['ece']['10']['calibrated']} | "
          f"Brier {comp['brier']['raw']} -> {comp['brier']['calibrated']} | "
          f"NLL {comp['nll']['raw']} -> {comp['nll']['calibrated']}")
    print(f"referable decisions changed: {ref['changed_referable_decisions']} | "
          f"argmax changed: {comp['prediction_invariance']['n_changed']}")
    print(f"wrote {out}")
    return 0


def _sanitize(obj):
    import math

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _weights_hash(model_path):
    import hashlib

    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
