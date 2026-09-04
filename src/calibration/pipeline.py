"""Calibration application pipeline (SIH26038 Phase 7). Array-level API —
no model object, no weight access (structural immutability, tested)."""

import numpy as np

from src.calibration.temperature_scaling import (apply_temperature,
                                                 multiclass_nll)
from src.calibration.types import CalibrationResult


def calibrate_record(probs, label, temperature, eps=1e-12):
    """One image: raw + calibrated side-by-side. Never mutates inputs."""
    p = np.asarray(probs, dtype=float)
    c = apply_temperature(p, temperature, eps)
    rp, cp = int(np.argmax(p)), int(np.argmax(c))
    return CalibrationResult(
        temperature=float(temperature),
        raw_probabilities=list(p), calibrated_probabilities=list(c),
        raw_confidence=float(p.max()), calibrated_confidence=float(c.max()),
        raw_prediction=rp, calibrated_prediction=cp,
        raw_nll=float(-np.log(max(p[int(label)], eps))),
        calibrated_nll=float(-np.log(max(c[int(label)], eps))),
        warnings=([] if rp == cp else
                  ["Argmax changed by calibration — investigate."]),
    )
