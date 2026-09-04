"""Calibration placeholder (SIH26038 Phase 5).

There is deliberately NO calibration here. This module exists so that:
1. tests can assert raw softmax is never presented as calibrated, and
2. Phase 7 has a defined integration point.

Any function claiming calibration raises NotImplementedError until Phase 7.
"""

CALIBRATED = False


def assert_not_calibrated():
    raise NotImplementedError(
        "Confidence calibration is Phase 7 work. Raw softmax must be "
        "labelled raw/uncalibrated everywhere (see raw_confidence_analysis).")


def describe_raw_confidence(records):
    """Descriptive stats only — NOT calibration."""
    import numpy as np

    conf = np.array([r.raw_confidence for r in records])
    correct = np.array([r.predicted_grade == r.ground_truth for r in records])
    return {
        "mean_max_prob": round(float(conf.mean()), 4),
        "median_max_prob": round(float(np.median(conf)), 4),
        "mean_correct": round(float(conf[correct].mean()), 4) if correct.any() else None,
        "mean_incorrect": round(float(conf[~correct].mean()), 4) if (~correct).any() else None,
        "histogram_10bin": [int(v) for v in np.histogram(conf, bins=10, range=(0, 1))[0]],
    }
