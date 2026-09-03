"""Fovea metrics (SIH26038 Phase 4C).

GT disc diameter is unavailable in IDRiD Localization (centers only), so
the primary normalization is FOV width (documented alternative). Where a
disc prediction exists, a secondary disc-diameter normalization is also
reported. Detection thresholds (0.025 / 0.05 / 0.10 FOV-width ≈ 0.17 /
0.33 / 0.67 disc diameters at typical anatomy) are REPORTED, never tuned.
"""

import numpy as np


def center_error(pred_xy, gt_xy):
    return float(np.hypot(pred_xy[0] - gt_xy[0], pred_xy[1] - gt_xy[1]))


def normalized_error_fov(pred_xy, gt_xy, fov_width):
    if fov_width <= 0:
        raise ValueError("FOV width must be positive.")
    return center_error(pred_xy, gt_xy) / float(fov_width)


def summarize(records, thresholds=(0.025, 0.05, 0.1)):
    n = len(records)
    det = [r for r in records if r["detected"] and r["norm_err"] is not None]
    errs = [r["norm_err"] for r in det]
    px = [r["px_err"] for r in det]
    out = {
        "n_images": n,
        "n_detected": sum(1 for r in records if r["status"] == "DETECTED"),
        "n_low_confidence": sum(1 for r in records if r["status"] == "LOW_CONFIDENCE"),
        "n_not_detected": sum(1 for r in records if r["status"] == "NOT_DETECTED"),
        "detection_rate": round(sum(1 for r in records if r["detected"]) / n, 4) if n else 0.0,
        "mean_normalized_error": round(float(np.mean(errs)), 4) if errs else None,
        "median_normalized_error": round(float(np.median(errs)), 4) if errs else None,
        "mean_center_error_px": round(float(np.mean(px)), 1) if px else None,
    }
    for t in thresholds:
        out[f"detection_at_{t}"] = (round(sum(1 for r in det
                                              if r["norm_err"] <= t) / n, 4)
                                    if n else 0.0)
    for lat in ("left", "right", "unknown"):
        sub = [r for r in records if r.get("laterality_gt") == lat]
        dsub = [r for r in sub if r["detected"] and r["norm_err"] is not None]
        out[f"laterality_{lat}"] = {
            "n": len(sub),
            "detection_rate": (round(sum(1 for r in sub if r["detected"]) / len(sub), 4)
                               if sub else None),
            "median_normalized_error": (round(float(np.median([r["norm_err"]
                                                               for r in dsub])), 4)
                                        if dsub else None),
        }
    return out
