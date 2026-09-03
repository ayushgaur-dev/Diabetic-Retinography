"""Localization metrics (SIH26038 Phase 4B).

Definitions (all explicit):
- center_error_px: Euclidean distance predicted vs GT center (pixels).
- normalized_error: center_error_px / GT disc diameter (unitless). A value
  of 0.5 means the prediction is off by one disc radius.
- detection_at(t): fraction of images with normalized_error <= t AND
  detected. Thresholds are REPORTED (0.25 / 0.5 / 1.0), never tuned.
- bbox_iou: IoU of predicted circle-bbox vs GT circle-bbox (GT circle from
  softmap centroid + equivalent diameter), where GT masks exist.
"""

import numpy as np


def center_error(pred_xy, gt_xy):
    return float(np.hypot(pred_xy[0] - gt_xy[0], pred_xy[1] - gt_xy[1]))


def normalized_error(pred_xy, gt_xy, gt_diameter):
    if gt_diameter <= 0:
        raise ValueError("GT disc diameter must be positive.")
    return center_error(pred_xy, gt_xy) / float(gt_diameter)


def bbox_iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def summarize(records, thresholds=(0.25, 0.5, 1.0)):
    """records: list of dicts with detected, center_error_px,
    normalized_error (or None when undetected). Returns aggregate dict."""
    n = len(records)
    det = [r for r in records if r["detected"] and r["normalized_error"] is not None]
    errs = [r["normalized_error"] for r in det]
    px = [r["center_error_px"] for r in det]
    out = {
        "n_images": n,
        "n_detected": len(det),
        "detection_rate": round(len(det) / n, 4) if n else 0.0,
        "mean_normalized_error": round(float(np.mean(errs)), 4) if errs else None,
        "median_normalized_error": round(float(np.median(errs)), 4) if errs else None,
        "mean_center_error_px": round(float(np.mean(px)), 1) if px else None,
        "mean_bbox_iou": (round(float(np.mean([r["bbox_iou"] for r in det
                                               if r.get("bbox_iou") is not None])), 4)
                          if det else None),
    }
    for t in thresholds:
        out[f"detection_at_{t}"] = (round(sum(1 for r in det
                                              if r["normalized_error"] <= t) / n, 4)
                                    if n else 0.0)
    return out
