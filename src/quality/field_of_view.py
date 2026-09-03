"""Retinal field-of-view detection (SIH26038 Phase 2).

Baseline quality-control detector, NOT a clinical segmentation:
red-channel threshold (fundus is red/orange, background is near-black) +
morphology + largest connected component. Returns the mask plus fitted
geometry used by the coverage module.
"""

import cv2
import numpy as np


def detect_retinal_field(rgb, cfg):
    """Detect the retinal field in an RGB uint8 image.

    Returns dict with: mask (uint8 0/255), mask_fraction, center (x, y),
    radius (min-enclosing-circle), completeness (mask area / circle area),
    plausible (bool), reason (str).
    """
    fov = cfg["field_of_view"]
    h, w = rgb.shape[:2]
    red = rgb[:, :, 0].astype(np.float32)
    thresh = max(float(fov["red_floor"]), float(red.mean()) * float(fov["red_factor"]))
    mask = ((red > thresh).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return _empty(h, w, "no retinal region detected above background")
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[biggest, cv2.CC_STAT_AREA])
    mask_fraction = area / float(h * w)
    if mask_fraction < float(fov["min_plausible_fraction"]):
        return _empty(h, w, f"retinal region too small ({mask_fraction:.3f} of frame)",
                      mask_fraction=mask_fraction)
    field = (labels == biggest).astype(np.uint8)
    contours, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty(h, w, "retinal contour extraction failed", mask_fraction=mask_fraction)
    (cx, cy), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    circle_area = float(np.pi * max(radius, 1e-6) ** 2)
    completeness = min(area / circle_area, 1.0)
    return {
        "mask": (field * 255).astype(np.uint8),
        "mask_fraction": mask_fraction,
        "center": (float(cx), float(cy)),
        "radius": float(radius),
        "completeness": float(completeness),
        "plausible": True,
        "reason": "",
    }


def _empty(h, w, reason, mask_fraction=0.0):
    return {
        "mask": np.zeros((h, w), dtype=np.uint8),
        "mask_fraction": float(mask_fraction),
        "center": (0.0, 0.0),
        "radius": 0.0,
        "completeness": 0.0,
        "plausible": False,
        "reason": reason,
    }
