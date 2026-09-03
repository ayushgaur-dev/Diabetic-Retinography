"""Hemorrhage evidence: larger dark/red regions (SIH26038 Phase 4D).

Separate detector from MA (scale differs ~10x): regional darkness —
pixel darker than its large-scale background (median-blurred green) —
PLUS red-channel darkness, because big soft blotches have weak edges
that top-hat misses. Generous size range, elongated/irregular morphology
allowed, higher vessel tolerance (bleeds touch vessels). Interpretable.
"""

import cv2
import numpy as np

from src.retina.lesions.microaneurysm import _score_components


def detect_hemorrhages(work_rgb, fov, vessel, dist_vessel, green_norm,
                       cfg, common, width):
    c = cfg["hemorrhage"]
    k = int(c.get("background_kernel", 51)) | 1
    bg = cv2.medianBlur((green_norm * 255).astype(np.uint8), k).astype(np.float32) / 255.0
    dark_regional = np.clip(bg - green_norm, 0, 1)
    red = work_rgb[:, :, 0].astype(np.float32) / 255.0
    red_in = red[fov > 0]
    if red_in.size == 0:
        return np.zeros(fov.shape, np.uint8), []
    red_dark = 1.0 - (red - red_in.min()) / max(red_in.max() - red_in.min(), 1e-6)
    resp = np.clip(0.6 * dark_regional / max(dark_regional.max(), 1e-6)
                   + 0.4 * np.clip(red_dark, 0, 1), 0, 1)
    resp[fov == 0] = 0.0
    inside = resp[fov > 0]
    if inside.size == 0 or inside.max() <= 0:
        return np.zeros(fov.shape, np.uint8), []
    thresh = float(np.percentile(inside, c["response_percentile"]))
    raw = (((resp > thresh) & (fov > 0)).astype(np.uint8)) * 255
    return _score_components(raw, resp, work_rgb, fov, vessel, dist_vessel,
                             c, common, width, "hemorrhage")
