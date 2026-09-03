"""Bright-region candidate generation (SIH26038 Phase 4B).

Pipeline: FOV mask -> red-channel top-percentile -> connected components ->
geometry/area filters (all FOV-normalized) -> candidate features. Kept
SEPARATE from scoring (scoring.py). The brightest pixel is never blindly
trusted: reflections/exudates survive the brightness gate and must lose on
geometry, contrast, or vessel-convergence downstream.
"""

import cv2
import numpy as np

from src.retina.optic_disc.types import DiscCandidate


def generate_candidates(rgb, fov_mask, cfg):
    """Return list[DiscCandidate] in working-resolution coordinates."""
    c = cfg["candidates"]
    red = rgb[:, :, 0].astype(np.float32)
    inside = red[fov_mask > 0]
    if inside.size == 0:
        return []
    thresh = float(np.percentile(inside, c["red_percentile"]))
    bright = (((red >= thresh) & (fov_mask > 0)).astype(np.uint8)) * 255
    # Close vessel-induced splits: dark vessels crossing the disc fragment
    # its bright region; kernel scales with image width (resolution free).
    ksize = max(3, int(round(rgb.shape[1] * c.get("closing_kernel_frac", 0.02))) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, 8)
    fov_area = float((fov_mask > 0).sum())
    green = rgb[:, :, 1].astype(np.float32)
    candidates = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        area_frac = area / max(fov_area, 1.0)
        if not (c["min_area_frac_fov"] <= area_frac <= c["max_area_frac_fov"]):
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        perim = float(cv2.arcLength(cnt, True))
        circularity = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0.0
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / max(min(w, h), 1)
        if circularity < c["min_circularity"] or aspect > c["max_aspect_ratio"]:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        contrast = _ring_contrast(green, comp, cx, cy, area, c["contrast_ring_width_frac"])
        candidates.append(DiscCandidate(
            centroid_x=cx, centroid_y=cy, area=area,
            mean_brightness=float(red[comp > 0].mean()),
            contrast=float(contrast), circularity=float(circularity),
            aspect_ratio=float(aspect), convergence=0.0,
        ))
    return candidates


def _ring_contrast(green, comp, cx, cy, area, ring_frac):
    """Mean inside vs surrounding ring (local contrast of the region)."""
    r = float(np.sqrt(area / np.pi))
    outer = r * (1.0 + ring_frac)
    yy, xx = np.mgrid[0:green.shape[0], 0:green.shape[1]]
    ring = ((xx - cx) ** 2 + (yy - cy) ** 2 <= outer ** 2) & (comp == 0)
    ring_vals = green[ring]
    if ring_vals.size == 0:
        return 0.0
    return float(green[comp > 0].mean() - ring_vals.mean())
