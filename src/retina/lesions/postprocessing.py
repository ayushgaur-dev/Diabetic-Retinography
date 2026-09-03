"""Shared postprocessing (SIH26038 Phase 4D).

Conservative by design: hole filling + FOV clipping (+ optional disc
exclusion applied by callers). NO aggressive smoothing or large-object
removal — microaneurysms must survive postprocessing.
"""

import cv2
import numpy as np


def cleanup_mask(mask, fov_mask, fill_holes=True, min_size=0):
    out = ((np.asarray(mask) > 0).astype(np.uint8)) * 255
    if fill_holes:
        # Flood-fill the background from the corner; unfilled zeros are holes.
        ff = out.copy()
        ff_mask = np.zeros((out.shape[0] + 2, out.shape[1] + 2), np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 128)
        out[ff == 0] = 255
    if min_size and min_size > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, 8)
        keep = np.zeros_like(out)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                keep[labels == i] = 255
        out = keep
    out[fov_mask == 0] = 0
    return out


def components(mask):
    """List of dicts: label, area, centroid (x=column, y=row), bbox, contour."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ((np.asarray(mask) > 0).astype(np.uint8)) * 255, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, c]) for c in
                            (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                             cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
        out.append({"label": i, "area": float(area),
                    "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                    "bbox": [x, y, x + w, y + h],
                    "mask": (labels == i)})
    return out


def circularity(area, perimeter):
    """4*pi*A/P^2 in [0, 1]. Pixelated perimeters of tiny blobs can yield
    >1 (arc-length underestimation) — clamped so shape terms stay sane."""
    if perimeter <= 0:
        return 0.0
    return max(0.0, min(1.0, 4 * np.pi * area / (perimeter ** 2)))
