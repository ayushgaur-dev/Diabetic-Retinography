"""Thresholding + conservative postprocessing (SIH26038 Phase 4A)."""

import cv2
import numpy as np


def segment_response(response, fov_mask, method="percentile", percentile=89.0,
                     fixed_threshold=None):
    """Response map -> binary vessel mask (uint8 0/255), restricted to FOV."""
    inside = response[fov_mask > 0]
    if inside.size == 0:
        return np.zeros(response.shape, dtype=np.uint8)
    if method == "percentile":
        thresh = float(np.percentile(inside, percentile))
    elif method == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed thresholding requires fixed_threshold.")
        thresh = float(fixed_threshold)
    else:
        raise ValueError(f"Unknown segmentation method: {method!r}.")
    # Strictly-greater comparison: a degenerate all-zero response (threshold
    # 0) must yield an empty mask, not the whole FOV.
    return (((response > thresh) & (fov_mask > 0)).astype(np.uint8)) * 255


def postprocess_mask(mask, fov_mask, min_component_size=25, closing_kernel=3,
                     apply_closing=True):
    """Conservative cleanup: drop tiny isolated components (noise) and an
    optional small closing to bridge single-pixel gaps. min_component_size
    is deliberately small so thin vessels survive."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
            out[labels == i] = 255
    if apply_closing and closing_kernel and closing_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (closing_kernel, closing_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    out[fov_mask == 0] = 0
    return out
