"""Common-frame evidence map (SIH26038 Phase 6).

Every source is resampled into ORIGINAL image pixels (x=column, y=row):
- continuous heatmaps (Grad-CAM): bilinear (INTER_LINEAR)
- binary masks (FOV, vessels, lesions): nearest-neighbor (INTER_NEAREST)
- points/circles (disc, fovea): rasterized directly at original resolution

No coordinate shifts: all resampling maps array corners onto array
corners (cv2 dsize=(W,H)); verified by the coordinate test.
"""

import cv2
import numpy as np


def to_original(arr, shape_hw, is_mask=True):
    """Resample arr to (H, W). Masks -> nearest, heatmaps -> bilinear."""
    h, w = shape_hw
    if arr.shape[:2] == (h, w):
        return np.asarray(arr)
    flag = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    out = cv2.resize(np.asarray(arr), (w, h), interpolation=flag)
    if is_mask:
        out = ((out > 0).astype(np.uint8)) * 255
    return out


def upsample_heatmap(heat, shape_hw):
    h = np.asarray(heat, dtype=float)
    h = (h - h.min()) / (h.max() - h.min()) if h.max() > h.min() else h * 0
    return to_original(h, shape_hw, is_mask=False)


def rasterize_circle(shape_hw, center_xy, radius):
    yy, xx = np.mgrid[0:shape_hw[0], 0:shape_hw[1]]
    return ((((xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2)
             <= radius ** 2).astype(np.uint8)) * 255


def rasterize_point(shape_hw, center_xy, radius=5):
    return rasterize_circle(shape_hw, center_xy, max(float(radius), 1.0))


def hot_mask(heat_hires, quantile=0.75):
    """Top-activation binary mask (quantile over the heatmap)."""
    t = float(np.quantile(np.asarray(heat_hires).ravel(), quantile))
    return (((np.asarray(heat_hires) >= t).astype(np.uint8)) * 255)
