"""Disc preprocessing: working resize + red-channel luminance (SIH26038 4B).

The disc is a yellow-white structure: brightest in the RED channel, while
vessels (dark in green) interfere least there. Green is kept for the local
contrast term. All downstream thresholds are FOV-normalized, so the working
resize (documented, back-mapped) does not bake in a resolution.
"""

import cv2
import numpy as np


def to_working_resolution(rgb, max_width=800):
    """Downscale so width <= max_width. Returns (small, scale) where
    original = small / scale. scale == 1.0 when already small."""
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb.copy(), 1.0
    scale = max_width / float(w)
    small = cv2.resize(rgb, (max_width, int(round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return small, scale
