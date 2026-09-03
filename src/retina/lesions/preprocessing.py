"""Shared lesion preprocessing (SIH26038 Phase 4D).

Deliberately NOT 224px: microaneurysms are ~10-30px at native 4288px and
would vanish. Default working width 1072 (configurable) preserves tiny
lesions while keeping morphology affordable. Outputs: working RGB,
FOV mask, normalized green, and scale factor for back-mapping.
"""

import cv2
import numpy as np


def to_working_resolution(rgb, max_width):
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb.copy(), 1.0
    scale = max_width / float(w)
    return cv2.resize(rgb, (max_width, int(round(h * scale))),
                      interpolation=cv2.INTER_AREA), scale


def lesion_preprocess(rgb, fov_mask, low_pct=1.0, high_pct=99.0):
    """FOV-restricted percentile-stretched green (float32 0..1) + FOV mask."""
    green = rgb[:, :, 1].astype(np.float32) / 255.0
    inside = green[fov_mask > 0]
    if inside.size == 0:
        return np.zeros_like(green), fov_mask
    lo, hi = np.percentile(inside, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(green), fov_mask
    out = np.clip((green - lo) / (hi - lo), 0, 1).astype(np.float32)
    out[fov_mask == 0] = 0.0
    return out, fov_mask
