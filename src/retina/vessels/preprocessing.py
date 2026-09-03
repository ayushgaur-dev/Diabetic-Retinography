"""Vessel preprocessing: green channel + FOV-restricted normalisation.

Why green? Haemoglobin absorbs green light strongly, so vessels appear
darkest against the retinal background in the green channel; the red
channel is often saturated and the blue channel is noisy. Dedicated path —
Phase 3 colour normalisation is NOT reused (it would compress exactly the
vessel contrast we need).
"""

import cv2
import numpy as np


def extract_green(rgb):
    """RGB uint8 -> float32 green channel normalised to 0..1."""
    _check_rgb(rgb)
    return rgb[:, :, 1].astype(np.float32) / 255.0


def preprocess_green(rgb, fov_mask, low_pct=1.0, high_pct=99.0):
    """Green channel with mild FOV-restricted percentile stretch.
    Returns float32 0..1, finite, same HxW. Background outside FOV is 0."""
    green = extract_green(rgb)
    fov = _check_mask(fov_mask, rgb.shape[:2])
    inside = green[fov > 0]
    if inside.size == 0:
        return np.zeros_like(green)
    lo, hi = np.percentile(inside, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(green)
    out = np.clip((green - lo) / (hi - lo), 0, 1).astype(np.float32)
    out[fov == 0] = 0.0
    return out


def _check_rgb(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
            or rgb.dtype != np.uint8):
        raise ValueError(f"Expected RGB uint8 array, got {type(rgb)} {getattr(rgb, 'shape', None)}.")


def _check_mask(mask, shape):
    m = np.asarray(mask)
    if m.shape != shape:
        raise ValueError(f"FOV mask shape {m.shape} != image shape {shape}.")
    return (m > 0).astype(np.uint8) * 255
