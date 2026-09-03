"""Atomic enhancement operations (SIH26038 Phase 3).

Each op: (rgb uint8, mask uint8 0/255, params) -> rgb uint8.
All ops apply INSIDE the Phase 2 retinal mask and preserve the original
background outside it (no enhancing black borders). No op mutates its input.
"""

import cv2
import numpy as np


def apply_inside_mask(original, enhanced, mask):
    out = original.copy()
    out[mask > 0] = enhanced[mask > 0]
    return out


def apply_clahe(rgb, mask, clip_limit=2.0, tile_grid=(8, 8)):
    """CLAHE on the L channel of LAB (never independent RGB channels,
    which would distort hue). tile_grid is (width, height) per OpenCV."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid[0]), int(tile_grid[1])))
    l_eq = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)
    return apply_inside_mask(rgb, enhanced, mask)


def normalize_illumination(rgb, mask, median_kernel=41, gain_cap_lo=0.5,
                           gain_cap_hi=2.0):
    """Divide each channel by its slow background (large median blur) and
    rescale to the channel's masked mean. Gains are clipped so local
    pathology is never excessively amplified."""
    k = int(median_kernel) | 1  # medianBlur requires odd kernel
    out = rgb.astype(np.float32)
    for c in range(3):
        ch = rgb[:, :, c].astype(np.float32)
        # cv2.medianBlur requires CV_8U input — blur the uint8 channel,
        # then compute the gain in float.
        bg = cv2.medianBlur(rgb[:, :, c], k).astype(np.float32)
        bg = np.maximum(bg, 1.0)
        mean = float(ch[mask > 0].mean()) if (mask > 0).any() else float(ch.mean())
        gain = np.clip(mean / bg, gain_cap_lo, gain_cap_hi)
        out[:, :, c] = np.clip(ch * gain, 0, 255)
    return apply_inside_mask(rgb, out.astype(np.uint8), mask)


def denoise(rgb, mask, diameter=5, sigma_color=25.0, sigma_space=5.0):
    """Conservative edge-preserving bilateral filter. Deliberately mild:
    later phases analyse small lesions, so aggressive smoothing is banned."""
    smooth = cv2.bilateralFilter(rgb, int(diameter), float(sigma_color),
                                 float(sigma_space))
    return apply_inside_mask(rgb, smooth, mask)


def normalize_color(rgb, mask, max_gain=1.25):
    """Masked gray-world: scale channels toward their common mean with a
    capped gain. Corrects global camera cast while preserving the relative
    colour differences that carry clinical meaning. NOT a domain-shift fix."""
    f = rgb.astype(np.float32)
    means = np.array([f[:, :, c][mask > 0].mean() if (mask > 0).any() else f[:, :, c].mean()
                      for c in range(3)])
    target = float(means.mean())
    gains = np.clip(target / np.maximum(means, 1.0), 1.0 / max_gain, max_gain)
    out = np.clip(f * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    return apply_inside_mask(rgb, out, mask)


def channel_spread(rgb, mask):
    """Inter-channel mean spread inside the mask — the color-norm trigger."""
    f = rgb.astype(np.float32)
    means = [float(f[:, :, c][mask > 0].mean()) if (mask > 0).any() else 0.0
             for c in range(3)]
    return max(means) - min(means)
