"""Vessel-sparsity heuristic (SIH26038 Phase 4C).

The fovea sits in a relatively avascular zone: vessel density in a small
inner disk should be LOW relative to a surrounding ring. Score =
clipped(outer/inner ratio) mapped to 0..1. This is an image-derived
HEURISTIC — explicitly NOT a validated clinical biomarker.
"""

import numpy as np


def vessel_sparsity(x, y, vessel_mask, disc_diameter, cfg, xx=None, yy=None):
    v = cfg["vessel_sparsity"]
    inner_r = v["inner_radius_dd"] * disc_diameter
    outer_r = v["outer_radius_dd"] * disc_diameter
    if xx is None or yy is None:  # fallback: build grids (slower)
        yy, xx = np.mgrid[0:vessel_mask.shape[0], 0:vessel_mask.shape[1]]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    inner = d2 <= inner_r ** 2
    ring = (d2 > inner_r ** 2) & (d2 <= outer_r ** 2)
    vessels = vessel_mask > 0
    di = float((vessels & inner).sum()) / max(int(inner.sum()), 1)
    do = float((vessels & ring).sum()) / max(int(ring.sum()), 1)
    if di <= 0:
        return 1.0 if do > 0 else 0.5  # empty center: avascular iff ring has vessels
    return min(do / di, v["ratio_cap"]) / v["ratio_cap"]
