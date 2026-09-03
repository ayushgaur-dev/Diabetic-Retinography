"""Vessel-convergence support evidence (SIH26038 Phase 4B).

The disc is where major vessels converge: vessel density in an annulus
around the candidate should exceed the background FOV density. This is
SUPPORTING evidence (a weight in the score), never a hard gate — vessel
segmentation is imperfect. When no vessel mask is available, convergence
is 0 for every candidate and a warning is recorded (robust fallback).
"""

import numpy as np


def compute_convergence(candidate, vessel_mask, fov_mask, cfg):
    """Annulus vessel-density ratio for one candidate (working-res coords)."""
    c = cfg["convergence"]
    r = float(np.sqrt(candidate.area / np.pi))
    yy, xx = np.mgrid[0:vessel_mask.shape[0], 0:vessel_mask.shape[1]]
    d2 = (xx - candidate.centroid_x) ** 2 + (yy - candidate.centroid_y) ** 2
    ring = ((d2 >= (r * c["ring_inner_factor"]) ** 2)
            & (d2 <= (r * c["ring_outer_factor"]) ** 2)
            & (fov_mask > 0))
    vessels = vessel_mask > 0
    ring_px = int(ring.sum())
    if ring_px == 0:
        return 0.0
    ring_density = float((vessels & ring).sum()) / ring_px
    fov_px = int((fov_mask > 0).sum())
    global_density = float((vessels & (fov_mask > 0)).sum()) / max(fov_px, 1)
    if global_density <= 0:
        return 0.0
    return ring_density / global_density


def attach_convergence(candidates, vessel_mask, fov_mask, cfg):
    for cand in candidates:
        cand.convergence = float(compute_convergence(cand, vessel_mask, fov_mask, cfg))
    return candidates
