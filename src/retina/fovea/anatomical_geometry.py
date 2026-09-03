"""Anatomical geometry: laterality + disc-relative search region (Phase 4C).

Laterality is INFERRED per image (never assumed): the disc centroid x vs
the FOV centroid x decides the temporal side (disc nasal). If the margin
is below min_margin_frac_fov, laterality is 'unknown' and BOTH temporal
directions are searched — scoring picks the winner and confidence drops.

All distances in disc-diameter (DD) units. Without a disc, geometry is
absent (search falls back to the whole FOV in candidates.py).
"""

import numpy as np


def infer_laterality(disc_xy, fov_mask, cfg):
    """Return (laterality, temporal_sign, confident). temporal_sign is +1
    (temporal = +x, right eye) or -1 (left eye); 0 when unknown."""
    fov = fov_mask > 0
    ys, xs = np.nonzero(fov)
    fov_cx = float(xs.mean()) if xs.size else 0.0
    fov_w = float(fov.shape[1])
    margin = (disc_xy[0] - fov_cx) / max(fov_w, 1.0)
    if abs(margin) < cfg["laterality"]["min_margin_frac_fov"]:
        return "unknown", 0, False
    if margin < 0:  # disc left of field -> temporal is +x -> right eye
        return "right", +1, True
    return "left", -1, True


def expected_fovea(disc_xy, disc_diameter, temporal_sign, cfg):
    """Expected fovea point (x=column, y=row) for one temporal hypothesis."""
    g = cfg["geometry"]
    return (disc_xy[0] + temporal_sign * g["disc_fovea_distance_dd"] * disc_diameter,
            disc_xy[1] + g["vertical_offset_dd"] * disc_diameter)


def search_points(expected_xy, disc_diameter, fov_mask, cfg):
    """Grid of candidate points inside the search disk, clipped to FOV."""
    g = cfg["geometry"]
    step = max(g["grid_step_dd"] * disc_diameter, 1.0)
    rad = g["search_radius_dd"] * disc_diameter
    ex, ey = expected_xy
    xs = np.arange(ex - rad, ex + rad + step, step)
    ys = np.arange(ey - rad, ey + rad + step, step)
    pts = []
    h, w = fov_mask.shape
    for x in xs:
        for y in ys:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h and fov_mask[iy, ix] > 0:
                if (x - ex) ** 2 + (y - ey) ** 2 <= rad ** 2:
                    pts.append((float(x), float(y)))
    return pts
