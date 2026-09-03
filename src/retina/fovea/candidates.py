"""Fovea candidate generation + features (SIH26038 Phase 4C).

Grid points come from the disc-relative search region (one grid per
temporal hypothesis when laterality is unknown). Appearance features use
the green channel: darkness (patch mean vs FOV median), contrast (patch
vs ring), texture (smooth-dark patch vs textured surround). Boundary
distance uses a distance transform so edge-adjacent candidates are
down-weighted. Every value is computed, never hard-coded per image.
"""

import cv2
import numpy as np

from src.retina.fovea.anatomical_geometry import search_points
from src.retina.fovea.types import FoveaCandidate


def _disk_masks(yy, xx, x, y, r_in, r_out):
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    return d2 <= r_in ** 2, (d2 > r_in ** 2) & (d2 <= r_out ** 2)


def _appearance(x, y, green255, yy, xx, fov_median, disc_diameter, cfg):
    """green255/FOV-median in 0..255 units to match darkness_range /
    contrast_range config scale. Texture uses a std RATIO (scale-free)."""
    a = cfg["appearance"]
    r = a["patch_radius_dd"] * disc_diameter
    inner, ring = _disk_masks(yy, xx, x, y, r, r * a["ring_factor"])
    patch = green255[inner]
    ringv = green255[ring]
    if patch.size == 0 or ringv.size == 0:
        return 0.0, 0.0, 0.0
    pm, rm = float(patch.mean()), float(ringv.mean())
    darkness = max(0.0, min(1.0, (fov_median - pm) / a["darkness_range"] + 0.5))
    contrast = max(0.0, min(1.0, abs(pm - rm) / a["contrast_range"]))
    rstd = max(float(ringv.std()), 1e-6)
    texture = max(0.0, min(1.0, 1.0 - float(patch.std()) / rstd))
    return darkness, contrast, texture


def generate_candidates(green, fov_mask, anchor_xy, disc_diameter, points,
                        vessel_mask, cfg):
    """Score one candidate per grid point. anchor_xy is the disc center when
    geometry is used, else the FOV center (disc-relative fields then only
    describe position, and scoring drops the geometry term). FOV validity is
    a hard gate (background points are never generated)."""
    from src.retina.fovea.vessel_features import vessel_sparsity

    fov_median = float(np.median(green[fov_mask > 0])) * 255.0
    green255 = green * 255.0
    edge_dist = cv2.distanceTransform(((fov_mask > 0).astype(np.uint8)) * 255,
                                      cv2.DIST_L2, 3)
    fov_scale = max(float(edge_dist.max()), 1.0)
    yy, xx = np.mgrid[0:green.shape[0], 0:green.shape[1]]  # built ONCE
    ax, ay = anchor_xy
    cands = []
    for (x, y) in points:
        dark, cont, tex = _appearance(x, y, green255, yy, xx, fov_median,
                                      disc_diameter, cfg)
        dd = float(np.hypot(x - ax, y - ay)) / disc_diameter
        ang = float(np.degrees(np.arctan2(y - ay, x - ax)))
        spars = vessel_sparsity(x, y, vessel_mask, disc_diameter, cfg, xx, yy)
        ix = min(max(int(round(x)), 0), fov_mask.shape[1] - 1)
        iy = min(max(int(round(y)), 0), fov_mask.shape[0] - 1)
        bdist = float(edge_dist[iy, ix] / fov_scale)
        cands.append(FoveaCandidate(
            x=x, y=y, disc_relative_distance_dd=dd,
            disc_relative_angle_deg=ang, temporal_geometry_score=0.0,
            local_darkness_score=dark, local_contrast_score=cont,
            local_texture_score=tex, vessel_sparsity_score=spars,
            fov_validity=1.0, boundary_distance=round(bdist, 3)))
    return cands
