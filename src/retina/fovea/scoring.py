"""Weighted fovea scoring (SIH26038 Phase 4C).

total = w_geo*geometry + w_app*appearance + w_ves*vessel + w_fov*fov,
weights from config (sum to 1.0). Geometry is a Gaussian falloff around
the expected point (per-hypothesis); appearance averages darkness,
contrast, texture; fov combines validity with boundary distance.
Contributions are inspectable per candidate — no opaque classifier.
"""

import numpy as np


def score_candidates(candidates, expected_points, disc_diameter, cfg,
                     use_geometry=True):
    w = cfg["scoring"]["weights"]
    sigma = cfg["geometry"]["geometry_sigma_dd"] * disc_diameter
    for cand in candidates:
        if use_geometry and expected_points:
            geo = max(float(np.exp(-((cand.x - ex) ** 2 + (cand.y - ey) ** 2)
                                   / (2 * sigma ** 2))) for ex, ey in expected_points)
        else:
            geo = 0.0  # fallback: no expected point exists
        cand.temporal_geometry_score = round(geo, 3)
        app = (cand.local_darkness_score + cand.local_contrast_score
               + cand.local_texture_score) / 3.0
        fov = 0.7 * cand.fov_validity + 0.3 * cand.boundary_distance
        contribs = {
            "geometry": w["geometry"] * geo,
            "appearance": w["appearance"] * app,
            "vessel": w["vessel"] * cand.vessel_sparsity_score,
            "fov": w["fov"] * fov,
        }
        cand.score = float(sum(contribs.values()))
        cand.contributions = contribs
    return sorted(candidates, key=lambda k: k.score, reverse=True)
