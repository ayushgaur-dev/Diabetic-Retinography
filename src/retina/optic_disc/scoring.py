"""Interpretable candidate scoring (SIH26038 Phase 4B).

score = w_bri * brightness + w_con * contrast + w_geo * geometry
        + w_conv * convergence, each term 0..1, weights from config.
Every candidate keeps its per-term contributions for debugging — no
black box. Normalisation ranges are fixed scale assumptions documented
below (engineering heuristics, FOV-relative where possible).
"""

import numpy as np


def _norm(x, lo, hi):
    return max(0.0, min(1.0, (x - lo) / max(hi - lo, 1e-9)))


def score_candidates(candidates, cfg, red_median=0.0, red_max=255.0):
    """Brightness is normalized to the IMAGE's own FOV red range
    (median..max), not an absolute 0..255 scale — dark but valid captures
    must still be scorable. Other terms are scale-free."""
    w = cfg["scoring"]["weights"]
    for cand in candidates:
        b = _norm(cand.mean_brightness, red_median, max(red_max, red_median + 1.0))
        t = _norm(cand.contrast, 0.0, 40.0)
        g = _norm(cand.circularity, 0.35, 1.0)
        v = _norm(cand.convergence, 1.0, 3.0)  # 1x background -> 0, 3x+ -> 1
        contribs = {
            "brightness": w["brightness"] * b,
            "contrast": w["contrast"] * t,
            "geometry": w["geometry"] * g,
            "convergence": w["convergence"] * v,
        }
        cand.score = float(sum(contribs.values()))
        cand.contributions = contribs
    return sorted(candidates, key=lambda k: k.score, reverse=True)
