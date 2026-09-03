"""Illumination assessment (SIH26038 Phase 2).

ENGINEERING HEURISTIC — not clinically validated. Uses the median gray
level inside the retinal mask (robust to small bright/dark lesions) plus
non-uniformity = std of 4x4 block means inside the mask (catches
half-shadow / vignetting that a global mean would hide).
"""

import numpy as np

from src.quality.types import BAD, BORDERLINE, GOOD, ComponentResult


def assess_illumination(gray_f32, mask, cfg):
    c = cfg["illumination"]
    inside = gray_f32[mask > 0]
    median = float(np.median(inside)) if inside.size else 0.0
    nonuniformity = _block_nonuniformity(gray_f32, mask)
    score = _median_score(median, c) * 0.7 + _uniformity_score(nonuniformity, c) * 0.3
    notes = []
    if median < c["median_borderline_lo"]:
        m_status, m_note = BAD, "image is too dark"
    elif median > c["median_borderline_hi"]:
        m_status, m_note = BAD, "image is too bright"
    elif median < c["median_good_lo"] or median > c["median_good_hi"]:
        m_status, m_note = BORDERLINE, "illumination is outside the preferred range"
    else:
        m_status, m_note = GOOD, "illumination is within the preferred range"
    notes.append(m_note)
    if nonuniformity >= c["nonuniformity_borderline"]:
        u_status, u_note = BAD, "illumination is strongly uneven across the retina"
    elif nonuniformity >= c["nonuniformity_good"]:
        u_status, u_note = BORDERLINE, "illumination is somewhat uneven"
    else:
        u_status, u_note = GOOD, "illumination is even"
    notes.append(u_note)
    status = _worst(m_status, u_status)
    expl = (f"Retinal brightness {median:.0f}/255 with {u_note} "
            f"(unevenness {nonuniformity:.1f}).")
    return ComponentResult(name="illumination", measurement=median,
                           measurement_unit="median_gray_0_255",
                           score=score, status=status, explanation=expl,
                           details={"nonuniformity": round(nonuniformity, 2),
                                    "notes": notes})


def _block_nonuniformity(gray, mask, blocks=4):
    h, w = gray.shape
    means = []
    for i in range(blocks):
        for j in range(blocks):
            cell = gray[i * h // blocks:(i + 1) * h // blocks,
                        j * w // blocks:(j + 1) * w // blocks]
            cell_mask = mask[i * h // blocks:(i + 1) * h // blocks,
                             j * w // blocks:(j + 1) * w // blocks]
            vals = cell[cell_mask > 0]
            if vals.size:
                means.append(float(vals.mean()))
    return float(np.std(means)) if means else 0.0


def _median_score(median, c):
    if c["median_good_lo"] <= median <= c["median_good_hi"]:
        return 1.0
    if c["median_borderline_lo"] <= median <= c["median_borderline_hi"]:
        return 0.5
    span = max(c["median_borderline_lo"], 255.0 - c["median_borderline_hi"], 1.0)
    dist = min(abs(median - c["median_good_lo"]), abs(median - c["median_good_hi"]))
    return max(0.0, 0.5 - 0.5 * dist / span)


def _uniformity_score(nu, c):
    if nu < c["nonuniformity_good"]:
        return 1.0
    if nu < c["nonuniformity_borderline"]:
        return 0.5
    return max(0.0, 0.5 - 0.5 * (nu - c["nonuniformity_borderline"]) / 30.0)


def _worst(a, b):
    order = {GOOD: 0, BORDERLINE: 1, BAD: 2}
    return a if order[a] >= order[b] else b
