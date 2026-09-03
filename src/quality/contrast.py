"""Contrast assessment (SIH26038 Phase 2).

ENGINEERING HEURISTIC — not clinically validated. Primary metric: std of
gray inside the retinal mask. A p95-p5 spread guard catches flat images
that std alone could miss; an upper guard flags excessive contrast
(clipping / aggressive processing — higher is NOT always better).
"""

import numpy as np

from src.quality.types import BAD, BORDERLINE, GOOD, ComponentResult


def assess_contrast(gray_f32, mask, cfg):
    c = cfg["contrast"]
    inside = gray_f32[mask > 0]
    if inside.size == 0:
        return ComponentResult(name="contrast", measurement=0.0,
                               measurement_unit="std_gray", score=0.0,
                               status=BAD, explanation="No retinal pixels to assess.",
                               details={})
    std = float(inside.std())
    spread = float(np.percentile(inside, 95) - np.percentile(inside, 5))
    if std >= c["std_good"] and spread >= c["spread_borderline"]:
        if std > c["std_excessive"]:
            status, expl = (BORDERLINE,
                            f"Contrast is excessively high ({std:.1f}); possible clipping "
                            f"or aggressive processing.")
        else:
            status, expl = GOOD, f"Contrast adequate (variation {std:.1f})."
    elif std >= c["std_borderline"]:
        status, expl = (BORDERLINE,
                        f"Contrast is below the preferred level (variation {std:.1f}).")
    else:
        status, expl = (BAD,
                        f"Contrast too low to distinguish retinal detail (variation {std:.1f}).")
    score = max(0.0, min(1.0, (std - c["std_borderline"]) /
                         max(c["std_good"] - c["std_borderline"], 1e-6)))
    if std > c["std_excessive"]:
        score = 0.5
    return ComponentResult(name="contrast", measurement=std,
                           measurement_unit="std_gray", score=score,
                           status=status, explanation=expl,
                           details={"spread_p95_p5": round(spread, 2)})
