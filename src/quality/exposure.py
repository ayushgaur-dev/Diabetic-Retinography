"""Exposure / clipping assessment (SIH26038 Phase 2).

ENGINEERING HEURISTIC — not clinically validated. Measures the fraction
of retinal-mask pixels at/near black (<= black_level) and white
(>= white_level). Severe clipping destroys lesion detail, so BAD exposure
alone forces UNGRADABLE in the aggregation rules.
"""

import numpy as np

from src.quality.types import BAD, BORDERLINE, GOOD, ComponentResult


def assess_exposure(gray_f32, mask, cfg):
    c = cfg["exposure"]
    inside = gray_f32[mask > 0]
    if inside.size == 0:
        return ComponentResult(name="exposure", measurement=0.0,
                               measurement_unit="clipped_fraction", score=0.0,
                               status=BAD, explanation="No retinal pixels to assess.",
                               details={})
    black_frac = float((inside <= c["black_level"]).mean())
    white_frac = float((inside >= c["white_level"]).mean())
    clipped = black_frac + white_frac
    b_status = _level(black_frac, c["black_borderline"], c["black_bad"])
    w_status = _level(white_frac, c["white_borderline"], c["white_bad"])
    status = b_status if _rank(b_status) >= _rank(w_status) else w_status
    if status == BAD:
        if _rank(b_status) >= _rank(w_status):
            expl = (f"Image is too dark ({black_frac:.1%} of retina near black). "
                    f"Improve illumination and recapture.")
        else:
            expl = (f"Image contains excessive bright saturation ({white_frac:.1%} "
                    f"of retina near white). Adjust illumination and recapture.")
    elif status == BORDERLINE:
        expl = (f"Exposure is marginal (dark {black_frac:.1%}, bright {white_frac:.1%}).")
    else:
        expl = f"Exposure acceptable (clipped {clipped:.2%} of retina)."
    score = max(0.0, 1.0 - clipped / max(c["black_bad"], c["white_bad"]))
    return ComponentResult(name="exposure", measurement=clipped,
                           measurement_unit="clipped_fraction", score=min(score, 1.0),
                           status=status, explanation=expl,
                           details={"black_fraction": round(black_frac, 4),
                                    "white_fraction": round(white_frac, 4)})


def _level(frac, borderline, bad):
    if frac >= bad:
        return BAD
    if frac >= borderline:
        return BORDERLINE
    return GOOD


def _rank(s):
    return {GOOD: 0, BORDERLINE: 1, BAD: 2}[s]
