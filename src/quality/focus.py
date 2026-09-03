"""Focus / blur assessment via Laplacian variance (SIH26038 Phase 2).

ENGINEERING HEURISTIC — not clinically validated. Measured inside the
retinal mask (background excluded) on uint8 gray. Score is log-normalised
between config log_lo..log_hi; status from good_var / borderline_var.
"""

import cv2
import numpy as np

from src.quality.types import BAD, BORDERLINE, GOOD, ComponentResult


def assess_focus(gray_u8, mask, cfg):
    c = cfg["focus"]
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    inside = lap[mask > 0]
    var = float(inside.var()) if inside.size else 0.0
    score = _normalise(var, c["log_lo"], c["log_hi"])
    if var >= c["good_var"]:
        status, expl = GOOD, f"Focus adequate (sharpness {var:.0f})."
    elif var >= c["borderline_var"]:
        status, expl = (BORDERLINE,
                        f"Focus is below the preferred level (sharpness {var:.0f}).")
    else:
        status, expl = (BAD,
                        f"Image appears out of focus (sharpness {var:.0f}).")
    return ComponentResult(name="focus", measurement=var,
                           measurement_unit="laplacian_variance",
                           score=score, status=status, explanation=expl,
                           details={"good_var": c["good_var"],
                                    "borderline_var": c["borderline_var"]})


def _normalise(var, lo, hi):
    import math

    v = math.log10(var + 1.0)
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))
