"""Consistency analysis: spatial agreement categories (SIH26038 Phase 6).

SUPPORTIVE / PARTIALLY_SUPPORTIVE / INCONCLUSIVE / CONFLICTING describe
whether Grad-CAM and explicit evidence spatially agree — NEVER diagnostic
certainty. CONFLICTING does not mean the model is wrong.
"""

from src.explainability.types import (CONFLICTING, INCONCLUSIVE,
                                      PARTIALLY_SUPPORTIVE, SUPPORTIVE)


def classify(overlap_stats, heat_max, cfg):
    """overlap_stats: per-type lesion_inside_gradcam_fraction (or None).
    heat_max: peak normalized activation (inactivity test)."""
    c = cfg["consistency"]
    fracs = [v["lesion_inside_gradcam_fraction"] for v in overlap_stats.values()
             if v["lesion_inside_gradcam_fraction"] is not None]
    n_lesion = sum(1 for v in overlap_stats.values()
                   if (v["lesion_area"] or 0) > 0)
    if n_lesion == 0:
        if heat_max < c["inactive_max"]:
            return INCONCLUSIVE, "no lesion evidence and no strong activation"
        return INCONCLUSIVE, "activation present but no explicit lesion evidence"
    best = max(fracs) if fracs else 0.0
    if best >= c["high_overlap"]:
        return SUPPORTIVE, f"high spatial agreement (overlap {best:.2f})"
    if best >= c["low_overlap"]:
        return PARTIALLY_SUPPORTIVE, f"moderate spatial agreement ({best:.2f})"
    return CONFLICTING, (f"lesion evidence exists but lies largely outside "
                         f"high Grad-CAM regions ({best:.2f}) — spatial "
                         f"disagreement only, not proof of error")
