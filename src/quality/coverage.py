"""Field-of-view and retinal-coverage components (SIH26038 Phase 2).

ENGINEERING HEURISTICS — not clinically validated.
- field_of_view: wraps detect_retinal_field(); statuses the mask fraction.
- retinal_coverage: statuses the fitted-circle completeness, i.e.
  usable retinal area / expected circular field area. A full circular
  field scores ~0.9; a half-cropped field scores ~0.5.
"""

from src.quality.types import BAD, BORDERLINE, GOOD, ComponentResult


def assess_field_of_view(fov_info, cfg):
    c = cfg["field_of_view"]
    frac = fov_info["mask_fraction"]
    if not fov_info["plausible"]:
        return ComponentResult(name="field_of_view", measurement=frac,
                               measurement_unit="mask_fraction", score=0.0,
                               status=BAD,
                               explanation=f"No plausible fundus region: {fov_info['reason']}.",
                               details={"plausible": False})
    if frac >= c["mask_fraction_good"]:
        status = GOOD
        expl = f"Retinal field fills an adequate portion of the image ({frac:.0%})."
    elif frac >= c["mask_fraction_borderline"]:
        status = BORDERLINE
        expl = f"Retinal field is smaller than preferred ({frac:.0%} of image)."
    else:
        status = BAD
        expl = (f"Retinal field of view is insufficient ({frac:.0%} of image). "
                f"Reposition the camera and recapture.")
    score = max(0.0, min(1.0, (frac - c["mask_fraction_borderline"]) /
                         max(c["mask_fraction_good"] - c["mask_fraction_borderline"], 1e-6)))
    return ComponentResult(name="field_of_view", measurement=frac,
                           measurement_unit="mask_fraction", score=score,
                           status=status, explanation=expl,
                           details={"plausible": True,
                                    "center": fov_info["center"],
                                    "radius": round(fov_info["radius"], 1)})


def assess_retinal_coverage(fov_info, cfg):
    c = cfg["retinal_coverage"]
    comp = fov_info["completeness"]
    if not fov_info["plausible"]:
        return ComponentResult(name="retinal_coverage", measurement=0.0,
                               measurement_unit="field_completeness", score=0.0,
                               status=BAD,
                               explanation="Coverage cannot be estimated without a retinal field.",
                               details={"plausible": False})
    if comp >= c["completeness_good"]:
        status = GOOD
        expl = f"Retinal field appears complete ({comp:.0%} of expected circular field)."
    elif comp >= c["completeness_borderline"]:
        status = BORDERLINE
        expl = f"Part of the retinal field appears cut off ({comp:.0%} of expected field)."
    else:
        status = BAD
        expl = (f"Too little of the retinal field is visible ({comp:.0%} of expected field). "
                f"Reposition the camera and recapture.")
    score = max(0.0, min(1.0, (comp - c["completeness_borderline"]) /
                         max(c["completeness_good"] - c["completeness_borderline"], 1e-6)))
    return ComponentResult(name="retinal_coverage", measurement=comp,
                           measurement_unit="field_completeness", score=score,
                           status=status, explanation=expl,
                           details={"plausible": True})
