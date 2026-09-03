"""Quality gate: integration layer around DR inference (SIH26038 Phase 2).

Flow enforced here:
  image -> quality assessment -> gate -> (model_fn called ONLY if gradable)

- GOOD:       model_fn(image) is called, prediction returned.
- BORDERLINE: model_fn(image) is called (backward compatible), but the result
              carries enhancement_required=True / enhancement_applied=False.
              Real enhancement arrives in Phase 3.
- UNGRADABLE: model_fn is NEVER called. Returns quality result + reasons +
              recapture feedback. An ungradable image must not silently
              produce a DR grade.

model_fn is injected (any callable image -> dict), so the gate is testable
without loading the network and the existing inference code is untouched.
"""

from src.quality.quality_pipeline import assess_image
from src.quality.types import BORDERLINE, GOOD, UNGRADABLE


def screen_image(rgb, model_fn, config=None):
    """Run the quality gate around one DR inference call.

    Returns dict with: quality (QualityResult.to_dict()), gradable (bool),
    model_called (bool), prediction (model_fn output or None),
    blocked_reason (str or None).
    """
    result, info = assess_image(rgb, config=config)
    out = {
        "quality": result.to_dict(),
        "gradable": result.status in (GOOD, BORDERLINE),
        "model_called": False,
        "prediction": None,
        "blocked_reason": None,
    }
    if result.status == UNGRADABLE:
        out["blocked_reason"] = "; ".join(result.reasons) or "Image ungradable."
        return out
    out["model_called"] = True
    out["prediction"] = model_fn(rgb)
    return out


__all__ = ["screen_image", "GOOD", "BORDERLINE", "UNGRADABLE"]
