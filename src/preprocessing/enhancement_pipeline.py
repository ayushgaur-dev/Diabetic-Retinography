"""Enhancement orchestration (SIH26038 Phase 3).

assess -> (GOOD: bypass | UNGRADABLE: reject | BORDERLINE: select -> apply
-> reassess -> safety checks -> accept-or-discard).

screen_enhance_infer() is the Phase 3 analogue of quality.gate.screen_image:
it additionally routes BORDERLINE images through enhancement and hands the
model ONLY the original (GOOD) or the accepted enhanced image — never a
rejected one. No labels are consulted anywhere (no leakage by construction:
the only inputs are pixels + quality verdicts).
"""

import numpy as np

from src.preprocessing.config import load_enhancement_config
from src.preprocessing.enhancement import apply_operations, select_operations
from src.preprocessing.types import (ENHANCEMENT_VERSION, IMPROVED, UNCHANGED,
                                     WORSE, EnhancementResult)
from src.quality.config import load_config as load_quality_config
from src.quality.quality_pipeline import assess_image
from src.quality.types import BAD, BORDERLINE, GOOD, UNGRADABLE

_RANK = {UNGRADABLE: 0, BORDERLINE: 1, GOOD: 2}
_COMPONENTS = ("focus", "illumination", "contrast", "exposure",
               "field_of_view", "retinal_coverage")


def _rank_of(status):
    return _RANK[status]


def _status_rank(s):
    return {"GOOD": 2, "BORDERLINE": 1, "BAD": 0}[s]


def enhance_image(rgb, quality_config=None, enh_config=None):
    """Adaptive enhancement of one RGB uint8 image. Never mutates `rgb`.

    Returns (EnhancementResult, info_dict). GOOD inputs are bypassed,
    UNGRADABLE inputs are rejected — both without touching any pixel.
    """
    qcfg = quality_config if quality_config is not None else load_quality_config()
    ecfg = enh_config if enh_config is not None else load_enhancement_config()
    before, before_info = assess_image(rgb, config=qcfg)
    mask = before_info["mask"]

    if before.status == GOOD:
        return EnhancementResult(enhanced_image=None, before_quality=before.to_dict(),
                                 bypassed=True), before_info
    if before.status == UNGRADABLE:
        return EnhancementResult(enhanced_image=None, before_quality=before.to_dict(),
                                 rejected=True,
                                 warnings=["UNGRADABLE images are not enhanced; recapture required."]), before_info

    op_names, warnings = select_operations(before, rgb, mask, ecfg)
    if not op_names:
        return EnhancementResult(enhanced_image=None, before_quality=before.to_dict(),
                                 warnings=warnings + ["No operation matched the "
                                                      "borderline dimensions."]), before_info

    enhanced, applied = apply_operations(rgb, mask, op_names, ecfg)
    after, _ = assess_image(enhanced, config=qcfg)
    safety_ok, safety_notes = _safety_checks(rgb, enhanced, before, after, ecfg)
    warnings = warnings + safety_notes

    improved_dims = [n for n in _COMPONENTS
                     if _status_rank(after.to_dict()[n]["status"])
                     > _status_rank(before.to_dict()[n]["status"])]
    worsened = any(_status_rank(after.to_dict()[n]["status"])
                   < _status_rank(before.to_dict()[n]["status"]) for n in _COMPONENTS)

    if not safety_ok or _rank_of(after.status) < _rank_of(before.status):
        comparison = WORSE
        successful = False
    elif _rank_of(after.status) > _rank_of(before.status) or (improved_dims and not worsened):
        comparison = IMPROVED
        successful = True
    else:
        comparison = UNCHANGED
        successful = False
        warnings.append("Enhancement did not improve any quality dimension; "
                        "original retained.")

    return EnhancementResult(
        enhanced_image=enhanced if successful else None,
        operations_applied=applied,
        before_quality=before.to_dict(),
        after_quality=after.to_dict(),
        enhancement_successful=successful,
        comparison=comparison,
        changed_quality_dimensions=improved_dims,
        warnings=warnings,
        enhancement_version=ecfg.get("version", ENHANCEMENT_VERSION),
    ), before_info


def _safety_checks(original, enhanced, before, after, ecfg):
    """Return (ok, notes). Any failure discards the enhanced image."""
    s = ecfg["sanity_checks"]
    notes = []
    if not np.isfinite(enhanced.astype(np.float32)).all():
        return False, ["Non-finite pixels introduced; enhanced image discarded."]
    before_clip = before.exposure.measurement
    after_clip = after.exposure.measurement
    if after_clip - before_clip > s["max_extra_clipped_fraction"]:
        notes.append(f"Enhancement added clipping ({before_clip:.2%} -> {after_clip:.2%}); discarded.")
        return False, notes
    g0 = original.astype(np.float32).mean()
    g1 = enhanced.astype(np.float32).mean()
    if abs(g1 - g0) > s["max_mean_shift"]:
        notes.append(f"Excessive brightness shift ({g0:.0f} -> {g1:.0f}); discarded.")
        return False, notes
    for c in range(3):
        d = abs(float(enhanced[:, :, c].mean()) - float(original[:, :, c].mean()))
        if d > s["max_channel_shift"]:
            notes.append(f"Extreme colour shift in channel {c} ({d:.0f}); discarded.")
            return False, notes
    bf = before.field_of_view.measurement
    af = after.field_of_view.measurement
    if bf > 0 and af / bf < s["min_mask_fraction_ratio"]:
        notes.append(f"Retinal mask degraded ({bf:.2f} -> {af:.2f}); discarded.")
        return False, notes
    return True, notes


def screen_enhance_infer(rgb, model_fn, quality_config=None, enh_config=None):
    """Full Phase 3 screening flow. model_fn maps RGB uint8 -> prediction dict.

    GOOD -> model_fn(original). BORDERLINE -> enhance -> reassess ->
      accepted: model_fn(enhanced) | discarded: NO model call, recapture.
    UNGRADABLE -> NO model call, recapture.
    """
    result, _ = enhance_image(rgb, quality_config=quality_config, enh_config=enh_config)
    before_status = result.before_quality["status"]
    out = {
        "initial_status": before_status,
        "enhancement": result.to_dict(),
        "used_image": None,  # 'original' | 'enhanced'
        "model_called": False,
        "prediction": None,
        "final_status": before_status,
        "recapture_feedback": [],
    }
    if result.bypassed:  # GOOD
        out["used_image"] = "original"
        out["model_called"] = True
        out["prediction"] = model_fn(rgb)
        return out
    if result.rejected or not result.enhancement_successful:
        if result.rejected:
            out["recapture_feedback"] = result.before_quality.get("recapture_feedback", [])
        else:
            out["recapture_feedback"] = (
                ["Enhancement did not rescue image quality. Please recapture."]
                + result.warnings)
        out["final_status"] = UNGRADABLE if result.rejected else before_status
        return out
    out["used_image"] = "enhanced"
    out["model_called"] = True
    out["prediction"] = model_fn(result.enhanced_image)
    out["final_status"] = result.after_quality["status"]
    return out
