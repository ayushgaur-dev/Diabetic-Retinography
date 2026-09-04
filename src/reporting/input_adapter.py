"""Input adapter: phase artifacts -> ScreeningReportInput (SIH26038 Phase 9).
No inference, no recompute — pure reshaping with explicit Nones."""

from src.reporting.types import ScreeningReportInput


def _d(obj):
    return obj.to_dict() if hasattr(obj, "to_dict") else (obj or {})


def adapt(image_id="unknown", quality=None, enhancement=None, grade=None,
          raw_probs=None, calibrated_probs=None, calibrated_confidence=None,
          referable_score=None, referable_threshold=0.7, lesions=None,
          vessels=None, optic_disc=None, fovea=None, explainability=None,
          triage=None, warnings=None):
    grading = {}
    if grade is not None:
        grading = {"predicted_grade": int(grade),
                   "raw_probabilities": [float(v) for v in raw_probs or []],
                   "calibrated_probabilities": [float(v) for v in calibrated_probs or []],
                   "raw_confidence": float(max(raw_probs)) if raw_probs else None,
                   "calibrated_confidence": calibrated_confidence}
    return ScreeningReportInput(
        image_id=str(image_id), quality=_d(quality), enhancement=_d(enhancement),
        grading=grading,
        referable={"score": referable_score, "threshold": referable_threshold},
        lesions={k: _d(v) for k, v in (lesions or {}).items()},
        vessels=_d(vessels), optic_disc=_d(optic_disc), fovea=_d(fovea),
        explainability=_d(explainability), triage=_d(triage),
        warnings=list(warnings or []),
    )


def adapt_triage_result(triage_result, quality=None, grading_extra=None):
    """Shortcut when a Phase 8 TriageResult dict already bundles grade,
    confidences, and reason codes."""
    t = _d(triage_result)
    return ScreeningReportInput(
        quality=_d(quality), grading=dict(grading_extra or {}),
        referable={"score": t.get("referable_score"),
                   "threshold": 0.7},
        triage=t, warnings=list(t.get("warnings", [])),
    )
