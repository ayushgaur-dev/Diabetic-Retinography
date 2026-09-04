"""Input validation (SIH26038 Phase 9). Missing required data ->
REPORT_INCOMPLETE with explicit missing fields; nothing fabricated.
Optional evidence degrades to 'unavailable' notes."""

from src.reporting.types import REPORT_INCOMPLETE


def _finite5(probs):
    import math

    if not isinstance(probs, (list, tuple)) or len(probs) != 5:
        return False
    try:
        vals = [float(v) for v in probs]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in vals)


def validate(inp):
    """inp: ScreeningReportInput.to_dict(). Returns (ok, missing_fields)."""
    missing = []
    if not inp.get("quality", {}).get("status"):
        missing.append("quality.status")
    tri = inp.get("triage", {})
    if not tri.get("decision"):
        missing.append("triage.decision")
    graded = tri.get("decision") not in ("UNGRADABLE",)
    g = inp.get("grading", {})
    if graded:
        if g.get("predicted_grade") not in (0, 1, 2, 3, 4):
            missing.append("grading.predicted_grade")
        if not _finite5(g.get("raw_probabilities")):
            missing.append("grading.raw_probabilities[5 finite]")
        else:
            s = sum(float(v) for v in g["raw_probabilities"])
            if abs(s - 1.0) > 0.02:
                missing.append("grading.raw_probabilities sum~=1")
        if g.get("calibrated_confidence") is None:
            missing.append("grading.calibrated_confidence")
    return (not missing), missing


def optional_note(value, present_text, missing_text="unavailable"):
    if value is None or value == {} or value == []:
        return missing_text
    return present_text
