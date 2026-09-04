"""Quality + input-validity rules (SIH26038 Phase 8). Highest priority:
INVALID_INPUT, then UNGRADABLE (nothing overrides it), then enhancement."""

from src.triage.types import REFER, ROUTINE, TECHNICAL_REVIEW, UNGRADABLE


def check_input_valid(t):
    """Returns reason string or None. Guards grade range, probability shape/
    values, and presence of a prediction."""
    if t.predicted_grade not in (0, 1, 2, 3, 4):
        return "INVALID_INPUT: predicted_grade out of range"
    for name in ("raw_probabilities", "calibrated_probabilities"):
        p = getattr(t, name)
        if len(p) != 5:
            return f"INVALID_INPUT: {name} must have 5 entries"
        try:
            vals = [float(v) for v in p]
        except (TypeError, ValueError):
            return f"INVALID_INPUT: {name} non-numeric"
        if any(v < 0 or v > 1 for v in vals):
            return f"INVALID_INPUT: {name} outside [0,1]"
    return None


def quality_decision(t):
    """Returns (decision_or_None, reason_codes, warnings)."""
    if t.quality_status == "UNGRADABLE":
        return (UNGRADABLE, ["IMAGE_UNGRADABLE", "RECAPTURE_REQUIRED"],
                ["Technical recapture workflow recommended; grading not attempted."])
    if t.quality_status == "BORDERLINE":
        if t.enhancement_status == "failed":
            return (TECHNICAL_REVIEW, ["ENHANCEMENT_FAILED", "RECAPTURE_REQUIRED"],
                    ["Borderline image could not be enhanced; grading withheld."])
        if t.enhancement_status == "success":
            return (None, [], ["BORDERLINE_IMAGE_ENHANCED: graded enhanced input."])
        return (None, [], ["BORDERLINE_IMAGE_UNENHANCED: graded original input."])
    return (None, [], [])
