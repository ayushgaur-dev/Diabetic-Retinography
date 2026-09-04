"""Confidence rules (SIH26038 Phase 8). Uses Phase 7 CALIBRATED confidence
only — never invents a new score. Low confidence escalates ROUTINE-bound
cases to review; urgent/refer decisions keep their state (with a flag)."""

from src.triage.types import TECHNICAL_REVIEW


def low_confidence_action(calibrated_confidence, threshold, provisional):
    """Returns (decision_override_or_None, codes, flag)."""
    if calibrated_confidence >= threshold:
        return (None, [], False)
    if provisional == "ROUTINE":
        return (TECHNICAL_REVIEW, ["LOW_MODEL_CONFIDENCE"], True)
    return (None, ["LOW_MODEL_CONFIDENCE"], True)
