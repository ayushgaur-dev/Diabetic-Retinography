"""Severity + referable rules (SIH26038 Phase 8). Grade mapping mirrors the
legacy BASE_RULES actions (0,1->routine / 2->refer / 3,4->urgent) reframed
as screening-workflow states. Frozen referable threshold 0.7 preserved."""

from src.triage.types import REFER, URGENT_REVIEW


def severity_decision(grade, severe_threshold):
    if grade >= severe_threshold:
        return (URGENT_REVIEW, ["HIGH_DR_SEVERITY"])
    return (None, [])


def referable_decision(grade, score, grade_min, threshold):
    if grade >= grade_min or score >= threshold:
        return (REFER, ["REFERABLE_DR_PREDICTION"])
    return (None, [])
