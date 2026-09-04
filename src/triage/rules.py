"""Canonical rule-priority registry (SIH26038 Phase 8).

Order is load-bearing and tested: INVALID_INPUT > UNGRADABLE >
ENHANCEMENT_FAILED > PROCESSING_FAILURE > HIGH_SEVERITY > REFERABLE >
EVIDENCE_CONFLICT > LOW_CONFIDENCE > ROUTINE.

Conflict sits AFTER referable by design: a score-based referral is never
downgraded by evidence logic; conflict only diverts otherwise-ROUTINE
(low-grade, low-score) cases to human review. Severity sits before
referable so grades 3/4 always escalate to URGENT even when evidence is
thin. decision.py implements exactly this order (asserted in tests).
"""

RULE_PRIORITY = ["INVALID_INPUT", "UNGRADABLE", "ENHANCEMENT_FAILED",
                 "PROCESSING_FAILURE", "HIGH_SEVERITY", "REFERABLE",
                 "EVIDENCE_CONFLICT", "LOW_CONFIDENCE", "ROUTINE"]

RULE_DESCRIPTIONS = {
    "INVALID_INPUT": "Prediction/probabilities fail validation -> TECHNICAL_REVIEW.",
    "UNGRADABLE": "Quality UNGRADABLE blocks everything -> UNGRADABLE + recapture.",
    "ENHANCEMENT_FAILED": "BORDERLINE rescue failed -> TECHNICAL_REVIEW, grading withheld.",
    "PROCESSING_FAILURE": "Critical component errors -> TECHNICAL_REVIEW.",
    "HIGH_SEVERITY": "Grade >= severe threshold -> URGENT_REVIEW (screening priority).",
    "REFERABLE": "Grade >= 2 or calibrated score >= 0.7 -> REFER.",
    "EVIDENCE_CONFLICT": "Low grade/score + strong lesion evidence -> review (grade kept).",
    "LOW_CONFIDENCE": "Calibrated confidence below threshold diverts ROUTINE -> review.",
    "ROUTINE": "Non-referable, acceptable quality, no conflict -> ROUTINE.",
}
