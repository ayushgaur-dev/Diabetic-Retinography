"""Triage types (SIH26038 Phase 8). Workflow recommendations, never
diagnoses. Model prediction is stored separately and stays immutable."""

from dataclasses import dataclass, field

UNGRADABLE = "UNGRADABLE"
ROUTINE = "ROUTINE"
REFER = "REFER"
URGENT_REVIEW = "URGENT_REVIEW"
TECHNICAL_REVIEW = "TECHNICAL_REVIEW"


@dataclass
class TriageInput:
    quality_status: str = "UNKNOWN"  # GOOD|BORDERLINE|UNGRADABLE|...
    quality_score: float = 0.0
    enhancement_status: str = "none"  # none|success|failed
    predicted_grade: int = -1
    raw_probabilities: list = field(default_factory=list)
    calibrated_probabilities: list = field(default_factory=list)
    calibrated_confidence: float = 0.0
    referable_score: float = 0.0  # calibrated P2+P3+P4
    lesion_evidence: dict = field(default_factory=dict)  # type -> {status,count,confidence}
    vessel_available: bool = False
    optic_disc_status: str = "UNKNOWN"
    fovea_status: str = "UNKNOWN"
    consistency: str = "UNKNOWN"
    processing_errors: list = field(default_factory=list)

    def to_dict(self):
        return {
            "quality_status": self.quality_status, "quality_score": self.quality_score,
            "enhancement_status": self.enhancement_status,
            "predicted_grade": self.predicted_grade,
            "raw_probabilities": list(self.raw_probabilities),
            "calibrated_probabilities": list(self.calibrated_probabilities),
            "calibrated_confidence": self.calibrated_confidence,
            "referable_score": self.referable_score,
            "lesion_evidence": self.lesion_evidence,
            "vessel_available": self.vessel_available,
            "optic_disc_status": self.optic_disc_status,
            "fovea_status": self.fovea_status,
            "consistency": self.consistency,
            "processing_errors": list(self.processing_errors),
        }


@dataclass
class TriageResult:
    decision: str
    priority: str
    referable: bool
    reason_codes: list = field(default_factory=list)
    evidence_summary: dict = field(default_factory=dict)
    safety_flags: dict = field(default_factory=dict)
    confidence: float = 0.0  # calibrated model confidence, NOT clinical certainty
    predicted_grade: int = -1  # immutable copy of model output
    calibrated_confidence: float = 0.0
    referable_score: float = 0.0  # calibrated P2+P3+P4 driving the decision
    warnings: list = field(default_factory=list)
    method: str = "evidence_triage_v1"
    explanation: str = ""

    def to_dict(self):
        return {
            "decision": self.decision, "priority": self.priority,
            "referable": self.referable,
            "reason_codes": list(self.reason_codes),
            "evidence_summary": self.evidence_summary,
            "safety_flags": self.safety_flags,
            "confidence": self.confidence,
            "predicted_grade": self.predicted_grade,
            "calibrated_confidence": self.calibrated_confidence,
            "referable_score": round(float(self.referable_score), 4),
            "warnings": list(self.warnings), "method": self.method,
            "explanation": self.explanation,
        }
