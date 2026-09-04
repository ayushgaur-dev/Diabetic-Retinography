"""Reporting types (SIH26038 Phase 9). Read-only layer: every field must
trace to structured phase outputs. No patient metadata unless supplied."""

from dataclasses import dataclass, field

REPORT_INCOMPLETE = "REPORT_INCOMPLETE"


@dataclass
class ScreeningReportInput:
    image_id: str = "unknown"
    quality: dict = field(default_factory=dict)
    enhancement: dict = field(default_factory=dict)
    grading: dict = field(default_factory=dict)  # grade, raw/cal probs, confidences
    referable: dict = field(default_factory=dict)  # score, threshold
    lesions: dict = field(default_factory=dict)  # type -> {status,count,...}
    vessels: dict = field(default_factory=dict)
    optic_disc: dict = field(default_factory=dict)
    fovea: dict = field(default_factory=dict)
    explainability: dict = field(default_factory=dict)  # consistency, overlap, figure
    triage: dict = field(default_factory=dict)  # decision, reasons, flags, explanation
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "image_id": self.image_id, "quality": self.quality,
            "enhancement": self.enhancement, "grading": self.grading,
            "referable": self.referable, "lesions": self.lesions,
            "vessels": self.vessels, "optic_disc": self.optic_disc,
            "fovea": self.fovea, "explainability": self.explainability,
            "triage": self.triage, "warnings": list(self.warnings),
        }


@dataclass
class ScreeningReport:
    sections: dict = field(default_factory=dict)
    status: str = "COMPLETE"  # COMPLETE | REPORT_INCOMPLETE
    missing_fields: list = field(default_factory=list)
    report_version: str = "1.0.0"
    generator_version: str = "1.0.0"

    def to_dict(self):
        return {"report_version": self.report_version,
                "generator_version": self.generator_version,
                "status": self.status, "missing_fields": list(self.missing_fields),
                "sections": self.sections}
