"""Shared types for the image-quality subsystem (SIH26038 Phase 2).

Conventions follow the existing repo: plain stdlib dataclasses, no new
dependencies (no pydantic in requirements). All scores are 0..1 advisory
values; the STATUS field is what the gate acts on.
"""

from dataclasses import dataclass, field


GOOD = "GOOD"
BORDERLINE = "BORDERLINE"
UNGRADABLE = "UNGRADABLE"
BAD = "BAD"  # component-level severe failure (aggregates to UNGRADABLE)

ENGINEERING_HEURISTIC = "ENGINEERING HEURISTIC — not clinically validated"


@dataclass
class ComponentResult:
    """One quality component: raw measurement + normalised score + status + reason."""

    name: str
    measurement: float
    measurement_unit: str
    score: float  # 0..1, advisory
    status: str  # GOOD / BORDERLINE / BAD
    explanation: str
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "name": self.name,
            "measurement": round(float(self.measurement), 4),
            "measurement_unit": self.measurement_unit,
            "score": round(float(self.score), 3),
            "status": self.status,
            "explanation": self.explanation,
            "details": self.details,
        }


@dataclass
class QualityResult:
    """Structured quality verdict for one fundus image."""

    overall_score: float  # 0..1, advisory mean of component scores
    status: str  # GOOD / BORDERLINE / UNGRADABLE
    focus: ComponentResult
    illumination: ComponentResult
    contrast: ComponentResult
    exposure: ComponentResult
    field_of_view: ComponentResult
    retinal_coverage: ComponentResult
    reasons: list = field(default_factory=list)
    recapture_feedback: list = field(default_factory=list)
    enhancement_required: bool = False  # True when BORDERLINE (Phase 3 hook; no enhancement in Phase 2)
    enhancement_applied: bool = False  # Always False in Phase 2

    def to_dict(self):
        return {
            "overall_score": round(float(self.overall_score), 3),
            "status": self.status,
            "focus": self.focus.to_dict(),
            "illumination": self.illumination.to_dict(),
            "contrast": self.contrast.to_dict(),
            "exposure": self.exposure.to_dict(),
            "field_of_view": self.field_of_view.to_dict(),
            "retinal_coverage": self.retinal_coverage.to_dict(),
            "reasons": list(self.reasons),
            "recapture_feedback": list(self.recapture_feedback),
            "enhancement_required": self.enhancement_required,
            "enhancement_applied": self.enhancement_applied,
        }
