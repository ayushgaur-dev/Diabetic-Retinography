"""Types for lesion evidence (SIH26038 Phase 4D).

Vocabulary is deliberate: CANDIDATE / EVIDENCE / HEURISTIC — never
'confirmed', 'diagnosed', or 'probability'. Coordinates: x=column, y=row,
ORIGINAL image pixels on export. Confidence is uncalibrated heuristic
evidence confidence.
"""

from dataclasses import dataclass, field

DETECTED = "DETECTED"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
NOT_DETECTED = "NOT_DETECTED"

LESION_TYPES = ("microaneurysm", "hemorrhage", "hard_exudate", "soft_exudate")


@dataclass
class LesionCandidate:
    candidate_id: str
    lesion_type: str
    bbox: list  # [x0, y0, x1, y1], x=column y=row
    centroid_x: float
    centroid_y: float
    area: float
    score: float = 0.0
    features: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "lesion_type": self.lesion_type,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "centroid_x_y": [round(self.centroid_x, 1), round(self.centroid_y, 1)],
            "area": round(float(self.area), 1),
            "score": round(float(self.score), 3),
            "features": {k: (round(float(v), 3) if isinstance(v, (int, float)) else v)
                         for k, v in self.features.items()},
        }


@dataclass
class LesionResult:
    lesion_type: str
    status: str  # DETECTED | LOW_CONFIDENCE | NOT_DETECTED
    evidence_mask: object  # uint8 0/255, working resolution
    candidate_count: int = 0
    candidates: list = field(default_factory=list)  # LesionCandidate.to_dict()
    total_evidence_area: int = 0
    confidence: float = 0.0  # heuristic evidence confidence, NOT probability
    method: str = "classical_multichannel_candidates"
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "lesion_type": self.lesion_type,
            "status": self.status,
            "candidate_count": self.candidate_count,
            "candidates": self.candidates,
            "total_evidence_area": int(self.total_evidence_area),
            "confidence": round(float(self.confidence), 3),
            "method": self.method,
            "warnings": list(self.warnings),
            "mask_shape": list(self.evidence_mask.shape),
        }
