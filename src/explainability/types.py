"""Explainability types (SIH26038 Phase 6).

Language contract: 'activation heatmap' / 'spatial agreement' /
'heuristic evidence confidence'. Never 'probability map', never
'caused the prediction', never 'diagnosed'. Raw confidence stays
explicitly UNCALIBRATED.
"""

from dataclasses import dataclass, field

SUPPORTIVE = "SUPPORTIVE"
PARTIALLY_SUPPORTIVE = "PARTIALLY_SUPPORTIVE"
INCONCLUSIVE = "INCONCLUSIVE"
CONFLICTING = "CONFLICTING"


@dataclass
class EvidenceRegion:
    source: str  # gradcam|vessel|optic_disc|fovea|microaneurysm|...
    evidence_type: str
    bbox: list  # [x0,y0,x1,y1], original-frame pixels
    centroid: list  # [x, y]
    area: float
    score: float = 0.0
    overlap_with_gradcam: float = 0.0  # frac of region inside hot mask
    distance_to_fovea: float = -1.0
    distance_to_optic_disc: float = -1.0
    features: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "source": self.source, "evidence_type": self.evidence_type,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "centroid": [round(float(v), 1) for v in self.centroid],
            "area": round(float(self.area), 1),
            "score": round(float(self.score), 3),
            "overlap_with_gradcam": round(float(self.overlap_with_gradcam), 3),
            "distance_to_fovea": round(float(self.distance_to_fovea), 3),
            "distance_to_optic_disc": round(float(self.distance_to_optic_disc), 3),
            "features": self.features, "warnings": list(self.warnings),
        }


@dataclass
class ExplainabilityResult:
    predicted_grade: int
    probabilities: list
    raw_confidence: float  # UNCALIBRATED raw softmax confidence
    explained_class: int
    explained_class_probability: float
    gradcam: dict = field(default_factory=dict)
    fov: dict = field(default_factory=dict)
    vessels: dict = field(default_factory=dict)
    optic_disc: dict = field(default_factory=dict)
    fovea: dict = field(default_factory=dict)
    lesions: dict = field(default_factory=dict)
    evidence_regions: list = field(default_factory=list)
    consistency: dict = field(default_factory=dict)
    faithfulness: dict = field(default_factory=dict)
    summary: str = ""
    warnings: list = field(default_factory=list)
    method: str = "unified_explainability_v1"

    def to_dict(self):
        return {
            "predicted_grade": int(self.predicted_grade),
            "probabilities": [round(float(p), 4) for p in self.probabilities],
            "raw_confidence": round(float(self.raw_confidence), 4),
            "raw_confidence_note": "UNCALIBRATED raw softmax confidence",
            "explained_class": int(self.explained_class),
            "explained_class_probability": round(float(self.explained_class_probability), 4),
            "gradcam": self.gradcam, "fov": self.fov, "vessels": self.vessels,
            "optic_disc": self.optic_disc, "fovea": self.fovea,
            "lesions": self.lesions, "evidence_regions": self.evidence_regions,
            "consistency": self.consistency, "faithfulness": self.faithfulness,
            "summary": self.summary, "warnings": list(self.warnings),
            "method": self.method,
        }
