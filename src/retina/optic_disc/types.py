"""Types for optic-disc localization (SIH26038 Phase 4B).

Localization, NOT segmentation: the output is an approximate region
(center + radius + bbox). Confidence is an ALGORITHMIC candidate score,
never a calibrated probability.
"""

from dataclasses import dataclass, field

DETECTED = "DETECTED"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
NOT_DETECTED = "NOT_DETECTED"


@dataclass
class DiscCandidate:
    centroid_x: float
    centroid_y: float
    area: float
    mean_brightness: float
    contrast: float
    circularity: float
    aspect_ratio: float
    convergence: float
    score: float = 0.0
    contributions: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "centroid": [round(self.centroid_x, 1), round(self.centroid_y, 1)],
            "area": round(self.area, 1),
            "mean_brightness": round(self.mean_brightness, 2),
            "contrast": round(self.contrast, 2),
            "circularity": round(self.circularity, 3),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "convergence": round(self.convergence, 3),
            "score": round(self.score, 3),
            "contributions": {k: round(float(v), 3) for k, v in self.contributions.items()},
        }


@dataclass
class OpticDiscResult:
    detected: bool
    status: str  # DETECTED | LOW_CONFIDENCE | NOT_DETECTED
    center_x: float  # original-image coordinates (None when not detected)
    center_y: float
    radius: float  # equivalent-circle radius, original-image pixels
    radius_normalized: float  # radius / image width
    bounding_box: list  # [x0, y0, x1, y1] original-image pixels
    confidence: float  # algorithmic candidate confidence 0..1 (NOT calibrated)
    candidate_count: int = 0
    selected_candidate_score: float = 0.0
    candidates: list = field(default_factory=list)  # DiscCandidate.to_dict()
    method: str = "bright_candidate_plus_vessel_convergence"
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "detected": self.detected,
            "status": self.status,
            "center": ([round(self.center_x, 1), round(self.center_y, 1)]
                       if self.detected else None),
            "radius": round(self.radius, 1) if self.detected else None,
            "radius_normalized": round(self.radius_normalized, 4) if self.detected else None,
            "bounding_box": [round(float(v), 1) for v in self.bounding_box]
            if self.detected else None,
            "confidence": round(float(self.confidence), 3),
            "candidate_count": self.candidate_count,
            "selected_candidate_score": round(float(self.selected_candidate_score), 3),
            "candidates": self.candidates,
            "method": self.method,
            "warnings": list(self.warnings),
        }
