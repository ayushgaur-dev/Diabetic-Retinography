"""Types for fovea localization (SIH26038 Phase 4C).

Coordinate convention (see Phase 4B lesson — impossible to misunderstand):
  x = COLUMN (horizontal, grows right), y = ROW (vertical, grows down),
  in ORIGINAL image pixels. Confidence is an uncalibrated heuristic
  candidate score — never a probability.
"""

from dataclasses import dataclass, field

DETECTED = "DETECTED"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
NOT_DETECTED = "NOT_DETECTED"


@dataclass
class FoveaCandidate:
    x: float  # column (working resolution internally; mapped to original on export)
    y: float  # row (working resolution internally; mapped to original on export)
    disc_relative_distance_dd: float
    disc_relative_angle_deg: float
    temporal_geometry_score: float
    local_darkness_score: float
    local_contrast_score: float
    local_texture_score: float
    vessel_sparsity_score: float
    fov_validity: float
    boundary_distance: float
    score: float = 0.0
    contributions: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "x_y": [round(self.x, 1), round(self.y, 1)],
            "disc_relative_distance_dd": round(self.disc_relative_distance_dd, 3),
            "disc_relative_angle_deg": round(self.disc_relative_angle_deg, 1),
            "temporal_geometry_score": round(self.temporal_geometry_score, 3),
            "local_darkness_score": round(self.local_darkness_score, 3),
            "local_contrast_score": round(self.local_contrast_score, 3),
            "local_texture_score": round(self.local_texture_score, 3),
            "vessel_sparsity_score": round(self.vessel_sparsity_score, 3),
            "fov_validity": round(self.fov_validity, 3),
            "boundary_distance": round(self.boundary_distance, 3),
            "score": round(self.score, 3),
            "contributions": {k: round(float(v), 3) for k, v in self.contributions.items()},
        }


@dataclass
class FoveaResult:
    detected: bool
    status: str  # DETECTED | LOW_CONFIDENCE | NOT_DETECTED
    center_x: float  # column, original pixels (None when not detected)
    center_y: float  # row, original pixels (None when not detected)
    candidate_count: int = 0
    selected_candidate_score: float = 0.0
    confidence: float = 0.0  # heuristic candidate confidence, NOT calibrated
    estimated_region: list = None  # [x0,y0,x1,y1] original pixels (None if undetected)
    laterality: str = "unknown"  # 'left' | 'right' | 'unknown'
    disc_used: bool = False
    method: str = "disc_relative_grid_plus_appearance_and_vessel_sparsity"
    candidates: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "detected": self.detected,
            "status": self.status,
            "center_x_y": ([round(self.center_x, 1), round(self.center_y, 1)]
                           if self.detected else None),
            "candidate_count": self.candidate_count,
            "selected_candidate_score": round(float(self.selected_candidate_score), 3),
            "confidence": round(float(self.confidence), 3),
            "estimated_region": ([round(float(v), 1) for v in self.estimated_region]
                                 if self.estimated_region else None),
            "laterality": self.laterality,
            "disc_used": self.disc_used,
            "method": self.method,
            "candidates": self.candidates,
            "warnings": list(self.warnings),
        }
