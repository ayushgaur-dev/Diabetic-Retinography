"""Calibration types (SIH26038 Phase 7). Raw vs calibrated are stored
side-by-side and NEVER merged. Confidence wording: raw = uncalibrated raw
softmax; calibrated = temperature-scaled (still not clinical probability).
"""

from dataclasses import dataclass, field


@dataclass
class CalibrationResult:
    temperature: float
    raw_probabilities: list = field(default_factory=list)
    calibrated_probabilities: list = field(default_factory=list)
    raw_confidence: float = 0.0
    calibrated_confidence: float = 0.0
    raw_prediction: int = 0
    calibrated_prediction: int = 0
    raw_nll: float = 0.0
    calibrated_nll: float = 0.0
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "temperature": round(float(self.temperature), 4),
            "raw_probabilities": [round(float(p), 4) for p in self.raw_probabilities],
            "calibrated_probabilities": [round(float(p), 4) for p in self.calibrated_probabilities],
            "raw_confidence": round(float(self.raw_confidence), 4),
            "calibrated_confidence": round(float(self.calibrated_confidence), 4),
            "raw_prediction": int(self.raw_prediction),
            "calibrated_prediction": int(self.calibrated_prediction),
            "raw_nll": round(float(self.raw_nll), 4),
            "calibrated_nll": round(float(self.calibrated_nll), 4),
            "warnings": list(self.warnings),
        }
