"""Evaluation types (SIH26038 Phase 5). Terminology: raw_softmax_confidence
is the max softmax value — explicitly UNCALIBRATED (see calibration_placeholder)."""

from dataclasses import dataclass, field


@dataclass
class PredictionRecord:
    image_id: str
    ground_truth: int
    predicted_grade: int
    probabilities: list  # [P0..P4]
    raw_confidence: float  # max softmax; UNCALIBRATED, not calibrated confidence
    split: str = "test"

    def to_dict(self):
        d = {"image_id": self.image_id, "ground_truth": self.ground_truth,
             "predicted_grade": self.predicted_grade,
             "raw_confidence": round(float(self.raw_confidence), 4),
             "split": self.split}
        for i, p in enumerate(self.probabilities):
            d[f"probability_{i}"] = round(float(p), 4)
        return d
