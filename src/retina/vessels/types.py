"""Types for vessel segmentation (SIH26038 Phase 4A). Plain stdlib
dataclasses, matching repo convention. Terminology is deliberate: the
classical baseline outputs a response map, NOT a probability map."""

from dataclasses import dataclass, field


@dataclass
class VesselSegmentationResult:
    vessel_mask: object  # uint8 0/255 binary mask (H, W)
    vessel_response: object  # float32 response map 0..1 (H, W) — NOT a probability
    fov_mask: object  # uint8 0/255 evaluation/processing mask actually used
    vessel_density: float  # vessel pixels / FOV pixels (image statistic, NOT a biomarker)
    vessel_area: int  # vessel pixel count inside FOV
    processing_time_ms: float = 0.0
    method: str = "multiscale_tophat_classical"
    fov_source: str = "provided"  # 'provided' | 'phase2_fallback'
    warnings: list = field(default_factory=list)

    def to_dict(self):
        import numpy as np

        return {
            "vessel_density": round(float(self.vessel_density), 4),
            "vessel_area": int(self.vessel_area),
            "processing_time_ms": round(float(self.processing_time_ms), 1),
            "method": self.method,
            "fov_source": self.fov_source,
            "warnings": list(self.warnings),
            "mask_shape": list(self.vessel_mask.shape),
            "response_finite": bool(np.isfinite(
                self.vessel_response.astype(np.float64)).all()),
            "response_nonempty": bool((self.vessel_response > 0).any()),
        }
