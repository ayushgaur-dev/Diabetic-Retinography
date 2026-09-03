"""Quality assessment pipeline (SIH26038 Phase 2).

assess_image() is the single entry point: RGB uint8 array in, QualityResult
out. Runs field-of-view detection first (other components measure INSIDE
the mask), then focus / illumination / contrast / exposure, then coverage,
then aggregation + recapture feedback.
"""

import time

import cv2
import numpy as np

from src.quality.config import load_config
from src.quality.contrast import assess_contrast
from src.quality.coverage import assess_field_of_view, assess_retinal_coverage
from src.quality.exposure import assess_exposure
from src.quality.field_of_view import detect_retinal_field
from src.quality.focus import assess_focus
from src.quality.illumination import assess_illumination
from src.quality.quality_score import aggregate
from src.quality.types import BORDERLINE, GOOD, UNGRADABLE, QualityResult

RECAPTURE_MESSAGES = {
    "focus": "Image appears out of focus. Keep the camera steady and recapture.",
    "illumination": "Lighting is unsuitable. Adjust illumination so the retina is evenly lit and recapture.",
    "contrast": "Image lacks usable contrast. Check focus and illumination, then recapture.",
    "exposure": "Exposure is unsuitable. Adjust illumination and recapture.",
    "field_of_view": "Retinal field of view is insufficient. Reposition the camera and recapture.",
    "retinal_coverage": "Too little of the retina is visible. Reposition the camera and recapture.",
}


def validate_image(rgb):
    if not isinstance(rgb, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(rgb).__name__}.")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}.")
    if rgb.shape[0] < 32 or rgb.shape[1] < 32:
        raise ValueError(f"Image too small for quality assessment: {rgb.shape}.")
    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got {rgb.dtype}.")
    return rgb


def assess_image(rgb, config=None):
    """Assess one fundus image. Returns (QualityResult, info_dict with mask + timing)."""
    t0 = time.perf_counter()
    rgb = validate_image(rgb)
    cfg = config if config is not None else load_config()

    fov_info = detect_retinal_field(rgb, cfg)
    mask = fov_info["mask"]
    gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_f32 = gray_u8.astype(np.float32)

    components = {
        "focus": assess_focus(gray_u8, mask, cfg),
        "illumination": assess_illumination(gray_f32, mask, cfg),
        "contrast": assess_contrast(gray_f32, mask, cfg),
        "exposure": assess_exposure(gray_f32, mask, cfg),
        "field_of_view": assess_field_of_view(fov_info, cfg),
        "retinal_coverage": assess_retinal_coverage(fov_info, cfg),
    }
    status, overall, reasons = aggregate(components, cfg)
    feedback = _recapture_feedback(components, status)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = QualityResult(
        overall_score=overall, status=status,
        focus=components["focus"], illumination=components["illumination"],
        contrast=components["contrast"], exposure=components["exposure"],
        field_of_view=components["field_of_view"],
        retinal_coverage=components["retinal_coverage"],
        reasons=reasons, recapture_feedback=feedback,
        enhancement_required=(status == BORDERLINE),
        enhancement_applied=False,
    )
    info = {"mask": mask, "fov": fov_info, "elapsed_ms": elapsed_ms}
    return result, info


def _recapture_feedback(components, status):
    if status == GOOD:
        return []
    msgs = []
    for name, comp in components.items():
        if comp.status != GOOD and name in RECAPTURE_MESSAGES:
            msgs.append(RECAPTURE_MESSAGES[name])
    if status == UNGRADABLE:
        msgs.append("This image cannot be graded. Please recapture before screening.")
    elif status == BORDERLINE:
        msgs.append("Image quality is marginal and enhancement will be required (Phase 3).")
    return msgs
