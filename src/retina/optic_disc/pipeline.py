"""Localization decision + pipeline (SIH26038 Phase 4B).

localize_optic_disc(rgb, fov_mask=None, vessel_mask=None, config=None):
  working resize -> FOV (Phase 2 detector unless supplied) -> candidates ->
  vessel mask (Phase 4A at working res unless supplied) -> convergence ->
  scoring -> best candidate -> DETECTED / LOW_CONFIDENCE / NOT_DETECTED.

Coordinates are mapped back to ORIGINAL image pixels. Laterality is never
assumed: no left/right prior exists anywhere here (orientation-independent).
"""

import time

import numpy as np

from src.retina.optic_disc.candidates import generate_candidates
from src.retina.optic_disc.config import load_disc_config
from src.retina.optic_disc.preprocessing import to_working_resolution
from src.retina.optic_disc.scoring import score_candidates
from src.retina.optic_disc.types import (DETECTED, LOW_CONFIDENCE,
                                         NOT_DETECTED, OpticDiscResult)
from src.retina.optic_disc.vessel_convergence import attach_convergence


def localize_optic_disc(rgb, fov_mask=None, vessel_mask=None, config=None,
                        quality_config=None, vessel_config=None):
    t0 = time.perf_counter()
    _check_rgb(rgb)
    cfg = config or load_disc_config()
    orig_h, orig_w = rgb.shape[:2]

    work, scale = to_working_resolution(rgb, cfg["working_resolution"]["max_width"])
    warnings = [f"Processed at working width {work.shape[1]}px "
                f"(scale {scale:.3f}); coordinates mapped back."]

    if fov_mask is None:
        from src.quality.config import load_config as load_qconfig
        from src.quality.field_of_view import detect_retinal_field

        qcfg = quality_config or load_qconfig()
        info = detect_retinal_field(work, qcfg)
        fov = info["mask"]
        warnings.append("FOV from Phase 2 detector (no mask supplied).")
        if not info["plausible"]:
            return _fail(f"no plausible retinal field: {info['reason']}",
                         t0, warnings, cfg)
    else:
        fov = _check_mask(fov_mask, work.shape[:2], resample_from=rgb.shape[:2])

    candidates = generate_candidates(work, fov, cfg)
    if not candidates:
        return _fail("no bright-region candidate passed geometry filters",
                     t0, warnings, cfg, candidate_count=0)

    if vessel_mask is None:
        from src.retina.vessels.config import load_vessel_config
        from src.retina.vessels.pipeline import segment_vessels

        vcfg = vessel_config or load_vessel_config()
        vres = segment_vessels(work, fov, config=vcfg)
        vessel = vres.vessel_mask
        warnings.append("Vessel map from Phase 4A baseline (support evidence only).")
    else:
        vessel = _check_mask(vessel_mask, work.shape[:2], resample_from=rgb.shape[:2])
        if not (vessel > 0).any():
            warnings.append("Supplied vessel mask is empty; convergence is 0 for all.")

    attach_convergence(candidates, vessel, fov, cfg)
    red = work[:, :, 0].astype(np.float32)
    red_in = red[fov > 0]
    ranked = score_candidates(candidates, cfg, float(np.median(red_in)),
                              float(red_in.max()))
    best = ranked[0]
    dec = cfg["decision"]
    if best.score >= dec["detect_threshold"]:
        status, detected = DETECTED, True
    elif best.score >= dec["low_confidence_threshold"]:
        status, detected = LOW_CONFIDENCE, True
        warnings.append("Candidate confidence below the detection threshold — "
                        "verify before downstream use.")
    else:
        return _fail(f"best candidate score {best.score:.3f} below "
                     f"low-confidence threshold {dec['low_confidence_threshold']}",
                     t0, warnings, cfg, candidate_count=len(ranked),
                     cands=ranked)

    radius_work = float(np.sqrt(best.area / np.pi))
    radius = radius_work / scale
    cx, cy = best.centroid_x / scale, best.centroid_y / scale
    ms = (time.perf_counter() - t0) * 1000.0
    return OpticDiscResult(
        detected=detected, status=status, center_x=float(cx), center_y=float(cy),
        radius=float(radius), radius_normalized=float(radius / orig_w),
        bounding_box=[float(cx - radius), float(cy - radius),
                      float(cx + radius), float(cy + radius)],
        confidence=float(best.score), candidate_count=len(ranked),
        selected_candidate_score=float(best.score),
        candidates=[c.to_dict() for c in ranked],
        warnings=warnings + [f"processing_time_ms={ms:.1f}"],
    )


def _fail(reason, t0, warnings, cfg, candidate_count=0, cands=None):
    ms = (time.perf_counter() - t0) * 1000.0
    return OpticDiscResult(
        detected=False, status=NOT_DETECTED, center_x=None, center_y=None,
        radius=None, radius_normalized=None, bounding_box=None, confidence=0.0,
        candidate_count=candidate_count, selected_candidate_score=0.0,
        candidates=[c.to_dict() for c in (cands or [])], warnings=warnings + [
            reason, "No coordinates invented; downstream must handle missing disc.",
            f"processing_time_ms={ms:.1f}"],
    )


def _check_rgb(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
            or rgb.dtype != np.uint8):
        raise ValueError("Expected RGB uint8 array.")


def _check_mask(mask, shape, resample_from=None):
    import cv2

    m = np.asarray(mask)
    if resample_from is not None and m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    if m.shape != shape:
        raise ValueError(f"Mask shape {m.shape} != working shape {shape}.")
    return ((m > 0).astype(np.uint8)) * 255
