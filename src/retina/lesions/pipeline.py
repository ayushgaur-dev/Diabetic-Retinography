"""Lesion evidence pipeline (SIH26038 Phase 4D).

detect_lesions(rgb, ...) runs shared context ONCE (FOV, vessels, disc,
fovea) then all four detectors. Context modules are never modified;
each has a neutral fallback + warning when unavailable:
  vessels: None -> no overlap veto (overlap 0), distance -1
  disc: NOT_DETECTED -> no bright-lesion exclusion (explicit warning)
  fovea: NOT_DETECTED -> distances -1, pipeline continues

Returns dict lesion_type -> LesionResult (working-resolution masks;
candidate coordinates mapped to ORIGINAL pixels, x=column y=row).
"""

import time

import cv2
import numpy as np

from src.retina.lesions.config import load_lesion_config
from src.retina.lesions.exudates import (detect_hard_exudates,
                                         detect_soft_exudates)
from src.retina.lesions.hemorrhage import detect_hemorrhages
from src.retina.lesions.microaneurysm import detect_microaneurysms
from src.retina.lesions.postprocessing import cleanup_mask
from src.retina.lesions.preprocessing import lesion_preprocess, to_working_resolution
from src.retina.lesions.types import (DETECTED, LESION_TYPES, LOW_CONFIDENCE,
                                      NOT_DETECTED, LesionCandidate,
                                      LesionResult)


def detect_lesions(rgb, fov_mask=None, vessel_mask=None, disc=None, fovea=None,
                   config=None, quality_config=None, vessel_config=None,
                   disc_config=None, fovea_config=None, lesion_types=None):
    t0 = time.perf_counter()
    _check_rgb(rgb)
    cfg = config or load_lesion_config()
    common = cfg["common"]
    orig_h, orig_w = rgb.shape[:2]
    work, scale = to_working_resolution(rgb, common["working_width"])
    warnings = [f"Working width {work.shape[1]}px (scale {scale:.3f}); "
                "x=column, y=row; candidates mapped back."]

    fov = _resolve_fov(work, fov_mask, quality_config, warnings)
    green, _ = lesion_preprocess(work, fov, common["green_stretch_low"],
                                 common["green_stretch_high"])
    vessel, dist_vessel = _resolve_vessels(work, fov, vessel_mask, vessel_config,
                                           warnings)
    disc_info = _resolve_disc(work, fov, disc, disc_config, warnings, scale)
    fovea_xy = _resolve_fovea(work, disc_info, fovea, fovea_config, warnings, scale)
    disc_excl = _disc_exclusion(work.shape[:2], disc_info, cfg, warnings)

    types = lesion_types or list(LESION_TYPES)
    out = {}
    if "microaneurysm" in types:
        out["microaneurysm"] = _run("microaneurysm", detect_microaneurysms,
                                    (work, fov, vessel, dist_vessel, green,
                                     cfg, common, work.shape[1]),
                                    disc_info, fovea_xy, scale, common, warnings)
    if "hemorrhage" in types:
        out["hemorrhage"] = _run("hemorrhage", detect_hemorrhages,
                                 (work, fov, vessel, dist_vessel, green,
                                  cfg, common, work.shape[1]),
                                 disc_info, fovea_xy, scale, common, warnings)
    if "hard_exudate" in types:
        out["hard_exudate"] = _run("hard_exudate", detect_hard_exudates,
                                   (work, fov, disc_excl, disc_info["xy"],
                                    fovea_xy, cfg, common, work.shape[1]),
                                   disc_info, fovea_xy, scale, common, warnings)
    if "soft_exudate" in types:
        out["soft_exudate"] = _run("soft_exudate", detect_soft_exudates,
                                   (work, fov, disc_excl, disc_info["xy"],
                                    fovea_xy, cfg, common, work.shape[1]),
                                   disc_info, fovea_xy, scale, common, warnings)
    out["_info"] = {"scale": scale, "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                    "warnings": warnings}
    return out


def _run(ltype, fn, args, disc_info, fovea_xy, scale, common, warnings):
    mask, cands = fn(*args)
    mask = cleanup_mask(mask, args[1], fill_holes=True)
    mapped = []
    for i, c in enumerate(cands):
        x0, y0, x1, y1 = c["bbox"]
        cx, cy = c["centroid"]
        feats = dict(c["features"])
        feats["distance_to_disc"] = _dist((cx, cy), disc_info["xy"], args[0].shape[1])
        feats["distance_to_fovea"] = _dist((cx, cy), fovea_xy, args[0].shape[1])
        mapped.append(LesionCandidate(
            candidate_id=f"{ltype[:2].upper()}-{i + 1:03d}", lesion_type=ltype,
            bbox=[x0 / scale, y0 / scale, x1 / scale, y1 / scale],
            centroid_x=cx / scale, centroid_y=cy / scale,
            area=c["area"] / scale ** 2, score=c["score"], features=feats,
        ).to_dict())
    hi, lo = common["status_high_score"], common["status_low_score"]
    best = mapped[0]["score"] if mapped else 0.0
    # Both thresholds honored: strong -> DETECTED, middling -> LOW_CONFIDENCE,
    # weak/empty -> NOT_DETECTED (weak candidates are still listed, never claimed).
    if best >= hi:
        status = DETECTED
    elif mapped and best >= lo:
        status = LOW_CONFIDENCE
    else:
        status = NOT_DETECTED
    return LesionResult(
        lesion_type=ltype, status=status, evidence_mask=mask,
        candidate_count=len(mapped), candidates=mapped,
        total_evidence_area=int((mask > 0).sum()),
        confidence=float(best),
        warnings=[w for w in warnings if "disc" in w.lower() or "vessel" in w.lower()
                  or "fovea" in w.lower()],
    )


def _dist(p, q, width):
    if q is None:
        return -1.0
    return float(np.hypot(p[0] - q[0], p[1] - q[1])) / max(width, 1)


def _resolve_fov(work, fov_mask, quality_config, warnings):
    import cv2 as _cv

    if fov_mask is not None:
        m = np.asarray(fov_mask)
        if m.shape != work.shape[:2]:
            m = _cv.resize(m.astype(np.uint8), (work.shape[1], work.shape[0]),
                           interpolation=_cv.INTER_NEAREST)
        return ((m > 0).astype(np.uint8)) * 255
    from src.quality.config import load_config as load_qconfig
    from src.quality.field_of_view import detect_retinal_field

    qcfg = quality_config or load_qconfig()
    info = detect_retinal_field(work, qcfg)
    warnings.append("FOV from Phase 2 detector (no mask supplied).")
    return info["mask"]


def _resolve_vessels(work, fov, vessel_mask, vessel_config, warnings):
    import cv2 as _cv

    if vessel_mask is not None:
        m = np.asarray(vessel_mask)
        if m.shape != work.shape[:2]:
            m = _cv.resize(m.astype(np.uint8), (work.shape[1], work.shape[0]),
                           interpolation=_cv.INTER_NEAREST)
        v = ((m > 0).astype(np.uint8)) * 255
    else:
        try:
            from src.retina.vessels.config import load_vessel_config
            from src.retina.vessels.pipeline import segment_vessels

            vcfg = vessel_config or load_vessel_config()
            warnings.append("Vessel map from Phase 4A baseline (heuristic context).")
            v = segment_vessels(work, fov, config=vcfg).vessel_mask
        except Exception as e:
            warnings.append(f"Vessel map unavailable ({e}); neutral fallback.")
            return None, None
    dist = _cv.distanceTransform((255 - v), _cv.DIST_L2, 3)
    return v, dist


def _resolve_disc(work, fov, disc, disc_config, warnings, scale):
    """Return dict with xy (working px or None), status, source label."""
    d = disc
    supplied = d is not None
    if d is None:
        try:
            from src.retina.optic_disc.config import load_disc_config
            from src.retina.optic_disc.pipeline import localize_optic_disc

            d = localize_optic_disc(work, fov_mask=fov,
                                    config=disc_config or load_disc_config()).to_dict()
            warnings.append("Optic disc from Phase 4B module (not supplied).")
        except Exception as e:
            warnings.append(f"Optic-disc module failed ({e}); no exclusion.")
            return {"xy": None, "status": NOT_DETECTED, "source": "failed"}
    else:
        warnings.append("Optic disc supplied by caller.")
    if not d.get("detected") or d.get("center") is None:
        warnings.append(f"Optic disc {d.get('status')} — no bright-lesion exclusion.")
        return {"xy": None, "status": d.get("status", NOT_DETECTED),
                "source": "supplied" if supplied else "phase4b"}
    cx, cy = d["center"]
    if supplied:
        cx, cy = cx * scale, cy * scale
    if d.get("status") == LOW_CONFIDENCE:
        warnings.append("Optic disc LOW_CONFIDENCE — weaker exclusion.")
    return {"xy": (float(cx), float(cy)), "radius": float(d.get("radius") or 0),
            "status": d.get("status"), "source": "supplied" if supplied else "phase4b"}


def _resolve_fovea(work, disc_info, fovea, fovea_config, warnings, scale):
    if fovea is not None:
        c = fovea.get("center_x_y")
        warnings.append("Fovea context supplied by caller.")
        return (c[0] * scale, c[1] * scale) if c else None
    try:
        from src.retina.fovea.config import load_fovea_config
        from src.retina.fovea.pipeline import localize_fovea

        disc_arg = None
        if disc_info.get("xy"):
            # Fovea input IS `work`; its supplied-disc contract wants the
            # same frame -> pass working pixels through unchanged.
            x, y = disc_info["xy"]
            disc_arg = {"detected": True, "status": disc_info["status"],
                        "center": [x, y],
                        "radius": disc_info.get("radius", 0),
                        "confidence": 0.5}
        # NOTE: fovea module expects disc in INPUT frame; work IS its input here.
        r = localize_fovea(work, fov_mask=None, disc=disc_arg,
                           config=fovea_config or load_fovea_config()).to_dict()
        warnings.append("Fovea context from Phase 4C module (context only).")
        c = r.get("center_x_y")
        return (c[0], c[1]) if c else None  # working px (caller converts)
    except Exception as e:
        warnings.append(f"Fovea module unavailable ({e}); continuing.")
        return None


def _disc_exclusion(shape, disc_info, cfg, warnings):
    if not disc_info.get("xy") or not disc_info.get("radius"):
        return None
    import cv2 as _cv

    factor = cfg["optic_disc_exclusion"]["dilation_factor"]
    if disc_info.get("status") == LOW_CONFIDENCE:
        factor = cfg["optic_disc_exclusion"]["low_confidence_dilation_factor"]
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    x, y = disc_info["xy"]
    r = disc_info["radius"] * factor
    return ((((xx - x) ** 2 + (yy - y) ** 2) <= r ** 2).astype(np.uint8)) * 255


def _check_rgb(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
            or rgb.dtype != np.uint8):
        raise ValueError("Expected RGB uint8 array.")
