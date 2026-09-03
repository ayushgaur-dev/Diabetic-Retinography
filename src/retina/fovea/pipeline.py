"""Fovea localization pipeline (SIH26038 Phase 4C).

localize_fovea(rgb, fov_mask=None, disc=None, vessel_mask=None, ...):
  working resize -> FOV (Phase 2 unless supplied) -> disc (supplied dict or
  Phase 4B; recorded) -> laterality -> hypotheses -> grid candidates ->
  appearance + vessel-sparsity features -> weighted scoring -> ambiguity
  check -> DETECTED / LOW_CONFIDENCE / NOT_DETECTED in ORIGINAL pixels.

Disc dependency (documented choice): DETECTED/LOW_CONFIDENCE disc -> full
geometry (LOW adds a verify warning); NOT_DETECTED/missing disc ->
appearance+vessel fallback over the whole FOV with a pseudo-diameter
(FOV width / 8) and NO geometry term — scores run naturally lower, so the
same thresholds stay conservative. Never crashes on a missing disc.
"""

import time

import numpy as np

from src.retina.fovea.anatomical_geometry import (expected_fovea,
                                                  infer_laterality,
                                                  search_points)
from src.retina.fovea.candidates import generate_candidates
from src.retina.fovea.config import load_fovea_config
from src.retina.fovea.preprocessing import extract_green, to_working_resolution
from src.retina.fovea.scoring import score_candidates
from src.retina.fovea.types import (DETECTED, LOW_CONFIDENCE, NOT_DETECTED,
                                    FoveaResult)


def localize_fovea(rgb, fov_mask=None, disc=None, vessel_mask=None, config=None,
                   quality_config=None, disc_config=None, vessel_config=None):
    t0 = time.perf_counter()
    _check_rgb(rgb)
    cfg = config or load_fovea_config()
    orig_h, orig_w = rgb.shape[:2]
    work, scale = to_working_resolution(rgb, cfg["working_resolution"]["max_width"])
    warnings = [f"Processed at working width {work.shape[1]}px "
                f"(scale {scale:.3f}); x=column, y=row; mapped back."]

    fov, ok = _resolve_fov(work, fov_mask, quality_config, warnings)
    if not ok:
        return _fail("no usable retinal field", t0, warnings)

    disc_xy, disc_diam, disc_used, laterality, tsign, confident = _resolve_disc(
        work, fov, disc, disc_config, cfg, warnings, scale)
    vessel = _resolve_vessels(work, fov, vessel_mask, vessel_config, warnings)

    green = extract_green(work)
    if disc_used:
        hyps = _hypotheses(disc_xy, disc_diam, tsign, confident, cfg, warnings)
        points = [p for h in hyps for p in
                  search_points(h[0], disc_diam, fov, cfg)]
        geo_expected = [h[0] for h in hyps]
        anchor_xy = disc_xy
    else:
        points, disc_diam = _fallback_points(fov, cfg, warnings)
        geo_expected = []
        anchor_xy = (float(fov.shape[1]) / 2.0, float(fov.shape[0]) / 2.0)
        cfg = _drop_geometry_weight(cfg)

    cands = generate_candidates(green, fov, anchor_xy, disc_diam, points,
                                vessel, cfg)
    if not cands:
        return _fail("no fovea candidates in search region", t0, warnings)
    ranked = score_candidates(cands, geo_expected, disc_diam, cfg,
                              use_geometry=disc_used)
    best = ranked[0]

    confidence = best.score
    if len(ranked) > 1 and best.score > 0:
        gap = (best.score - ranked[1].score) / best.score
        if gap < cfg["decision"]["ambiguity_margin"]:
            confidence = best.score * (1.0 - cfg["decision"]["ambiguity_penalty"])
            warnings.append(f"Ambiguous top candidates (gap {gap:.2f}); confidence reduced.")
    if not confident and disc_used:
        warnings.append("Laterality ambiguous — both temporal sides searched; "
                        "confidence reflects the ambiguity.")

    dec = cfg["decision"]
    if confidence >= dec["detect_threshold"]:
        status, detected = DETECTED, True
    elif confidence >= dec["low_confidence_threshold"]:
        status, detected = LOW_CONFIDENCE, True
        warnings.append("Candidate confidence below detection threshold — verify "
                        "before downstream use.")
    else:
        return _fail(f"best confidence {confidence:.3f} below "
                     f"{dec['low_confidence_threshold']}", t0, warnings,
                     cands=ranked, laterality=laterality, disc_used=disc_used)

    rr = cfg["geometry"]["region_radius_dd"] * disc_diam / scale
    cx, cy = best.x / scale, best.y / scale
    ms = (time.perf_counter() - t0) * 1000.0
    # Candidate coordinates are mapped back to ORIGINAL pixels so every
    # exposed coordinate shares one frame (x=column, y=row).
    cand_dicts = []
    for c in ranked[:25]:
        cd = c.to_dict()
        cd["x_y"] = [round(c.x / scale, 1), round(c.y / scale, 1)]
        cand_dicts.append(cd)
    return FoveaResult(
        detected=detected, status=status, center_x=float(cx), center_y=float(cy),
        candidate_count=len(ranked), selected_candidate_score=float(best.score),
        confidence=float(confidence),
        estimated_region=[float(cx - rr), float(cy - rr),
                          float(cx + rr), float(cy + rr)],
        laterality=laterality, disc_used=disc_used,
        candidates=cand_dicts,
        warnings=warnings + [f"processing_time_ms={ms:.1f}"],
    )


def _resolve_fov(work, fov_mask, quality_config, warnings):
    import cv2

    if fov_mask is not None:
        m = np.asarray(fov_mask)
        if m.shape != work.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (work.shape[1], work.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        fov = ((m > 0).astype(np.uint8)) * 255
    else:
        from src.quality.config import load_config as load_qconfig
        from src.quality.field_of_view import detect_retinal_field

        qcfg = quality_config or load_qconfig()
        info = detect_retinal_field(work, qcfg)
        fov = info["mask"]
        warnings.append("FOV from Phase 2 detector (no mask supplied).")
        if not info["plausible"]:
            return fov, False
    if (fov > 0).sum() == 0:
        return fov, False
    return fov, True


def _resolve_disc(work, fov, disc, disc_config, cfg, warnings, scale):
    """Return (disc_xy_work, disc_diameter_work, used, laterality, tsign, confident).

    A SUPPLIED disc dict must be in the same pixel frame as the input rgb
    (documented contract) and is rescaled here to working resolution. A
    Phase 4B run on the working image is already in working pixels.
    Internal 4B runs happen on `work`, so nothing is double-scaled."""
    d = disc
    supplied = d is not None
    if d is None:
        from src.retina.optic_disc.config import load_disc_config
        from src.retina.optic_disc.pipeline import localize_optic_disc

        d = localize_optic_disc(work, fov_mask=fov,
                                config=disc_config or load_disc_config()).to_dict()
        warnings.append("Optic disc from Phase 4B module (not supplied).")
    status = d.get("status", NOT_DETECTED)
    # NOTE: OpticDiscResult.to_dict() uses 'center' / 'radius' keys.
    if not d.get("detected") or d.get("center") is None:
        warnings.append(f"Optic disc {status} — appearance/vessel fallback, no geometry.")
        return None, None, False, "unknown", 0, False
    cx, cy = d["center"]
    diam = 2.0 * (d.get("radius") or 0)
    if supplied:
        # Supplied dicts live in the input-rgb frame: rescale to working.
        cx, cy, diam = cx * scale, cy * scale, diam * scale
        warnings.append("Supplied disc mapped from input frame to working resolution.")
    if diam <= 0:
        warnings.append("Disc radius missing — appearance/vessel fallback.")
        return None, None, False, "unknown", 0, False
    if status == LOW_CONFIDENCE:
        warnings.append("Optic disc itself is LOW_CONFIDENCE — geometry weakened.")
    from src.retina.fovea.anatomical_geometry import infer_laterality as infer

    lat, sign, conf = infer((cx, cy), fov, cfg)
    return (cx, cy), diam, True, lat, sign, conf


def _hypotheses(disc_xy, disc_diam, tsign, confident, cfg, warnings):
    if confident:
        return [(expected_fovea(disc_xy, disc_diam, tsign, cfg), tsign)]
    return [(expected_fovea(disc_xy, disc_diam, +1, cfg), +1),
            (expected_fovea(disc_xy, disc_diam, -1, cfg), -1)]


def _fallback_points(fov, cfg, warnings):
    """No disc: sparse grid over the FOV bounding box with a pseudo-diameter
    (FOV width / 8). Geometry term is dropped (see _drop_geometry_weight)."""
    ys, xs = np.nonzero(fov > 0)
    pseudo_dd = max(float(xs.max() - xs.min()) / 8.0, 8.0)
    warnings.append("No disc geometry — whole-FOV fallback search "
                    f"(pseudo-diameter {pseudo_dd:.0f}px, geometry term dropped).")
    step = max(cfg["geometry"]["grid_step_dd"] * 4 * pseudo_dd, 4.0)
    pts = [(float(x), float(y)) for x in np.arange(xs.min(), xs.max(), step)
           for y in np.arange(ys.min(), ys.max(), step)
           if fov[int(y), int(x)] > 0][:2500]
    return pts, pseudo_dd


def _drop_geometry_weight(cfg):
    """Fallback has no expected point: move the geometry weight onto the
    remaining terms proportionally (weights still sum to 1.0). Recorded."""
    import copy

    cfg = copy.deepcopy(cfg)
    w = cfg["scoring"]["weights"]
    rest = w["appearance"] + w["vessel"] + w["fov"]
    w["geometry"] = 0.0
    for k in ("appearance", "vessel", "fov"):
        w[k] = w[k] / rest
    return cfg


def _resolve_vessels(work, fov, vessel_mask, vessel_config, warnings):
    import cv2

    if vessel_mask is not None:
        m = np.asarray(vessel_mask)
        if m.shape != work.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (work.shape[1], work.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        return ((m > 0).astype(np.uint8)) * 255
    try:
        from src.retina.vessels.config import load_vessel_config
        from src.retina.vessels.pipeline import segment_vessels

        vcfg = vessel_config or load_vessel_config()
        warnings.append("Vessel map from Phase 4A baseline (sparsity heuristic only).")
        return segment_vessels(work, fov, config=vcfg).vessel_mask
    except Exception as e:
        warnings.append(f"Vessel map unavailable ({e}); sparsity neutral.")
        return np.zeros(work.shape[:2], dtype=np.uint8)


def _fail(reason, t0, warnings, cands=None, laterality="unknown", disc_used=False):
    ms = (time.perf_counter() - t0) * 1000.0
    return FoveaResult(
        detected=False, status=NOT_DETECTED, center_x=None, center_y=None,
        candidate_count=len(cands or []), selected_candidate_score=0.0,
        confidence=0.0, estimated_region=None, laterality=laterality,
        disc_used=disc_used,
        candidates=[c.to_dict() for c in (cands or [])[:25]],
        warnings=warnings + [reason, "No coordinates invented.",
                             f"processing_time_ms={ms:.1f}"],
    )


def _check_rgb(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
            or rgb.dtype != np.uint8):
        raise ValueError("Expected RGB uint8 array.")
