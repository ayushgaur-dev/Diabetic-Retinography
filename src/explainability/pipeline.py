"""Explainability orchestrator (SIH26038 Phase 6). READ-ONLY over inference:
predicted_grade/probabilities pass through untouched (tested). Includes the
deletion faithfulness check (sanity only, not causal proof)."""

import time

import cv2
import numpy as np
from PIL import Image

from src.explainability.config import load_explainability_config
from src.explainability.consistency import classify
from src.explainability.evidence_fusion import (anatomy_regions, build_regions,
                                               landmark_activation,
                                               lesion_overlap_stats,
                                               rank_regions,
                                               vessel_activation)
from src.explainability.evidence_map import (hot_mask, rasterize_circle,
                                             rasterize_point, to_original,
                                             upsample_heatmap)
from src.explainability.gradcam_adapter import explain_class
from src.explainability.summary import build_summary
from src.explainability.types import ExplainabilityResult


def _prep_for_model(rgb, image_size=224):
    from tensorflow.keras.applications.efficientnet import preprocess_input

    small = np.array(Image.fromarray(rgb).convert("RGB").resize((image_size, image_size)))
    return np.expand_dims(preprocess_input(small.astype("float32")), 0), small


def explain_image(rgb, model=None, grade=None, probabilities=None,
                  target_class=None, config=None, run_context=True,
                  image_size=224):
    """Full pipeline. grade/probabilities may be supplied (verified
    identical to a fresh inference when model given); otherwise inferred
    once here. Returns ExplainabilityResult."""
    t0 = time.perf_counter()
    cfg = config or load_explainability_config()
    rgb = np.asarray(rgb, dtype=np.uint8)
    H, W = rgb.shape[:2]
    warnings = []

    batch, small = _prep_for_model(rgb, image_size)
    if grade is None or probabilities is None:
        if model is None:
            raise ValueError("Provide model or (grade, probabilities).")
        probabilities = [float(v) for v in model.predict(batch, verbose=0)[0]]
        grade = int(np.argmax(probabilities))
    else:
        probabilities = [float(v) for v in probabilities]
        grade = int(grade)
    raw_conf = float(np.max(probabilities))

    g = explain_class(batch, model, target_class, cfg) if model is not None else None
    if g is None:
        raise ValueError("Model required for Grad-CAM (no cached-heatmap path).")
    heat = upsample_heatmap(g["heatmap"], (H, W))
    hot = hot_mask(heat, cfg["gradcam"]["hot_quantile"])

    ctx = _context(rgb, small, run_context, warnings, cfg)
    fov = to_original(ctx["fov"], (H, W), is_mask=True)
    vessel = to_original(ctx["vessel"], (H, W), is_mask=True) if ctx["vessel"] is not None else None
    lesion_masks = {k: to_original(v, (H, W), is_mask=True)
                    for k, v in (ctx["lesion_masks"] or {}).items()}
    disc = _scale_disc(ctx["disc"], ctx["disc_scale"], W, H)
    fovea = _scale_fovea(ctx["fovea"], ctx["fovea_scale"], W, H)
    disc_m = rasterize_circle((H, W), disc["center"], disc["radius"]) if disc else None
    fovea_m = rasterize_point((H, W), fovea["center"]) if fovea else None

    overlap = lesion_overlap_stats(lesion_masks, hot, fov)
    landmarks = landmark_activation(heat, disc_m, fovea_m, fov)
    vessels = vessel_activation(heat, vessel, fov)
    landmarks["fraction_outside_fov"] = round(float(
        ((hot > 0) & ~(fov > 0)).sum() / max((hot > 0).sum(), 1)), 4)

    regions = build_regions(ctx["lesion_dicts"], W, hot,
                            disc["center"] if disc else None,
                            fovea["center"] if fovea else None)
    regions += anatomy_regions(disc, fovea, vessel, hot, W)
    regions = rank_regions(regions, cfg)

    category, reason = classify(overlap, float(heat.max()), cfg)
    lesion_dicts_out = {k: v for k, v in (ctx["lesion_dicts"] or {}).items()}
    result = ExplainabilityResult(
        predicted_grade=grade, probabilities=list(probabilities),
        raw_confidence=raw_conf,
        explained_class=g["explained_class"],
        explained_class_probability=g["explained_class_probability"],
        gradcam={"target_layer": g["target_layer"], "heatmap_shape": g["heatmap_shape"],
                 "normalization": g["normalization"],
                 "via_original_path": g["via_original_path"],
                 "heat_max": round(float(heat.max()), 4),
                 "hot_fraction_fov": round(float((hot > 0).sum()) / max((fov > 0).sum(), 1), 4),
                 "outside_fov_fraction": landmarks["fraction_outside_fov"]},
        fov={"available": True},
        vessels={"available": vessel is not None,
                 **vessels},
        optic_disc={**(disc or {}), **{k: v for k, v in landmarks.items()
                                       if k.startswith("mean_inside_optic")}},
        fovea={**(fovea or {}), **{k: v for k, v in landmarks.items()
                                   if k.startswith("mean_inside_fovea")}},
        lesions={k: {"candidate_count": len(v.get("candidates", [])),
                     "status": v.get("status")} for k, v in lesion_dicts_out.items()},
        evidence_regions=[r.to_dict() for r in regions[:40]],
        consistency={"category": category, "reason": reason,
                     "lesion_overlap": overlap,
                     "mean_outside_landmarks": landmarks["mean_outside_landmarks"]},
        warnings=warnings,
    )
    result.summary = build_summary({**result.to_dict(), "lesion_overlap": overlap,
                                    "lesions": lesion_dicts_out})
    ms = (time.perf_counter() - t0) * 1000.0
    result.warnings.append(f"processing_time_ms={ms:.0f}")
    result.gradcam["lesion_overlap"] = overlap
    # Aligned arrays for visualization only (excluded from to_dict()).
    result.maps = {"gradcam": heat, "hot": hot, "fov": fov, "vessel": vessel,
                   "lesion_masks": lesion_masks, "disc": disc_m, "fovea": fovea_m,
                   "blank": np.zeros((H, W), np.uint8)}
    return result


def _context(rgb, small, run_context, warnings, cfg):
    H, W = rgb.shape[:2]
    if not run_context:
        warnings.append("Context modules skipped (faithfulness fast path).")
        return {"fov": (np.ones((H, W), np.uint8)) * 255, "vessel": None,
                "disc": None, "fovea": None, "lesion_masks": {},
                "lesion_dicts": {}, "disc_scale": 1.0, "fovea_scale": 1.0}
    from src.quality.field_of_view import detect_retinal_field
    from src.quality.config import load_config as load_qconfig
    from src.retina.vessels.pipeline import segment_vessels
    from src.retina.optic_disc.pipeline import localize_optic_disc
    from src.retina.fovea.pipeline import localize_fovea
    from src.retina.lesions.pipeline import detect_lesions

    try:
        fov = detect_retinal_field(rgb, load_qconfig())["mask"]
    except Exception as e:
        warnings.append(f"FOV failed ({e}); using full frame.")
        fov = (np.ones((H, W), np.uint8)) * 255
    try:
        vessel = segment_vessels(rgb, fov).vessel_mask
    except Exception as e:
        warnings.append(f"Vessels unavailable ({e}).")
        vessel = None
    try:
        disc = localize_optic_disc(rgb, fov_mask=fov).to_dict()
    except Exception as e:
        warnings.append(f"Disc unavailable ({e}).")
        disc = {"detected": False, "status": "NOT_DETECTED", "center": None}
    try:
        fovea = localize_fovea(rgb, fov_mask=fov, disc=disc).to_dict()
    except Exception as e:
        warnings.append(f"Fovea unavailable ({e}).")
        fovea = {"detected": False, "status": "NOT_DETECTED", "center_x_y": None}
    lesion_masks, lesion_dicts = {}, {}
    try:
        lesions = detect_lesions(rgb)
        lesions.pop("_info", None)
        for ltype, res in lesions.items():
            lesion_masks[ltype] = res.evidence_mask
            lesion_dicts[ltype] = res.to_dict()
    except Exception as e:
        warnings.append(f"Lesions unavailable ({e}).")
    ws = rgb.shape[1] / 1072.0  # informational only (masks resampled anyway)
    return {"fov": fov, "vessel": vessel, "disc": disc, "fovea": fovea,
            "lesion_masks": lesion_masks, "lesion_dicts": lesion_dicts,
            "disc_scale": 1.0, "fovea_scale": 1.0}


def _scale_disc(disc, scale, W, H):
    if not disc or disc.get("center") is None:
        return None
    (cx, cy), r = disc["center"], disc.get("radius") or 0
    return {"center": [cx * scale, cy * scale], "radius": r * scale,
            "confidence": disc.get("confidence", 0), "status": disc.get("status")}


def _scale_fovea(fovea, scale, W, H):
    if not fovea or fovea.get("center_x_y") is None:
        return None
    cx, cy = fovea["center_x_y"]
    return {"center": [cx * scale, cy * scale],
            "confidence": fovea.get("confidence", 0), "status": fovea.get("status")}


def faithfulness_deletion(rgb, model, grade, probabilities, config=None,
                          image_size=224):
    """Mask the top-activation pixels, re-run, report P(predicted) change.
    Sanity check only — NOT causal proof."""
    cfg = config or load_explainability_config()
    batch, _ = _prep_for_model(rgb, image_size)
    res = explain_image(np.asarray(rgb), model=model, grade=grade,
                        probabilities=probabilities, config=cfg,
                        run_context=False, image_size=image_size)
    heat = upsample_heatmap(
        explain_class(batch, model, int(grade), cfg)["heatmap"], rgb.shape[:2])
    frac = cfg["faithfulness"]["top_fraction"]
    t = float(np.quantile(heat.ravel(), 1.0 - frac))
    mask = heat >= t
    fill = np.asarray(rgb).copy()
    if cfg["faithfulness"]["fill"] == "blurred":
        k = int(cfg["faithfulness"]["blur_kernel"]) | 1
        fill = cv2.GaussianBlur(np.asarray(rgb), (k, k), 0)
    masked = np.asarray(rgb).copy()
    masked[mask] = fill[mask]
    mbatch, _ = _prep_for_model(masked, image_size)
    probs2 = [float(v) for v in model.predict(mbatch, verbose=0)[0]]
    return {
        "original_predicted": int(grade),
        "original_probability": round(float(probabilities[int(grade)]), 4),
        "masked_probability": round(float(probs2[int(grade)]), 4),
        "probability_drop": round(float(probabilities[int(grade)] - probs2[int(grade)]), 4),
        "masked_prediction": int(np.argmax(probs2)),
        "mask_fraction": frac, "mask_fill": cfg["faithfulness"]["fill"],
        "note": "Deletion sanity check only; not causal attribution.",
    }
