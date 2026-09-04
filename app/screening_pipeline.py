"""Screening orchestration layer (Phase 10A). UI-side adaptor only: calls
the frozen public APIs of Phases 2-9 in order, never reimplements them.

Pipeline: quality -> enhancement? -> grading -> calibration -> anatomy
-> lesions -> explainability -> triage (-> report via Phase 9).

UNGRADABLE short-circuits BEFORE any model call (tested). Heavy stages
(vessels/disc/fovea/lesions/explainability) can be skipped via
stages={'full_evidence': False} for a quick screen; skipped evidence is
marked missing (warnings), never fabricated.
"""

import hashlib
import time

import numpy as np

STAGE_FULL_EVIDENCE = "full_evidence"


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gate_routing(reason, quality, warnings=None):
    """Quality-gate routing for blocked images (no grading occurred, so no
    Phase 8 decide() runs — there is no grade to preserve). Returns a
    TriageResult-shaped dict with method='quality_gate_routing', never
    presented as a Phase 8 decision."""
    if reason == "UNGRADABLE":
        decision, priority = "UNGRADABLE", "NORMAL"
        codes = ["IMAGE_UNGRADABLE", "RECAPTURE_REQUIRED"]
        workflow = "Image recapture recommended."
    elif reason == "GRADING_FAILED":
        decision, priority = "TECHNICAL_REVIEW", "HIGH"
        codes = ["PROCESSING_FAILURE"]
        workflow = "Human review recommended; grading failed."
    else:
        decision, priority = "TECHNICAL_REVIEW", "HIGH"
        codes = ["ENHANCEMENT_FAILED", "RECAPTURE_REQUIRED"]
        workflow = "Human review recommended; grading withheld."
    return {
        "decision": decision, "priority": priority, "referable": False,
        "reason_codes": codes, "evidence_summary": {"lesion_flags": []},
        "safety_flags": {"image_quality_issue": True, "processing_failure": False,
                         "model_uncertainty": False, "evidence_conflict": False,
                         "missing_anatomical_landmarks": True,
                         "missing_lesion_evidence": True,
                         "missing_vessel_evidence": True},
        "confidence": 0.0, "predicted_grade": None, "calibrated_confidence": 0.0,
        "referable_score": 0.0, "warnings": list(warnings or []),
        "method": "quality_gate_routing",
        "explanation": f"Decision: {decision}\n\nWhy:\nImage quality was "
                       f"{quality.get('status', 'UNKNOWN')} so grading was not "
                       f"attempted.\n\nWorkflow:\n{workflow}",
    }


def _now_ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


def run_screening(rgb, grade_fn=None, temperature=0.9542, stages=None,
                  calibrator=None, on_stage=None):
    """Run the full screening pipeline. rgb: uint8 RGB array.

    grade_fn(batch224) -> probabilities list (injected for tests; default
    uses the frozen EfficientNet via src.evaluation.inference).
    calibrator(probs) -> calibrated list (default: temperature scaling).
    on_stage(name) -> optional progress hook (UI polling only; default
    None preserves exact behavior).
    Returns a plain-dict result with numbers + display-sized arrays.
    """
    from src.preprocessing.enhancement_pipeline import enhance_image
    from src.quality.quality_pipeline import assess_image

    def stage(name):
        if on_stage is not None:
            try:
                on_stage(name)
            except Exception:
                pass

    t0 = time.perf_counter()
    stages = {"full_evidence": True, **(stages or {})}
    out = {"errors": {}, "timings_ms": {}, "warnings": []}
    stage("quality")

    # --- quality (224px reference scale, as calibrated) ---
    t1 = time.perf_counter()
    try:
        from PIL import Image as _I

        q224 = np.array(_I.fromarray(rgb).convert("RGB").resize((224, 224)))
        qres, _ = assess_image(q224)
        out["quality"] = qres.to_dict()
    except Exception as e:
        out["errors"]["quality"] = str(e)[:200]
        out["quality"] = {"status": "ERROR", "overall_score": 0.0}
    out["timings_ms"]["quality"] = _now_ms(t1)

    if out["quality"].get("status") == "UNGRADABLE":
        out["blocked"] = True
        out["blocked_reason"] = "UNGRADABLE"
        out["triage"] = gate_routing("UNGRADABLE", out["quality"],
                                     out["quality"].get("recapture_feedback", []))
        out["timings_ms"]["total"] = _now_ms(t0)
        return out  # model never called (tested)

    # --- enhancement for BORDERLINE ---
    work224, enh_status, enh_info = q224, "none", None
    if out["quality"].get("status") == "BORDERLINE":
        stage("enhancement")
        t1 = time.perf_counter()
        try:
            eres, _ = enhance_image(q224)
            enh_info = eres.to_dict()
            if eres.enhancement_successful:
                work224, enh_status = np.asarray(eres.enhanced_image), "success"
            else:
                enh_status = "failed"
        except Exception as e:
            out["errors"]["enhancement"] = str(e)[:200]
            enh_status = "failed"
        out["timings_ms"]["enhancement"] = _now_ms(t1)
    out["enhancement_status"] = enh_status
    out["enhancement"] = enh_info
    if enh_status == "failed":
        out["blocked"] = True
        out["blocked_reason"] = "ENHANCEMENT_FAILED"
        out["triage"] = gate_routing("ENHANCEMENT_FAILED", out["quality"],
                                     ["Borderline image could not be enhanced."])
        out["timings_ms"]["total"] = _now_ms(t0)
        return out

    # --- grading (exactly once) ---
    stage("grading")
    t1 = time.perf_counter()
    try:
        probs = [float(v) for v in grade_fn(work224)]
        grade = int(np.argmax(probs))
    except Exception as e:
        out["errors"]["grading"] = str(e)[:200]
        out["blocked"] = True
        out["blocked_reason"] = "GRADING_FAILED"
        out["triage"] = gate_routing("GRADING_FAILED", out["quality"],
                                     [out["errors"]["grading"]])
        out["timings_ms"]["total"] = _now_ms(t0)
        return out
    out["timings_ms"]["grading"] = _now_ms(t1)
    out["grade"], out["raw_probabilities"] = grade, probs

    # --- calibration (array math, no model) ---
    if calibrator is None:
        from src.calibration.temperature_scaling import apply_temperature

        def calibrator(p):
            return [float(v) for v in apply_temperature(p, temperature)]
    cal = calibrator(probs)
    out["calibrated_probabilities"] = cal
    out["calibrated_confidence"] = float(max(cal))
    out["referable_score"] = float(sum(cal[2:]))
    out["temperature"] = temperature

    # --- heavy evidence (skippable) ---
    evidence = {"vessel": None, "disc": None, "fovea": None,
                "lesions": None, "explainability": None, "gradcam": None}
    if stages["full_evidence"]:
        stage("evidence")
        evidence = _full_evidence(rgb, out, grade, probs)
        out["warnings"] += evidence.pop("warnings", [])
    else:
        out["warnings"].append("Full evidence workup skipped (quick screen).")
    out.update(evidence)

    # --- triage (verbatim Phase 8, never reimplemented) ---
    stage("triage")
    t1 = time.perf_counter()
    try:
        from src.triage.pipeline import triage_from_phases

        consistency_inner = (out["explainability"] or {}).get("consistency")
        tri = triage_from_phases(
            quality=out["quality"], grade=grade, raw_probs=probs,
            temperature=temperature,
            lesion_results=out["lesions"], vessel=out["vessel"],
            disc=out["disc"], fovea=out["fovea"],
            consistency=consistency_inner,
            enhancement_status=enh_status)
        out["triage"] = tri.to_dict()
    except Exception as e:
        out["errors"]["triage"] = str(e)[:200]
    out["timings_ms"]["triage"] = _now_ms(t1)

    out["blocked"] = False
    out["timings_ms"]["total"] = _now_ms(t0)
    return out


def _full_evidence(rgb, out, grade, probs):
    """Run vessels/disc/fovea/lesions/explainability with per-stage guards."""
    from src.explainability.config import load_explainability_config
    from src.explainability.evidence_map import hot_mask, upsample_heatmap
    from src.explainability.evidence_fusion import lesion_overlap_stats
    from src.explainability.gradcam_adapter import explain_class
    from src.explainability.consistency import classify as consistency_of
    from src.retina.fovea.pipeline import localize_fovea
    from src.retina.optic_disc.pipeline import localize_optic_disc
    from src.quality.config import load_config as load_qconfig
    from src.quality.field_of_view import detect_retinal_field
    from src.retina.lesions.pipeline import detect_lesions
    from src.retina.vessels.pipeline import segment_vessels

    ev = {"vessel": None, "disc": None, "fovea": None, "lesions": None,
          "explainability": None, "gradcam": None, "warnings": []}
    t1 = time.perf_counter()
    try:
        fov = detect_retinal_field(rgb, load_qconfig())["mask"]
        ev["fov"] = fov
    except Exception as e:
        ev["warnings"].append(f"FOV failed: {e}")
        fov = (np.ones(rgb.shape[:2], np.uint8)) * 255
        ev["fov"] = fov
    try:
        ev["vessel"] = segment_vessels(rgb, fov).vessel_mask
    except Exception as e:
        ev["warnings"].append(f"Vessels unavailable: {e}")
    try:
        ev["disc"] = localize_optic_disc(rgb, fov_mask=fov).to_dict()
    except Exception as e:
        ev["warnings"].append(f"Disc unavailable: {e}")
    try:
        ev["fovea"] = localize_fovea(rgb, fov_mask=fov, disc=ev["disc"]).to_dict()
    except Exception as e:
        ev["warnings"].append(f"Fovea unavailable: {e}")
    try:
        les = detect_lesions(rgb)
        les.pop("_info", None)
        ev["lesions"] = {k: v.to_dict() for k, v in les.items()}
    except Exception as e:
        ev["warnings"].append(f"Lesions unavailable: {e}")
    ev["timings_evidence"] = _now_ms(t1)

    # Grad-CAM needs the model: caller passes it via grade_fn? No — the
    # adapter needs the model object. The UI supplies it through the
    # module-level hook below (set_model_for_gradcam) to keep run_screening
    # model-agnostic and unit-testable without TensorFlow.
    model = _MODEL_HOOK.get("model")
    if model is not None:
        try:
            from tensorflow.keras.applications.efficientnet import preprocess_input

            from PIL import Image as _I

            small = np.array(_I.fromarray(rgb).convert("RGB").resize((224, 224)))
            batch = np.expand_dims(preprocess_input(small.astype("float32")), 0)
            ecfg = load_explainability_config()
            g = explain_class(batch, model, grade, ecfg)
            heat = upsample_heatmap(g["heatmap"], rgb.shape[:2])
            hot = hot_mask(heat, ecfg["gradcam"]["hot_quantile"])
            import cv2

            lmasks = {}
            Hh, Ww = rgb.shape[:2]
            for lt, d in (ev["lesions"] or {}).items():
                m = np.zeros((Hh, Ww), np.uint8)
                for c in d.get("candidates", [])[:50]:
                    x0, y0, x1, y1 = (int(v) for v in c["bbox"])
                    x0, y0 = max(x0, 0), max(y0, 0)
                    x1, y1 = min(x1, Ww), min(y1, Hh)
                    if x1 > x0 and y1 > y0:
                        m[y0:y1, x0:x1] = 255
                lmasks[lt] = m
            fov_b = (np.asarray(fov) > 0).astype(np.uint8) * 255
            overlap = lesion_overlap_stats(lmasks, hot, fov_b)
            cat, reason = consistency_of(overlap, float(heat.max()), ecfg)
            hot_fov_frac = float(((hot > 0) & (fov_b > 0)).sum()
                                 / max((fov_b > 0).sum(), 1))
            ev["explainability"] = {"consistency": {"category": cat, "reason": reason},
                                    "lesion_overlap": overlap,
                                    "gradcam": {"target_layer": g["target_layer"],
                                                "hot_fraction_fov": round(hot_fov_frac, 4),
                                                "explained_class": g["explained_class"]}}
            ev["gradcam"] = {"heatmap_small": heat[::4, ::4].tolist(),
                             "explained_class": g["explained_class"]}
        except Exception as e:
            ev["warnings"].append(f"Explainability unavailable: {e}")
    else:
        ev["warnings"].append("Model object not supplied; Grad-CAM skipped.")
    return ev


_MODEL_HOOK = {}


def set_model_for_gradcam(model):
    """UI startup hook: provide the loaded model for Grad-CAM only."""
    _MODEL_HOOK["model"] = model


def should_reuse(stored_hash, new_hash, has_case):
    """Cache-invalidation rule: reuse only when the identical image already
    has a computed case. Any new image invalidates stale results."""
    return bool(has_case) and stored_hash == new_hash


def downscale_for_display(arr, max_width=800):
    from PIL import Image as _I

    img = _I.fromarray(np.asarray(arr).astype(np.uint8))
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    return np.asarray(img)
