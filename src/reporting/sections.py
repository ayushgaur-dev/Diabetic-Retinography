"""Section builders: structured dicts per report section (SIH26038 Phase 9).
Each builder reads validated input only."""

from src.reporting import evidence as EV
from src.reporting import findings as F
from src.reporting import recommendations as REC


def _round_list(vals, nd=4):
    try:
        return [round(float(v), nd) for v in vals]
    except (TypeError, ValueError):
        return vals


def summary(inp, triage_rec, cfg):
    g = inp.get("grading", {})
    ref = inp.get("referable", {})
    labels = cfg.get("grade_labels", {})
    grade = g.get("predicted_grade")
    return {
        "image_quality": inp.get("quality", {}).get("status", "UNKNOWN"),
        "model_prediction": (f"{labels.get(str(grade), '?')} (Grade {grade})"
                             if grade in (0, 1, 2, 3, 4) else "not assessed"),
        "predicted_grade": grade,
        "calibrated_confidence": g.get("calibrated_confidence"),
        "referable_probability": ref.get("score"),
        "workflow_recommendation": triage_rec.get("decision", "UNKNOWN"),
        "priority": triage_rec.get("priority", "UNKNOWN"),
    }


def quality_section(inp):
    q = inp.get("quality", {})
    out = {"status": q.get("status", "UNKNOWN"),
           "score": q.get("overall_score"),
           "components": {k: (q.get(k, {}) or {}).get("status", "UNKNOWN")
                          for k in ("focus", "illumination", "contrast",
                                    "exposure", "field_of_view",
                                    "retinal_coverage")},
           "reasons": list(q.get("reasons", [])),
           "recapture_feedback": list(q.get("recapture_feedback", []))}
    if out["status"] == "UNGRADABLE":
        out["notice"] = ("Image quality is UNGRADABLE: no DR grade can be "
                         "trusted for this image; recapture is recommended.")
    return out


def enhancement_section(inp):
    e = inp.get("enhancement", {})
    if not e:
        return {"status": "not required", "operations": [], "result": None}
    if e.get("bypassed"):
        return {"status": "not required", "operations": [], "result": None}
    if e.get("rejected"):
        return {"status": "not attempted (ungradable input)", "operations": [],
                "result": None}
    return {"status": "applied" if e.get("enhancement_successful") else "failed",
            "operations": list(e.get("operations_applied", [])),
            "before": (e.get("before_quality") or {}).get("status"),
            "after": (e.get("after_quality") or {}).get("status"),
            "note": "Enhancement adjusts image quality; it does not change "
                    "the screening assessment."}


def grading_section(inp, cfg):
    g = inp.get("grading", {})
    labels = cfg.get("grade_labels", {})
    nd = cfg.get("formatting", {}).get("probability_decimals", 4)
    if g.get("predicted_grade") not in (0, 1, 2, 3, 4):
        return {"assessed": False,
                "notice": "DR grade was not assessed (grading blocked or unavailable)."}
    return {
        "assessed": True,
        "predicted_grade": g["predicted_grade"],
        "grade_label": labels.get(str(g["predicted_grade"]), "?"),
        "sentence": F.grade_sentence(g["predicted_grade"], {int(k): v for k, v in labels.items()}),
        "raw_probabilities": _round_list(g.get("raw_probabilities", []), nd),
        "calibrated_probabilities": _round_list(g.get("calibrated_probabilities", []), nd),
        "calibrated_confidence": g.get("calibrated_confidence"),
        "confidence_sentence": F.confidence_sentence(g["calibrated_confidence"])
        if g.get("calibrated_confidence") is not None else None,
    }


def referable_section(inp):
    ref = inp.get("referable", {})
    score, thr = ref.get("score"), ref.get("threshold", 0.7)
    cls = "REFERABLE" if score is not None and score >= thr else "NON_REFERABLE"
    return {"score": score, "threshold": thr, "classification": cls,
            "sentence": F.referable_sentence(score, thr),
            "definition": "Referable DR = predicted Grade >= 2; score = P2+P3+P4.",
            "threshold_note": "Threshold preserved from Phase 5; not claimed clinically validated."}


def lesion_section(inp, cfg):
    rows = EV.lesion_rows(inp.get("lesions", {}))
    out = []
    for r in rows:
        d = (inp.get("lesions", {}) or {}).get(r["lesion_type"], {})
        cands = d.get("candidates", [])[:10]
        out.append({**r, "sentence": F.lesion_sentence(
            r["lesion_type"], r["status"], r["candidate_count"]),
            "top_candidates": [
                {"bbox": c.get("bbox"), "centroid": c.get("centroid_x_y"),
                 "score": c.get("score")} for c in cands]})
    return {"lesions": out, "limitation": cfg.get("lesion_limitation", "")}


def anatomy_section(inp):
    od, fo, ves = inp.get("optic_disc", {}), inp.get("fovea", {}), inp.get("vessels", {})
    rows = EV.anatomy_rows(od, fo, ves)
    sentences = [
        F.landmark_sentence("optic disc", od.get("status"), od.get("confidence")),
        F.landmark_sentence("fovea", fo.get("status"), fo.get("confidence")),
        "Vessel map: available." if ves else "Vessel map: unavailable.",
    ]
    return {"structures": rows, "sentences": sentences}


def explainability_section(inp):
    ex = inp.get("explainability", {})
    cons = ex.get("consistency", {})
    overlap = ex.get("lesion_overlap", {})
    return {
        "gradcam": {"explained_class": ex.get("explained_class"),
                    "layer": (ex.get("gradcam") or {}).get("target_layer"),
                    "hot_fraction_fov": (ex.get("gradcam") or {}).get("hot_fraction_fov")},
        "consistency": cons.get("category", "UNKNOWN"),
        "consistency_reason": cons.get("reason", ""),
        "lesion_agreement": {k: v.get("lesion_inside_gradcam_fraction")
                             for k, v in overlap.items()},
        "figure": ex.get("figure"),
        "note": ("Detected lesion evidence overlapping Grad-CAM regions is "
                 "spatial agreement, not proof the lesion caused the prediction."),
    }


def triage_section(inp, cfg):
    rec = REC.recommend(inp.get("triage", {}), cfg)
    return rec


def limitations_section(cfg):
    return list(cfg.get("limitations", []))


def provenance_section(cfg):
    return dict(cfg.get("provenance", {}))
