"""Deterministic explanation summary (SIH26038 Phase 6). Template text from
actual outputs only — no LLM, no invented findings, 'predicted' language."""

GRADE_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR",
                3: "Severe NPDR", 4: "Proliferative DR"}


def build_summary(result):
    """result: ExplainabilityResult.to_dict(). Returns plain-text summary."""
    L = []
    L.append(f"Prediction: Grade {result['predicted_grade']} "
             f"({GRADE_LABELS.get(result['predicted_grade'], '?')})")
    L.append(f"Raw model confidence: {result['raw_confidence']:.2f} (uncalibrated)")
    L.append(f"Explained class: {result['explained_class']} "
             f"(p={result['explained_class_probability']:.2f})")
    g = result.get("gradcam", {})
    L.append(f"Model attention: peak {g.get('heat_max', 0):.2f}, "
             f"{g.get('hot_fraction_fov', 0):.1%} of FOV strongly activated, "
             f"{g.get('outside_fov_fraction', 0):.1%} of activation outside FOV")
    for ltype, s in result.get("lesion_overlap", {}).items():
        n = result.get("lesions", {}).get(ltype, {}).get("candidate_count", 0)
        L.append(f"Explicit evidence [{ltype}]: {n} candidates; "
                 f"spatial agreement with model attention "
                 f"{s.get('lesion_inside_gradcam_fraction')}")
    ana = []
    if result.get("optic_disc", {}).get("center") is not None:
        ana.append("optic disc localized")
    if result.get("fovea", {}).get("center_x_y") is not None:
        ana.append("fovea localized")
    ana.append("vessel map available" if result.get("vessels", {}).get("available")
               else "vessel map unavailable")
    L.append("Anatomical context: " + "; ".join(ana))
    cc = result.get("consistency", {})
    L.append(f"Spatial consistency: {cc.get('category')} — {cc.get('reason')}")
    L.append("Note: spatial agreement is not causal proof; lesion evidence is "
             "heuristic; confidence is uncalibrated.")
    return "\n".join(L)
