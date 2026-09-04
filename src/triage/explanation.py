"""Deterministic explanation generator (SIH26038 Phase 8). Template text
from actual inputs/outputs only — no LLM, no invented findings, screening
(not diagnostic) language throughout."""

GRADE_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR",
                3: "Severe NPDR", 4: "Proliferative DR"}

WORKFLOW = {
    "UNGRADABLE": "Technical recapture workflow recommended.",
    "ROUTINE": "Routine screening workflow recommended.",
    "REFER": "Specialist review recommended.",
    "URGENT_REVIEW": "Urgent specialist review recommended.",
    "TECHNICAL_REVIEW": "Human review recommended.",
}


def explain(res, t):
    L = [f"Decision: {res.decision}", "", "Why:"]
    L.append(f"Predicted DR grade is {t.predicted_grade} "
             f"({GRADE_LABELS.get(t.predicted_grade, '?')}); model output preserved.")
    L.append(f"Calibrated referable probability is {t.referable_score:.2f} "
             f"(threshold 0.70).")
    L.append(f"Calibrated model confidence is {t.calibrated_confidence:.2f} "
             f"(probability representation, not clinical certainty).")
    flags = res.evidence_summary.get("lesion_flags", [])
    if flags:
        L.append("Explicit lesion evidence: " + ", ".join(flags) + ".")
    else:
        L.append("No explicit lesion evidence recorded.")
    L.append(f"Image quality was {t.quality_status}.")
    if t.optic_disc_status not in ("NOT_DETECTED", "UNKNOWN"):
        L.append(f"Optic disc: {t.optic_disc_status}.")
    if t.fovea_status not in ("NOT_DETECTED", "UNKNOWN"):
        L.append(f"Fovea: {t.fovea_status}.")
    L.append("")
    L.append("Workflow:")
    L.append(WORKFLOW.get(res.decision, "Clinician review recommended."))
    if res.warnings:
        L.append("")
        L.append("Warnings:")
        L.extend(f"- {w}" for w in res.warnings[:6])
    return "\n".join(L)
