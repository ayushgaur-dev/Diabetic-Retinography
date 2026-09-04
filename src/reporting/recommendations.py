"""Workflow recommendations: translate-only (SIH26038 Phase 9). No new
triage logic lives here — decision/priority/reasons come verbatim from
the Phase 8 result."""

DEFAULT_WORKFLOW = {
    "UNGRADABLE": "Image recapture recommended.",
    "ROUTINE": "Routine screening workflow recommended.",
    "REFER": "Specialist review recommended.",
    "URGENT_REVIEW": "Urgent specialist review recommended.",
    "TECHNICAL_REVIEW": "Human review recommended.",
}


def recommend(triage_dict, cfg):
    t = triage_dict or {}
    wf = (cfg.get("workflow_text") or {})
    decision = t.get("decision", "UNKNOWN")
    translations = {c: (cfg.get("reason_translations") or {}).get(c, c)
                    for c in t.get("reason_codes", [])}
    return {
        "decision": decision,
        "priority": t.get("priority", "UNKNOWN"),
        "workflow": wf.get(decision, DEFAULT_WORKFLOW.get(
            decision, "Clinician review recommended.")),
        "reason_codes": list(t.get("reason_codes", [])),
        "reason_translations": translations,
        "safety_flags": dict(t.get("safety_flags", {})),
        "explanation": t.get("explanation", ""),
    }
