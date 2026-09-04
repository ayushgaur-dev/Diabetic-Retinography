"""Evidence + conflict rules (SIH26038 Phase 8). Lesion evidence is a
safety/context signal: it NEVER overwrites predicted_grade. Conflict
(low grade + strong evidence) routes to human review by default."""

FLAG_NAMES = {
    "microaneurysm": "MICROANEURYSM_EVIDENCE",
    "hemorrhage": "HEMORRHAGE_EVIDENCE",
    "hard_exudate": "HARD_EXUDATE_EVIDENCE",
    "soft_exudate": "SOFT_EXUDATE_EVIDENCE",
}


def evidence_flags(lesion_evidence):
    """Per-type presence flags for detected evidence (any count)."""
    flags = []
    for ltype, ev in (lesion_evidence or {}).items():
        if isinstance(ev, dict) and ev.get("status") == "DETECTED" \
                and (ev.get("candidate_count") or 0) > 0:
            flags.append(FLAG_NAMES.get(ltype, f"{ltype.upper()}_EVIDENCE"))
    n_types = len({f for f in flags})
    if n_types >= 2:
        flags.append("MULTIPLE_LESION_TYPES")
    return flags


def strong_types(lesion_evidence, floors):
    """Types meeting the strong-evidence candidate floors."""
    strong = []
    for ltype, ev in (lesion_evidence or {}).items():
        if not isinstance(ev, dict):
            continue
        floor = floors.get(ltype, 1)
        if ev.get("status") == "DETECTED" and (ev.get("candidate_count") or 0) >= floor:
            strong.append(ltype)
    return strong


def conflict_decision(grade, score, lesion_evidence, cfg):
    """Grade<=1 (and score below referable threshold) + strong lesion
    evidence -> conflict path. Returns (decision_or_None, codes)."""
    pgm = cfg["referable_grade_min"]
    if grade > 1 or score >= cfg["referable_threshold"]:
        return (None, [])
    strong = strong_types(lesion_evidence, cfg["evidence"]["strong_min_candidates"])
    if not strong:
        return (None, [])
    patterns = [tuple(sorted(p)) for p in cfg["evidence"]["escalate_patterns_to_refer"]]
    if tuple(sorted(strong)) in patterns:
        from src.triage.types import REFER

        return (REFER, ["EVIDENCE_MODEL_CONFLICT", "CONFLICT_ESCALATED_TO_REFER"])
    action = cfg["evidence"]["conflict_action"]
    return (action, ["EVIDENCE_MODEL_CONFLICT"])
