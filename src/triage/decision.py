"""Deterministic rule orchestration (SIH26038 Phase 8). Fixed priority:

INVALID_INPUT > UNGRADABLE > ENHANCEMENT_FAILED > PROCESSING_FAILURE >
HIGH_SEVERITY > REFERABLE > EVIDENCE_CONFLICT > LOW_CONFIDENCE > ROUTINE.

Conflict is evaluated AFTER referable so a score-based referral is never
downgraded by evidence logic; conflict only diverts otherwise-ROUTINE
cases to review. Pure function of (TriageInput, config): no randomness,
no network, no LLM.
"""

from src.triage.confidence_rules import low_confidence_action
from src.triage.evidence_rules import conflict_decision, evidence_flags
from src.triage.quality_rules import check_input_valid, quality_decision
from src.triage.severity_rules import referable_decision, severity_decision
from src.triage.types import (REFER, ROUTINE, TECHNICAL_REVIEW, UNGRADABLE,
                              URGENT_REVIEW, TriageResult)


def decide(t, cfg):
    reasons, warnings, flags = [], [], {}
    evidence_summary = {"lesion_flags": evidence_flags(t.lesion_evidence)}

    bad = check_input_valid(t)
    if bad:
        reasons += ["INVALID_INPUT", "PROCESSING_FAILURE"]
        flags = _flags(t, True, False, False)
        return _result(t, cfg, TECHNICAL_REVIEW, reasons, warnings, flags,
                       evidence_summary, "Input failed validation; safe fallback.")

    qdec, qreasons, qwarn = quality_decision(t)
    reasons += qreasons
    warnings += qwarn
    if qdec == UNGRADABLE:
        flags = _flags(t, False, False, False, quality=True)
        return _result(t, cfg, UNGRADABLE, reasons, warnings, flags,
                       evidence_summary, "Ungradable image; recapture workflow.")
    if qdec == TECHNICAL_REVIEW:
        flags = _flags(t, False, False, False, quality=True)
        return _result(t, cfg, TECHNICAL_REVIEW, reasons, warnings, flags,
                       evidence_summary, "Enhancement failed; grading withheld.")

    if t.processing_errors:
        reasons.append("PROCESSING_FAILURE")
        flags = _flags(t, True, False, False)
        warnings.append(f"Critical processing failures: {t.processing_errors}.")
        return _result(t, cfg, TECHNICAL_REVIEW, reasons, warnings, flags,
                       evidence_summary, "Processing failure; human review.")

    sdec, sreasons = severity_decision(t.predicted_grade, cfg["severe_grade_threshold"])
    if sdec:
        reasons += sreasons + (["REFERABLE_DR_PREDICTION"]
                               if t.predicted_grade >= cfg["referable_grade_min"] else [])
        flags = _flags(t, False, False, False)
        _, lreasons, low_flag = low_confidence_action(
            t.calibrated_confidence, cfg["low_confidence_threshold"], sdec)
        reasons += lreasons  # state stays URGENT; uncertainty only flagged
        flags["model_uncertainty"] = low_flag
        return _result(t, cfg, sdec, reasons, warnings, flags,
                       evidence_summary, "High predicted severity; urgent workflow.")

    rdec, rreasons = referable_decision(t.predicted_grade, t.referable_score,
                                        cfg["referable_grade_min"],
                                        cfg["referable_threshold"])
    if rdec:
        reasons += rreasons
        flags = _flags(t, False, False, False)
        _, lreasons, low_flag = low_confidence_action(
            t.calibrated_confidence, cfg["low_confidence_threshold"], rdec)
        reasons += lreasons
        flags["model_uncertainty"] = low_flag
        return _result(t, cfg, rdec, reasons, warnings, flags,
                       evidence_summary, "Referable screening result; specialist review.")

    cdecl, creasons = conflict_decision(t.predicted_grade, t.referable_score,
                                        t.lesion_evidence, cfg)
    if cdecl:
        reasons += creasons
        flags = _flags(t, False, True, False)
        return _result(t, cfg, cdecl, reasons, warnings, flags,
                       evidence_summary,
                       "Explicit evidence conflicts with low model grade; human review. "
                       "Model grade preserved, not overwritten.")

    provisional = ROUTINE
    over, lreasons, low_flag = low_confidence_action(
        t.calibrated_confidence, cfg["low_confidence_threshold"], provisional)
    reasons += lreasons
    flags = _flags(t, False, False, low_flag)
    if over:
        return _result(t, cfg, over, reasons, warnings, flags,
                       evidence_summary, "Low model confidence; human review.")
    reasons.append("NON_REFERABLE_PREDICTION")
    return _result(t, cfg, ROUTINE, reasons, warnings, flags,
                   evidence_summary, "Non-referable screening result; routine workflow.")


def _flags(t, processing_failure, conflict, uncertainty, quality=False):
    missing_landmarks = (t.optic_disc_status in ("NOT_DETECTED", "UNKNOWN")
                         or t.fovea_status in ("NOT_DETECTED", "UNKNOWN"))
    return {
        "image_quality_issue": quality or t.quality_status in ("UNGRADABLE", "BORDERLINE"),
        "processing_failure": bool(processing_failure or t.processing_errors),
        "model_uncertainty": bool(uncertainty),
        "evidence_conflict": bool(conflict),
        "missing_anatomical_landmarks": bool(missing_landmarks),
        "missing_lesion_evidence": not bool(t.lesion_evidence),
        "missing_vessel_evidence": not bool(t.vessel_available),
    }


def _result(t, cfg, decision, reasons, warnings, flags, evidence_summary, why):
    from src.triage.explanation import explain

    referable = decision in (REFER, URGENT_REVIEW)
    res = TriageResult(
        decision=decision, priority=cfg["priorities"][decision],
        referable=referable, reason_codes=list(reasons),
        evidence_summary=dict(evidence_summary), safety_flags=dict(flags),
        confidence=float(t.calibrated_confidence),
        predicted_grade=int(t.predicted_grade),
        calibrated_confidence=float(t.calibrated_confidence),
        referable_score=float(t.referable_score),
        warnings=list(warnings) + _optional_warnings(t),
        method="evidence_triage_v1",
    )
    res.explanation = explain(res, t)
    return res


def _optional_warnings(t):
    w = []
    if not t.vessel_available:
        w.append("Vessel evidence unavailable (optional; continued safely).")
    if t.optic_disc_status in ("NOT_DETECTED", "UNKNOWN"):
        w.append("Optic-disc localization unavailable (optional; continued safely).")
    if t.fovea_status in ("NOT_DETECTED", "UNKNOWN"):
        w.append("Fovea localization unavailable (optional; continued safely).")
    if not t.lesion_evidence:
        w.append("Lesion evidence unavailable (optional; continued safely).")
    return w
