"""Triage assembly pipeline (SIH26038 Phase 8). Builds TriageInput from
phase outputs (no expensive recompute when artifacts are supplied) and
runs the deterministic decider. Also hosts the offline triage card."""

from src.calibration.temperature_scaling import apply_temperature
from src.triage.config import load_triage_config
from src.triage.decision import decide
from src.triage.types import TriageInput


def build_input(quality=None, grade=None, raw_probs=None, temperature=0.9542,
                lesion_results=None, vessel=None, disc=None, fovea=None,
                consistency=None, enhancement_status="none",
                processing_errors=None, quality_score=0.0):
    """Assemble TriageInput from phase artifacts. quality/grade/probs may be
    dicts (their .to_dict() forms) or plain values; missing optional
    evidence degrades to warnings, never crashes."""
    q = quality or {}
    cal = apply_temperature(raw_probs, temperature).tolist()
    ref_min = 2
    lesions = {}
    for ltype, res in (lesion_results or {}).items():
        d = res.to_dict() if hasattr(res, "to_dict") else res
        lesions[ltype] = {"status": d.get("status", "UNKNOWN"),
                          "candidate_count": d.get("candidate_count", 0),
                          "confidence": d.get("confidence", 0.0)}
    disc_d = disc.to_dict() if hasattr(disc, "to_dict") else (disc or {})
    fovea_d = fovea.to_dict() if hasattr(fovea, "to_dict") else (fovea or {})
    # NOTE (Phase 10A crash-fix): availability is `is not None`. A numpy
    # vessel mask must not be bool()-coerced (ambiguous truth value crash);
    # an empty-but-present mask still means the module ran.
    return TriageInput(
        quality_status=q.get("status", "UNKNOWN"),
        quality_score=float(q.get("overall_score", quality_score)),
        enhancement_status=enhancement_status,
        predicted_grade=int(grade),
        raw_probabilities=[float(v) for v in raw_probs],
        calibrated_probabilities=[float(v) for v in cal],
        calibrated_confidence=float(max(cal)),
        referable_score=float(sum(cal[ref_min:])),
        lesion_evidence=lesions,
        vessel_available=vessel is not None,
        optic_disc_status=str(disc_d.get("status", "UNKNOWN")),
        fovea_status=str(fovea_d.get("status", "UNKNOWN")),
        consistency=str((consistency or {}).get("category", "UNKNOWN")
                        if isinstance(consistency, dict) else consistency),
        processing_errors=list(processing_errors or []),
    )


def triage_from_phases(config=None, **kwargs):
    """Build input and decide. Returns TriageResult (grade immutable)."""
    cfg = config or load_triage_config()
    return decide(build_input(**kwargs), cfg)
