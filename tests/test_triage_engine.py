"""Triage-engine tests — SIH26038 Phase 8. Pure unit tests, no datasets,
no model, no network. NOTE: legacy tests/test_triage.py (31 tests for
src/rules/triage.py) is preserved untouched; this file covers the new
src/triage/ screening engine."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.triage.config import DEFAULTS, load_triage_config
from src.triage.decision import decide
from src.triage.evidence_rules import conflict_decision, evidence_flags
from src.triage.explanation import explain
from src.triage.pipeline import build_input
from src.triage.quality_rules import check_input_valid
from src.triage.rules import RULE_PRIORITY
from src.triage.types import (REFER, ROUTINE, TECHNICAL_REVIEW, UNGRADABLE,
                              URGENT_REVIEW, TriageInput)


@pytest.fixture(scope="module")
def cfg():
    return load_triage_config()


def _base(**kw):
    probs = [0.7, 0.15, 0.08, 0.04, 0.03]
    d = dict(quality_status="GOOD", quality_score=0.9, enhancement_status="none",
             predicted_grade=0, raw_probabilities=list(probs),
             calibrated_probabilities=list(probs),
             calibrated_confidence=0.7, referable_score=0.15,
             lesion_evidence={}, vessel_available=True,
             optic_disc_status="DETECTED", fovea_status="DETECTED",
             consistency="INCONCLUSIVE", processing_errors=[])
    d.update(kw)
    return TriageInput(**d)


def _grade_probs(g, conf=0.8):
    p = [(1 - conf) / 4] * 5
    p[g] = conf
    return p


# --- Quality 1-3 --------------------------------------------------------------

def test_ungradable_blocks_everything(cfg):
    t = _base(quality_status="UNGRADABLE", predicted_grade=4,
              referable_score=0.95, calibrated_confidence=0.99)
    d = decide(t, cfg)
    assert d.decision == UNGRADABLE
    assert "IMAGE_UNGRADABLE" in d.reason_codes and "RECAPTURE_REQUIRED" in d.reason_codes
    assert d.priority == "NORMAL"


def test_borderline_success_proceeds_with_warning(cfg):
    t = _base(quality_status="BORDERLINE", enhancement_status="success")
    d = decide(t, cfg)
    assert d.decision == ROUTINE
    assert any("BORDERLINE_IMAGE_ENHANCED" in w for w in d.warnings)


def test_borderline_failed_withholds_grading(cfg):
    t = _base(quality_status="BORDERLINE", enhancement_status="failed",
              predicted_grade=4)
    d = decide(t, cfg)
    assert d.decision == TECHNICAL_REVIEW
    assert "ENHANCEMENT_FAILED" in d.reason_codes


# --- Severity 4-8 ---------------------------------------------------------------

@pytest.mark.parametrize("grade,expected", [(0, ROUTINE), (1, ROUTINE), (2, REFER),
                                           (3, URGENT_REVIEW), (4, URGENT_REVIEW)])
def test_severity_mapping(cfg, grade, expected):
    p = _grade_probs(grade)
    score = sum(p[2:])
    t = _base(predicted_grade=grade, raw_probabilities=p,
              calibrated_probabilities=p, calibrated_confidence=p[grade],
              referable_score=score)
    d = decide(t, cfg)
    assert d.decision == expected, f"grade {grade}"


def test_urgent_keeps_priority_and_flags_uncertainty(cfg):
    p = _grade_probs(4, conf=0.4)
    t = _base(predicted_grade=4, raw_probabilities=p, calibrated_probabilities=p,
              calibrated_confidence=0.4, referable_score=sum(p[2:]))
    d = decide(t, cfg)
    assert d.decision == URGENT_REVIEW and d.priority == "CRITICAL"
    assert "LOW_MODEL_CONFIDENCE" in d.reason_codes


# --- Referable 9-11 ---------------------------------------------------------------

@pytest.mark.parametrize("score,grade,expected", [
    (0.2, 0, ROUTINE), (0.69, 0, ROUTINE), (0.7, 0, REFER), (0.95, 1, REFER)])
def test_referable_threshold(cfg, score, grade, expected):
    p = _grade_probs(grade)
    t = _base(predicted_grade=grade, raw_probabilities=p,
              calibrated_probabilities=p, calibrated_confidence=0.8,
              referable_score=score)
    assert decide(t, cfg).decision == expected


# --- Confidence 12-13 ---------------------------------------------------------------

def test_low_confidence_diverts_routine(cfg):
    t = _base(calibrated_confidence=0.3)
    d = decide(t, cfg)
    assert d.decision == TECHNICAL_REVIEW
    assert "LOW_MODEL_CONFIDENCE" in d.reason_codes
    assert d.safety_flags["model_uncertainty"] is True


def test_normal_confidence_routine(cfg):
    assert decide(_base(calibrated_confidence=0.9), cfg).decision == ROUTINE


# --- Evidence 14-17 -------------------------------------------------------------------

def test_hemorrhage_evidence_flag(cfg):
    lev = {"hemorrhage": {"status": "DETECTED", "candidate_count": 4, "confidence": 0.7}}
    t = _base(predicted_grade=2, lesion_evidence=lev,
              raw_probabilities=_grade_probs(2), calibrated_probabilities=_grade_probs(2),
              calibrated_confidence=0.8, referable_score=0.8)
    d = decide(t, cfg)
    assert "HEMORRHAGE_EVIDENCE" in d.evidence_summary["lesion_flags"]


def test_multiple_lesion_types_flag(cfg):
    lev = {"hemorrhage": {"status": "DETECTED", "candidate_count": 4, "confidence": 0.7},
           "hard_exudate": {"status": "DETECTED", "candidate_count": 5, "confidence": 0.6}}
    assert "MULTIPLE_LESION_TYPES" in evidence_flags(lev)


def test_missing_lesion_evidence_safe(cfg):
    d = decide(_base(lesion_evidence={}), cfg)
    assert d.decision == ROUTINE
    assert d.safety_flags["missing_lesion_evidence"] is True


# --- Conflict 18-20 ---------------------------------------------------------------------

def test_conflict_routes_to_review_and_preserves_grade(cfg):
    lev = {"hemorrhage": {"status": "DETECTED", "candidate_count": 6, "confidence": 0.8}}
    t = _base(predicted_grade=0, lesion_evidence=lev, referable_score=0.1)
    d = decide(t, cfg)
    assert d.decision == TECHNICAL_REVIEW
    assert "EVIDENCE_MODEL_CONFLICT" in d.reason_codes
    assert d.predicted_grade == 0  # immutable
    assert d.safety_flags["evidence_conflict"] is True


def test_conflict_escalation_configurable(cfg):
    c2 = copy.deepcopy(cfg)
    c2["evidence"]["escalate_patterns_to_refer"] = [["hard_exudate", "hemorrhage"]]
    lev = {"hemorrhage": {"status": "DETECTED", "candidate_count": 6, "confidence": 0.8},
           "hard_exudate": {"status": "DETECTED", "candidate_count": 6, "confidence": 0.8}}
    t = _base(predicted_grade=1, lesion_evidence=lev, referable_score=0.2)
    assert decide(t, cfg).decision == TECHNICAL_REVIEW  # default: review
    assert decide(t, c2).decision == REFER  # configured pattern escalates


def test_no_conflict_without_strong_evidence(cfg):
    lev = {"hemorrhage": {"status": "DETECTED", "candidate_count": 1, "confidence": 0.4}}
    assert decide(_base(predicted_grade=0, lesion_evidence=lev,
                        referable_score=0.1), cfg).decision == ROUTINE


# --- Failure 21-25 --------------------------------------------------------------------------

def test_invalid_probabilities(cfg):
    t = _base(raw_probabilities=[0.5, 0.5])
    d = decide(t, cfg)
    assert d.decision == TECHNICAL_REVIEW
    assert "INVALID_INPUT" in d.reason_codes


def test_missing_model_output(cfg):
    t = _base(predicted_grade=-1)
    assert decide(t, cfg).decision == TECHNICAL_REVIEW


def test_missing_optional_vessel(cfg):
    d = decide(_base(vessel_available=False), cfg)
    assert d.decision == ROUTINE
    assert d.safety_flags["missing_vessel_evidence"] is True


def test_missing_disc_and_fovea_warnings(cfg):
    d = decide(_base(optic_disc_status="NOT_DETECTED", fovea_status="UNKNOWN"), cfg)
    assert d.decision == ROUTINE
    assert d.safety_flags["missing_anatomical_landmarks"] is True


def test_processing_errors_divert(cfg):
    t = _base(predicted_grade=0, processing_errors=["vessel crash"])
    d = decide(t, cfg)
    assert d.decision == TECHNICAL_REVIEW
    assert d.safety_flags["processing_failure"] is True


# --- Determinism 26 -------------------------------------------------------------------------------

def test_identical_input_identical_output(cfg):
    t1, t2 = _base(), _base()
    assert decide(t1, cfg).to_dict() == decide(t2, cfg).to_dict()


# --- Immutability 27-28 ---------------------------------------------------------------------------------

def test_model_grade_and_probs_unchanged(cfg):
    p = _grade_probs(1)
    t = _base(predicted_grade=1, raw_probabilities=p, calibrated_probabilities=p,
              lesion_evidence={"hemorrhage": {"status": "DETECTED",
                                             "candidate_count": 9, "confidence": 0.9}},
              referable_score=0.2)
    d = decide(t, cfg)
    assert d.predicted_grade == 1
    assert t.raw_probabilities == p and t.calibrated_probabilities == p


# --- Serialization 29-30 -----------------------------------------------------------------------------------

def test_config_schema(cfg):
    assert cfg["referable_threshold"] == 0.7
    assert cfg["severe_grade_threshold"] == 3
    assert cfg["rule_priority"] == RULE_PRIORITY == DEFAULTS["rule_priority"]
    assert set(cfg["decisions"]) == {"UNGRADABLE", "ROUTINE", "REFER",
                                    "URGENT_REVIEW", "TECHNICAL_REVIEW"}
    import json

    file_cfg = json.load(open(Path(__file__).resolve().parents[1] /
                              "configs" / "triage_config.json"))
    assert file_cfg["low_confidence_threshold"] == cfg["low_confidence_threshold"]


def test_result_schema(cfg):
    d = decide(_base(), cfg).to_dict()
    assert set(d) >= {"decision", "priority", "referable", "reason_codes",
                      "evidence_summary", "safety_flags", "confidence",
                      "predicted_grade", "calibrated_confidence",
                      "referable_score", "warnings",
                      "method", "explanation"}
    assert isinstance(d["explanation"], str) and "Decision:" in d["explanation"]
    assert "diagnos" not in d["explanation"].lower()


def test_explanation_mentions_actual_evidence(cfg):
    lev = {"hemorrhage": {"status": "DETECTED", "candidate_count": 4, "confidence": 0.7}}
    t = _base(predicted_grade=2, lesion_evidence=lev,
              raw_probabilities=_grade_probs(2), calibrated_probabilities=_grade_probs(2),
              calibrated_confidence=0.8, referable_score=0.8)
    text = explain(decide(t, cfg), t)
    assert "HEMORRHAGE_EVIDENCE" in text and "DR grade is 2" in text


def test_build_input_applies_temperature(cfg):
    raw = [0.5, 0.2, 0.15, 0.1, 0.05]
    t = build_input(quality={"status": "GOOD", "overall_score": 0.9}, grade=0,
                    raw_probs=raw, temperature=0.9542)
    assert abs(sum(t.calibrated_probabilities) - 1.0) < 1e-9
    assert t.predicted_grade == 0 and t.quality_status == "GOOD"


def test_rule_priority_documented_order(cfg):
    assert cfg["rule_priority"][0] == "INVALID_INPUT"
    assert cfg["rule_priority"].index("EVIDENCE_CONFLICT") > \
        cfg["rule_priority"].index("REFERABLE")
    # referable score wins over conflict: grade 1 + high score + strong lesions
    lev = {"hemorrhage": {"status": "DETECTED", "candidate_count": 9, "confidence": 0.9}}
    t = _base(predicted_grade=1, lesion_evidence=lev, referable_score=0.85)
    assert decide(t, cfg).decision == REFER


def test_invalid_input_checked_first(cfg):
    t = _base(quality_status="UNGRADABLE", predicted_grade=9)
    d = decide(t, cfg)
    assert d.decision == TECHNICAL_REVIEW and "INVALID_INPUT" in d.reason_codes
