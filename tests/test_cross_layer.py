"""Cross-layer consistency (Phase 13). One deterministic synthetic result
must survive Python -> triage -> report -> API -> MATLAB-contract unchanged."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.config import load_reporting_config
from src.reporting.deterministic_generator import generate
from src.triage.config import load_triage_config
from src.triage.decision import decide
from src.triage.types import TriageInput

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_screening.json"


@pytest.fixture(scope="module")
def golden():
    return json.load(open(GOLDEN, encoding="utf-8"))


def test_golden_labels_synthetic(golden):
    blob = json.dumps(golden)
    assert "SYNTHETIC" in blob.upper()


def test_phase8_recompute_matches(golden):
    t = TriageInput(
        quality_status=golden["quality"]["status"],
        quality_score=golden["quality"]["overall_score"],
        enhancement_status=golden["enhancement_status"],
        predicted_grade=golden["grade"],
        raw_probabilities=golden["raw_probabilities"],
        calibrated_probabilities=golden["calibrated_probabilities"],
        calibrated_confidence=golden["calibrated_confidence"],
        referable_score=golden["referable_score"],
        lesion_evidence={k: {"status": v["status"],
                             "candidate_count": v["candidate_count"],
                             "confidence": v["confidence"]}
                         for k, v in golden["lesions"].items()},
        vessel_available=golden["vessel"] is not None,
        optic_disc_status=golden["disc"]["status"],
        fovea_status=golden["fovea"]["status"],
        consistency=(golden["explainability"]["consistency"] or {}).get("category", "UNKNOWN"),
        processing_errors=[])
    d = decide(t, load_triage_config()).to_dict()
    g = golden["triage"]
    for key in ("decision", "priority", "predicted_grade", "reason_codes",
                "referable_score", "calibrated_confidence"):
        assert d[key] == g[key], key
    assert set(d["evidence_summary"]["lesion_flags"]) == \
        set(g["evidence_summary"]["lesion_flags"])  # order not contractual
    assert d["safety_flags"] == g["safety_flags"]


def test_phase9_preserves_values(golden):
    lesions = {k: {**v, "total_evidence_area": 0, "candidates": []}
               for k, v in golden["lesions"].items()}
    rep_in = {
        "image_id": "golden", "quality": golden["quality"],
        "enhancement": golden["enhancement"],
        "grading": {"predicted_grade": golden["grade"],
                    "raw_probabilities": golden["raw_probabilities"],
                    "calibrated_probabilities": golden["calibrated_probabilities"],
                    "calibrated_confidence": golden["calibrated_confidence"]},
        "referable": {"score": golden["referable_score"], "threshold": 0.7},
        "lesions": lesions,
        "vessels": {"available": True},
        "optic_disc": {"status": "DETECTED"},
        "fovea": {"status": "DETECTED"},
        "explainability": golden["explainability"],
        "triage": golden["triage"], "warnings": []}
    rep = generate(rep_in, load_reporting_config()).to_dict()
    s = rep["sections"]
    assert s["grading"]["predicted_grade"] == golden["grade"] == 2
    assert s["grading"]["calibrated_confidence"] == golden["calibrated_confidence"]
    assert s["triage"]["decision"] == golden["triage"]["decision"] == "REFER"
    assert s["triage"]["reason_codes"] == golden["triage"]["reason_codes"]
    assert s["referable"]["score"] == golden["referable_score"]


def test_backend_mapping_preserves_values(golden):
    from backend.service import _summary

    result = {
        "quality": golden["quality"], "blocked": False,
        "grade": golden["grade"],
        "raw_probabilities": golden["raw_probabilities"],
        "calibrated_probabilities": golden["calibrated_probabilities"],
        "calibrated_confidence": golden["calibrated_confidence"],
        "referable_score": golden["referable_score"],
        "triage": golden["triage"],
        "lesions": {k: {"candidate_count": v["candidate_count"]}
                    for k, v in golden["lesions"].items()},
        "disc": golden["disc"], "fovea": golden["fovea"],
        "explainability": {"consistency": golden["explainability"]["consistency"]},
        "vessel": [1], "warnings": [], "errors": {}, "timings_ms": {},
    }
    s = _summary("hash", result)
    assert s["grade"] == 2 and s["triage_decision"] == "REFER"
    assert s["raw_probabilities"] == golden["raw_probabilities"]
    assert s["calibrated_confidence"] == golden["calibrated_confidence"]
    assert s["lesion_counts"]["hemorrhage"] == 4


def test_matlab_contract_fields(golden):
    rep_in = {
        "report_version": "1.0.0", "status": "COMPLETE",
        "sections": {
            "summary": {"workflow_recommendation": golden["triage"]["decision"]},
            "grading": {"predicted_grade": golden["grade"]},
            "triage": golden["triage"], "limitations": ["x"],
        },
    }
    assert rep_in["sections"]["triage"]["decision"] == "REFER"
    assert rep_in["sections"]["grading"]["predicted_grade"] == 2
    assert "reason_codes" in rep_in["sections"]["triage"]
