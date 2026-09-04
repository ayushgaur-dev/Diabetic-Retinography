"""Streamlit pipeline tests — Phase 10A. No Streamlit runtime, no model,
no datasets: orchestrator logic with stub graders. The .py app file itself
is verified by AST (imports streamlit, so it cannot be imported)."""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.screening_pipeline import (content_hash, gate_routing, run_screening,
                                    should_reuse)

def _fundus(size=256, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    r = size // 2 - 10
    img[(xx - c) ** 2 + (yy - c) ** 2 <= r ** 2] = [110, 70, 40]
    img[(xx - c - 40) ** 2 + (yy - c) ** 2 <= 18 ** 2] = [225, 195, 135]
    for _ in range(60):  # vessel-like texture (contrast for the quality gate)
        x0, y0 = rng.integers(c - r, c + r, 2)
        cv2.line(img, (int(x0), int(y0)),
                 (int(x0 + rng.integers(-30, 30)), int(y0 + rng.integers(-30, 30))),
                 (int(rng.integers(40, 80)), 25, 15), 1)
    return np.clip(img.astype(float) + rng.normal(0, 5, img.shape),
                   0, 255).astype(np.uint8)


def _grader(probs):
    calls = []

    def fn(_arr):
        calls.append(1)
        return list(probs)

    fn.calls = calls
    return fn


def test_content_hash_deterministic():
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")
    assert len(content_hash(b"x")) == 64


def test_should_reuse_rule():
    assert should_reuse("h", "h", True) is True
    assert should_reuse("h", "h", False) is False
    assert should_reuse("a", "b", True) is False
    assert should_reuse(None, "h", False) is False


def test_ungradable_blocks_inference():
    black = np.zeros((256, 256, 3), np.uint8)
    g = _grader([0.2] * 5)
    out = run_screening(black, grade_fn=g, stages={"full_evidence": False})
    assert out["blocked"] is True and out["blocked_reason"] == "UNGRADABLE"
    assert g.calls == []  # model NEVER called
    assert out["triage"]["decision"] == "UNGRADABLE"
    assert out["triage"]["method"] == "quality_gate_routing"
    assert "grade" not in out  # no fabricated grade


def test_graded_path_calls_model_once():
    g = _grader([0.7, 0.1, 0.1, 0.05, 0.05])
    out = run_screening(_fundus(), grade_fn=g, stages={"full_evidence": False})
    assert out["blocked"] is False
    assert len(g.calls) == 1
    assert out["grade"] == 0
    assert out["triage"]["predicted_grade"] == 0  # triage matches model
    assert abs(sum(out["calibrated_probabilities"]) - 1.0) < 1e-9


def test_displayed_triage_matches_phase8():
    from src.triage.config import load_triage_config
    from src.triage.decision import decide
    from src.triage.types import TriageInput

    g = _grader([0.1, 0.1, 0.7, 0.05, 0.05])
    out = run_screening(_fundus(), grade_fn=g, stages={"full_evidence": False})
    t = TriageInput(
        quality_status=out["quality"]["status"], quality_score=0.0,
        enhancement_status=out["enhancement_status"], predicted_grade=2,
        raw_probabilities=out["raw_probabilities"],
        calibrated_probabilities=out["calibrated_probabilities"],
        calibrated_confidence=out["calibrated_confidence"],
        referable_score=out["referable_score"], lesion_evidence={},
        vessel_available=False, optic_disc_status="UNKNOWN",
        fovea_status="UNKNOWN", consistency="UNKNOWN", processing_errors=[])
    assert decide(t, load_triage_config()).to_dict() == out["triage"]


def test_reports_originate_from_phase9():
    from src.reporting.config import load_reporting_config
    from src.reporting.deterministic_generator import generate

    g = _grader([0.7, 0.1, 0.1, 0.05, 0.05])
    out = run_screening(_fundus(), grade_fn=g, stages={"full_evidence": False})
    rep_in = {"image_id": "t", "quality": out["quality"], "enhancement": {},
              "grading": {"predicted_grade": out["grade"],
                          "raw_probabilities": out["raw_probabilities"],
                          "calibrated_probabilities": out["calibrated_probabilities"],
                          "calibrated_confidence": out["calibrated_confidence"]},
              "referable": {"score": out["referable_score"], "threshold": 0.7},
              "lesions": {}, "vessels": {}, "optic_disc": {}, "fovea": {},
              "explainability": {}, "triage": out["triage"], "warnings": []}
    rep = generate(rep_in, load_reporting_config()).to_dict()
    assert rep["sections"]["triage"]["decision"] == out["triage"]["decision"]
    assert rep["sections"]["grading"]["predicted_grade"] == out["grade"]


def test_incomplete_evidence_stays_incomplete():
    g = _grader([0.7, 0.1, 0.1, 0.05, 0.05])
    out = run_screening(_fundus(), grade_fn=g, stages={"full_evidence": False})
    assert out["lesions"] is None and out["vessel"] is None
    assert any("skipped" in w for w in out["warnings"])
    assert out["triage"]["safety_flags"]["missing_lesion_evidence"] is True


def test_no_phi_generation():
    g = _grader([0.7, 0.1, 0.1, 0.05, 0.05])
    out = run_screening(_fundus(), grade_fn=g, stages={"full_evidence": False})
    import json

    blob = json.dumps(out, default=str).lower()
    for field in ("patient_id", "patient_name", '"hba1c"',
                  '"visual_acuity"', '"diabetes_years"', '"age"'):
        assert field not in blob


def test_grading_failure_routed():
    def bad(_arr):
        raise RuntimeError("model down")

    out = run_screening(_fundus(), grade_fn=bad, stages={"full_evidence": False})
    assert out["blocked"] is True
    assert out["triage"]["decision"] == "TECHNICAL_REVIEW"
    assert "traceback" not in json_dumps(out).lower()


def json_dumps(out):
    import json

    return json.dumps(out, default=str)


def test_gate_routing_schema():
    r = gate_routing("UNGRADABLE", {"status": "UNGRADABLE"}, ["recapture"])
    assert r["decision"] == "UNGRADABLE" and r["predicted_grade"] is None
    assert r["method"] == "quality_gate_routing"
    r2 = gate_routing("ENHANCEMENT_FAILED", {"status": "BORDERLINE"}, [])
    assert r2["decision"] == "TECHNICAL_REVIEW"


def test_app_file_structure():
    """AST check: app + orchestrator wire Phases 2-9 APIs, recreate no ML logic."""
    repo = Path(__file__).resolve().parents[1]
    imported = set()
    for fname in ("app/streamlit_app.py", "app/screening_pipeline.py"):
        tree = ast.parse((repo / fname).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
                imported.update(a.name for a in node.names)
    for api in ("assess_image", "enhance_image", "triage_from_phases", "generate",
                "apply_temperature", "detect_lesions", "localize_optic_disc",
                "localize_fovea", "segment_vessels", "explain_class"):
        assert api in imported, f"pipeline must reuse {api}"
    text = (repo / "app" / "streamlit_app.py").read_text()
    assert "st.set_page_config" in text
    assert text.count("st.header") >= 8  # dashboard sections present
    assert "softmax" not in text  # no hand-rolled inference math in UI


def test_app_startup_compiles_and_imports_pipeline():
    import app.screening_pipeline as sp

    assert callable(sp.run_screening) and callable(sp.should_reuse)
    assert callable(sp.gate_routing) and callable(sp.content_hash)
