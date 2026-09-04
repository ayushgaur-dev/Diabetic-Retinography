"""Safety-gate matrix (Phase 13, cases A-H). Stub graders; no datasets.
Every layer must preserve the same blocking/routing state."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.screening_pipeline import run_screening
from src.triage.config import load_triage_config
from src.triage.decision import decide
from src.triage.types import TriageInput


def _textured(size=256, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    r = size // 2 - 10
    img[(xx - c) ** 2 + (yy - c) ** 2 <= r ** 2] = [110, 70, 40]
    img[(xx - c - 40) ** 2 + (yy - c) ** 2 <= 18 ** 2] = [225, 195, 135]
    for _ in range(60):
        x0, y0 = rng.integers(c - r, c + r, 2)
        cv2.line(img, (int(x0), int(y0)),
                 (int(x0 + rng.integers(-30, 30)), int(y0 + rng.integers(-30, 30))),
                 (int(rng.integers(40, 80)), 25, 15), 1)
    return np.clip(img.astype(float) + rng.normal(0, 5, img.shape),
                   0, 255).astype(np.uint8)


def _grader(probs, calls=None):
    def fn(_arr):
        if calls is not None:
            calls.append(1)
        return list(probs)

    return fn


def _triage_input(grade, conf, score):
    p = [(1 - conf) / 4] * 5
    p[grade] = conf
    return TriageInput(
        quality_status="GOOD", quality_score=0.9, enhancement_status="none",
        predicted_grade=grade, raw_probabilities=list(p),
        calibrated_probabilities=list(p), calibrated_confidence=conf,
        referable_score=score, lesion_evidence={}, vessel_available=True,
        optic_disc_status="DETECTED", fovea_status="DETECTED",
        consistency="INCONCLUSIVE", processing_errors=[])


def test_case_a_good_allows_grading():
    calls = []
    out = run_screening(_textured(), grade_fn=_grader([0.8, 0.05, 0.05, 0.05, 0.05], calls),
                        stages={"full_evidence": False})
    assert out["blocked"] is False and len(calls) == 1
    assert out["grade"] == 0


def test_case_b_borderline_enhancement_path():
    f = _textured().astype(np.float32)
    low = np.clip((f - f.mean()) * 0.25 + 110, 0, 255).astype(np.uint8)
    calls = []
    out = run_screening(low, grade_fn=_grader([0.1, 0.1, 0.6, 0.1, 0.1], calls),
                        stages={"full_evidence": False})
    assert out["quality"]["status"] in ("BORDERLINE", "GOOD", "UNGRADABLE")
    if out["quality"]["status"] == "BORDERLINE":
        assert out["enhancement_status"] in ("success", "failed")
        if out["enhancement_status"] == "success":
            assert len(calls) == 1  # reassessed-accepted image graded once
        else:
            assert out["blocked"] is True and "grade" not in out
    elif out["quality"]["status"] == "UNGRADABLE":
        assert calls == []


def test_case_c_ungradable_blocks():
    calls = []
    out = run_screening(np.zeros((256, 256, 3), np.uint8),
                        grade_fn=_grader([0.2] * 5, calls),
                        stages={"full_evidence": False})
    assert out["blocked"] is True and calls == []
    assert "grade" not in out
    assert out["triage"]["decision"] == "UNGRADABLE"


def test_case_d_enhancement_failure_no_grade(monkeypatch):
    import src.preprocessing.enhancement_pipeline as EP
    import src.quality.quality_pipeline as QP

    calls = []

    class FakeQ:
        def to_dict(self):
            return {"status": "BORDERLINE", "overall_score": 0.5,
                    "recapture_feedback": []}

    class Dead:
        bypassed = False
        rejected = False
        enhancement_successful = False
        before_quality = {"status": "BORDERLINE"}
        after_quality = {"status": "BORDERLINE"}
        enhanced_image = None

        def to_dict(self):
            return {"before_quality": self.before_quality,
                    "after_quality": self.after_quality,
                    "enhancement_successful": False,
                    "operations_applied": ["clahe"], "warnings": []}

    monkeypatch.setattr(QP, "assess_image", lambda *a, **k: (FakeQ(), {}))
    monkeypatch.setattr(EP, "enhance_image", lambda *a, **k: (Dead(), {}))
    out = run_screening(_textured(),
                        grade_fn=_grader([0.2] * 5, calls),
                        stages={"full_evidence": False})
    assert out["quality"]["status"] == "BORDERLINE"
    assert out["blocked"] is True and calls == []
    assert "grade" not in out
    assert out["triage"]["decision"] == "TECHNICAL_REVIEW"


def test_case_e_processing_failure_propagates():
    def bad(_arr):
        raise RuntimeError("boom")

    out = run_screening(_textured(), grade_fn=bad, stages={"full_evidence": False})
    assert out["blocked"] is True
    assert out["errors"].get("grading")
    assert out["triage"]["decision"] == "TECHNICAL_REVIEW"
    assert "traceback" not in str(out).lower()


def test_case_f_missing_lesions_incomplete():
    out = run_screening(_textured(),
                        grade_fn=_grader([0.8, 0.05, 0.05, 0.05, 0.05]),
                        stages={"full_evidence": False})
    assert out["lesions"] is None
    assert out["triage"]["safety_flags"]["missing_lesion_evidence"] is True
    assert out["triage"]["decision"] in ("ROUTINE", "TECHNICAL_REVIEW")


def test_case_g_low_confidence_routing():
    cfg = load_triage_config()
    assert decide(_triage_input(0, 0.3, 0.1), cfg).decision == "TECHNICAL_REVIEW"
    assert decide(_triage_input(0, 0.9, 0.1), cfg).decision == "ROUTINE"


def test_case_h_high_severity_urgent():
    cfg = load_triage_config()
    for g in (3, 4):
        d = decide(_triage_input(g, 0.8, 0.9), cfg)
        assert d.decision == "URGENT_REVIEW" and d.priority == "CRITICAL"
        assert d.predicted_grade == g  # immutable
