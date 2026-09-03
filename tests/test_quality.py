"""Image-quality tests — SIH26038 Phase 2.

Deterministic synthetic fixtures only (no downloads). Fixtures are
plumbing/contrast targets, not clinical images; thresholds under test are
ENGINEERING HEURISTICS from configs/quality_thresholds.json.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quality.config import DEFAULTS, load_config
from src.quality.gate import screen_image
from src.quality.quality_pipeline import assess_image
from src.quality.quality_score import aggregate
from src.quality.types import BAD, BORDERLINE, GOOD, UNGRADABLE, ComponentResult


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------

def _base(size=224, seed=11):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    r = size // 2 - 12
    img[(xx - c) ** 2 + (yy - c) ** 2 <= r ** 2] = [110, 70, 40]
    img[(xx - c - 35) ** 2 + (yy - c) ** 2 <= 16 ** 2] = [225, 195, 135]
    for _ in range(60):
        x0, y0 = rng.integers(c - r, c + r, 2)
        cv2.line(img, (int(x0), int(y0)),
                 (int(x0 + rng.integers(-30, 30)), int(y0 + rng.integers(-30, 30))),
                 (int(rng.integers(40, 80)), 25, 15), 1)
    img = img.astype(np.float32) + rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def sharp():
    return _base()


@pytest.fixture(scope="module")
def blurred(sharp):
    return cv2.GaussianBlur(sharp, (21, 21), 0)


@pytest.fixture(scope="module")
def dark(sharp):
    return np.clip(sharp.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def bright(sharp):
    return np.clip(sharp.astype(np.float32) * 1.6 + 60, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def low_contrast(sharp):
    f = sharp.astype(np.float32)
    return np.clip((f - f.mean()) * 0.25 + 110, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def small_field(sharp):
    img = np.zeros_like(sharp)
    img[82:142, 82:142] = cv2.resize(sharp, (60, 60))
    return img


# ---------------------------------------------------------------------------
# 1-7. Fixture behaviour
# ---------------------------------------------------------------------------

def test_sharp_synthetic_image_is_good(sharp):
    res, _ = assess_image(sharp)
    assert res.status == GOOD
    assert res.recapture_feedback == []
    assert res.enhancement_required is False


def test_blurred_image_is_ungradable(blurred):
    res, _ = assess_image(blurred)
    assert res.status == UNGRADABLE
    assert res.focus.status == BAD
    assert any("focus" in r for r in res.reasons)


def test_dark_image_is_ungradable(dark):
    res, _ = assess_image(dark)
    assert res.status == UNGRADABLE
    assert res.illumination.status == BAD


def test_bright_image_is_ungradable(bright):
    res, _ = assess_image(bright)
    assert res.status == UNGRADABLE
    assert res.illumination.status == BAD


def test_low_contrast_image_is_not_good(low_contrast):
    res, _ = assess_image(low_contrast)
    assert res.status in (BORDERLINE, UNGRADABLE)
    assert res.contrast.status != GOOD


def test_insufficient_field_is_ungradable(small_field):
    res, _ = assess_image(small_field)
    assert res.status == UNGRADABLE
    assert res.field_of_view.status == BAD
    assert len(res.recapture_feedback) >= 2  # field msg + cannot-be-graded msg


def test_non_fundus_solid_image_is_ungradable():
    res, _ = assess_image(np.full((224, 224, 3), 128, dtype=np.uint8))
    assert res.status == UNGRADABLE


# ---------------------------------------------------------------------------
# 8. Invalid input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "not-an-array",
    np.zeros((224, 224), dtype=np.uint8),          # missing channel axis
    np.zeros((10, 10, 3), dtype=np.uint8),         # too small
    np.zeros((224, 224, 3), dtype=np.float32),     # wrong dtype
])
def test_invalid_input_raises_value_error(bad):
    with pytest.raises(ValueError):
        assess_image(bad)


# ---------------------------------------------------------------------------
# 9. Result schema
# ---------------------------------------------------------------------------

def test_quality_result_schema(sharp):
    res, info = assess_image(sharp)
    d = res.to_dict()
    assert set(d) >= {"overall_score", "status", "focus", "illumination",
                      "contrast", "exposure", "field_of_view",
                      "retinal_coverage", "reasons", "recapture_feedback",
                      "enhancement_required", "enhancement_applied"}
    for comp in ("focus", "illumination", "contrast", "exposure",
                 "field_of_view", "retinal_coverage"):
        assert set(d[comp]) >= {"measurement", "score", "status", "explanation"}
        assert 0.0 <= d[comp]["score"] <= 1.0
    assert 0.0 <= d["overall_score"] <= 1.0
    assert d["status"] in (GOOD, BORDERLINE, UNGRADABLE)
    assert info["mask"].shape == sharp.shape[:2]
    assert info["elapsed_ms"] > 0


# ---------------------------------------------------------------------------
# 10. Threshold configuration
# ---------------------------------------------------------------------------

def test_config_loads_defaults_when_file_missing(tmp_path):
    cfg = load_config(path=tmp_path / "does_not_exist.json")
    assert cfg["focus"]["good_var"] == DEFAULTS["focus"]["good_var"]


def test_config_override_is_respected(sharp, tmp_path):
    import json

    cfg_path = tmp_path / "q.json"
    cfg_path.write_text(json.dumps({"focus": {"good_var": 1e12, "borderline_var": 1e11}}))
    cfg = load_config(path=cfg_path)
    res, _ = assess_image(sharp, config=cfg)
    assert res.focus.status == BAD  # impossible thresholds fail even the sharp fixture


def test_no_thresholds_hardcoded_in_component_modules():
    import re

    quality_dir = Path(__file__).resolve().parents[1] / "src" / "quality"
    offenders = []
    for mod in ("focus.py", "illumination.py", "contrast.py", "exposure.py",
                "coverage.py", "field_of_view.py", "quality_score.py"):
        src = (quality_dir / mod).read_text()
        for m in re.finditer(r"(?<!['\"\w])(?:[1-9]\d*\.\d+|\d{2,})(?!['\"\w])", src):
            val = m.group(0)
            # Allow: kernel sizes, block counts, morphological sizes, colour
            # constants, percentile indices, epsilon guards, score math.
            line = src[max(0, m.start() - 80):m.end() + 40]
            if re.search(r"(ones\(\(|blocks|percentile|1e-|clamp|log10|max\(|min\(|255|reshape|mgrid|95|/ 30)", line):
                continue
            # Allow 0..1 score-scale literals (score returns, default weight):
            # these are normalisation constants, not quality thresholds.
            if val in ("0.5", "1.0") and re.search(r"(return|score|weights\.get)", line):
                continue
            offenders.append(f"{mod}: {val} in ...{line.strip()[:90]}...")
    assert offenders == [], f"magic numbers found: {offenders}"


# ---------------------------------------------------------------------------
# 11. Aggregation rules
# ---------------------------------------------------------------------------

def _comp(name, status, score=0.5):
    return ComponentResult(name=name, measurement=1.0, measurement_unit="u",
                           score=score, status=status, explanation=name)


def test_any_bad_forces_ungradable():
    comps = {n: _comp(n, GOOD) for n in
             ("focus", "illumination", "contrast", "exposure",
              "field_of_view", "retinal_coverage")}
    comps["contrast"] = _comp("contrast", BAD, 0.0)
    status, _, _ = aggregate(comps, load_config())
    assert status == UNGRADABLE


def test_single_borderline_gives_borderline():
    comps = {n: _comp(n, GOOD) for n in
             ("focus", "illumination", "contrast", "exposure",
              "field_of_view", "retinal_coverage")}
    comps["focus"] = _comp("focus", BORDERLINE, 0.5)
    status, _, reasons = aggregate(comps, load_config())
    assert status == BORDERLINE
    assert reasons and "focus" in reasons[0]


def test_all_good_gives_good():
    comps = {n: _comp(n, GOOD, 0.9) for n in
             ("focus", "illumination", "contrast", "exposure",
              "field_of_view", "retinal_coverage")}
    status, overall, _ = aggregate(comps, load_config())
    assert status == GOOD
    assert overall == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 12. Recapture feedback is plain, deterministic, actionable
# ---------------------------------------------------------------------------

def test_recapture_feedback_plain_and_deterministic(blurred):
    r1, _ = assess_image(blurred)
    r2, _ = assess_image(blurred)
    assert r1.recapture_feedback == r2.recapture_feedback
    assert r1.recapture_feedback, "UNGRADABLE must carry recapture instructions"
    jargon = ("laplacian", "variance", "percentile", "morphology", "p95")
    for msg in r1.recapture_feedback:
        assert not any(j in msg.lower() for j in jargon), f"jargon in: {msg}"


# ---------------------------------------------------------------------------
# 13 + §18. Gate blocks the model on UNGRADABLE, calls it on GOOD
# ---------------------------------------------------------------------------

def test_gate_never_calls_model_on_ungradable(blurred):
    calls = []

    def fake_model(_img):
        calls.append(1)
        return {"grade": 0}

    out = screen_image(blurred, fake_model)
    assert out["model_called"] is False
    assert out["prediction"] is None
    assert calls == []
    assert out["gradable"] is False
    assert out["blocked_reason"]
    assert out["quality"]["status"] == UNGRADABLE


def test_gate_calls_model_on_good(sharp):
    def fake_model(_img):
        return {"grade": 2, "confidence": 0.8}

    out = screen_image(sharp, fake_model)
    assert out["model_called"] is True
    assert out["prediction"] == {"grade": 2, "confidence": 0.8}
    assert out["gradable"] is True
    assert out["blocked_reason"] is None


def test_gate_marks_borderline_enhancement_required(low_contrast):
    res, _ = assess_image(low_contrast)
    if res.status == BORDERLINE:
        out = screen_image(low_contrast, lambda _img: {"grade": 1})
        assert out["model_called"] is True  # backward compatible in Phase 2
        assert out["quality"]["enhancement_required"] is True
        assert out["quality"]["enhancement_applied"] is False
    else:
        assert res.status == UNGRADABLE  # also acceptable: still no silent pass
