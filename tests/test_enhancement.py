"""Adaptive-enhancement tests — SIH26038 Phase 3.

Deterministic synthetic fixtures only. Thresholds/parameters under test are
ENGINEERING HEURISTICS from configs/enhancement_config.json.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import operations as ops
from src.preprocessing.config import DEFAULTS, load_enhancement_config
from src.preprocessing.enhancement import apply_operations, select_operations
from src.preprocessing.enhancement_pipeline import (enhance_image,
                                                    screen_enhance_infer)
from src.preprocessing.types import ENHANCEMENT_VERSION
from src.quality.field_of_view import detect_retinal_field
from src.quality.quality_pipeline import assess_image


# ---------------------------------------------------------------------------
# Fixtures
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
def low_contrast(sharp):
    f = sharp.astype(np.float32)
    return np.clip((f - f.mean()) * 0.25 + 110, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def uneven(sharp):
    yy, _ = np.mgrid[0:224, 0:224]
    gain = (0.5 + yy / 224.0).astype(np.float32)  # dark top, bright bottom
    return np.clip(sharp.astype(np.float32) * gain[:, :, None], 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def cast(sharp):
    f = sharp.astype(np.float32)
    f[:, :, 0] *= 1.35
    f[:, :, 2] *= 0.70
    return np.clip(f, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def noisy(sharp):
    rng = np.random.default_rng(3)
    return np.clip(sharp.astype(np.float32) + rng.normal(0, 12, sharp.shape),
                   0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def mask(sharp):
    return detect_retinal_field(sharp, {"field_of_view": {"red_factor": 0.35,
                                                          "red_floor": 10.0,
                                                          "min_plausible_fraction": 0.05}})["mask"]


# ---------------------------------------------------------------------------
# 1-4. Individual operations
# ---------------------------------------------------------------------------

def test_clahe_raises_local_contrast(low_contrast, mask):
    from src.quality.contrast import assess_contrast
    from src.quality.config import load_config

    cfg = load_config()
    gray = cv2.cvtColor(low_contrast, cv2.COLOR_RGB2GRAY).astype(np.float32)
    before = assess_contrast(gray, mask, cfg).measurement
    out = ops.apply_clahe(low_contrast, mask, 2.0, (8, 8))
    gray2 = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
    after = assess_contrast(gray2, mask, cfg).measurement
    assert after > before
    assert out.shape == low_contrast.shape and out.dtype == np.uint8


def test_clahe_preserves_hue_direction(sharp, mask):
    """LAB-L-only CLAHE must not flip dominant channel ordering."""
    out = ops.apply_clahe(sharp, mask, 2.0, (8, 8))
    m = mask > 0
    for c in range(3):
        assert out[:, :, c][m].mean() > 0
    # red channel stays dominant inside the mask (fundus is red/orange)
    means = [out[:, :, c][m].mean() for c in range(3)]
    assert means[0] >= means[2]


def test_illumination_normalization_evens_field(uneven, mask):
    from src.quality.illumination import _block_nonuniformity

    gray = cv2.cvtColor(uneven, cv2.COLOR_RGB2GRAY).astype(np.float32)
    before = _block_nonuniformity(gray, mask)
    out = ops.normalize_illumination(uneven, mask)
    gray2 = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
    after = _block_nonuniformity(gray2, mask)
    assert after < before
    assert out.dtype == np.uint8


def test_denoise_reduces_noise_without_reshape(noisy, mask):
    flat = noisy[100:130, 100:130].astype(np.float32)
    out = ops.denoise(noisy, mask)
    flat_out = out[100:130, 100:130].astype(np.float32)
    assert flat_out.std() < flat.std()
    assert out.shape == noisy.shape and out.dtype == np.uint8


def test_color_normalization_reduces_cast(cast, mask):
    before = ops.channel_spread(cast, mask)
    out = ops.normalize_color(cast, mask, 1.25)
    after = ops.channel_spread(out, mask)
    assert after < before
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# 5. Masking: background outside the retinal field is preserved
# ---------------------------------------------------------------------------

def test_operations_preserve_background(low_contrast, mask):
    for fn in (lambda i: ops.apply_clahe(i, mask, 2.0, (8, 8)),
               lambda i: ops.normalize_illumination(i, mask),
               lambda i: ops.denoise(i, mask),
               lambda i: ops.normalize_color(i, mask, 1.25)):
        out = fn(low_contrast)
        assert (out[mask == 0] == low_contrast[mask == 0]).all()


def test_mask_reuses_phase2_detector(sharp):
    import json

    cfg = json.load(open(Path(__file__).resolve().parents[1]
                         / "configs" / "quality_thresholds.json"))
    info = detect_retinal_field(sharp, cfg)
    assert info["plausible"] and info["mask_fraction"] > 0.4


# ---------------------------------------------------------------------------
# 6. Result schema
# ---------------------------------------------------------------------------

def test_enhancement_result_schema(low_contrast):
    res, _ = enhance_image(low_contrast)
    d = res.to_dict()
    assert set(d) >= {"operations_applied", "before_quality", "after_quality",
                      "enhancement_successful", "comparison",
                      "changed_quality_dimensions", "warnings", "bypassed",
                      "rejected", "enhancement_version", "enhanced_shape"}
    assert d["enhancement_version"] == ENHANCEMENT_VERSION
    assert d["comparison"] in ("IMPROVED", "UNCHANGED", "WORSE")


# ---------------------------------------------------------------------------
# 7-9. State transitions
# ---------------------------------------------------------------------------

def test_good_image_bypasses_enhancement(sharp):
    res, _ = enhance_image(sharp)
    assert res.bypassed is True
    assert res.operations_applied == []
    assert res.enhanced_image is None


def test_borderline_image_triggers_enhancement(low_contrast):
    res, _ = enhance_image(low_contrast)
    assert res.bypassed is False and res.rejected is False
    assert res.operations_applied, "BORDERLINE must select at least one op"
    assert res.after_quality, "quality must be reassessed after enhancement"


def test_ungradable_image_is_rejected():
    res, _ = enhance_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert res.rejected is True
    assert res.operations_applied == []
    assert res.enhanced_image is None


# ---------------------------------------------------------------------------
# 10-11. Reassessment + discard-on-worse
# ---------------------------------------------------------------------------

def test_safety_discard_path(low_contrast):
    """An impossible mask-preservation demand must discard the enhanced
    image — exercises the discard machinery through config alone."""
    cfg = load_enhancement_config()
    cfg["sanity_checks"]["min_mask_fraction_ratio"] = 2.0  # unachievable
    res, _ = enhance_image(low_contrast, enh_config=cfg)
    assert res.bypassed is False and res.rejected is False
    assert res.operations_applied, "ops ran before the safety veto"
    assert res.enhancement_successful is False
    assert res.enhanced_image is None  # discarded, original retained
    assert res.comparison == "WORSE"
    assert any("mask" in w for w in res.warnings)


def _quality_dict_with(status_map):
    q = {}
    for name in ("focus", "illumination", "contrast", "exposure",
                 "field_of_view", "retinal_coverage"):
        q[name] = {"status": status_map.get(name, "GOOD"),
                   "details": {"black_fraction": 0.05, "white_fraction": 0.0}}
    return q


def test_selection_is_quality_driven_not_fixed(sharp, mask):
    """Same image, different quality verdicts -> different operations."""
    from src.preprocessing.config import load_enhancement_config as lec

    cfg = lec()
    contrast_q = _quality_dict_with({"contrast": "BORDERLINE"})
    illum_q = _quality_dict_with({"illumination": "BORDERLINE"})
    ops_c, _ = select_operations(contrast_q, sharp, mask, cfg)
    ops_i, _ = select_operations(illum_q, sharp, mask, cfg)
    assert "clahe" in ops_c and "illumination_normalization" not in ops_c
    assert "illumination_normalization" in ops_i and "clahe" not in ops_i
    assert ops_c != ops_i


def test_focus_borderline_selects_no_sharpening(sharp, mask):
    from src.preprocessing.config import load_enhancement_config as lec

    cfg = lec()
    ops_f, warnings = select_operations(_quality_dict_with({"focus": "BORDERLINE"}),
                                        sharp, mask, cfg)
    # focus alone selects no corrective op (blur is unrecoverable); only the
    # independent image-based color-cast trigger may fire.
    assert set(ops_f) <= {"color_normalization"}
    assert any("blur" in w for w in warnings)


# ---------------------------------------------------------------------------
# 12-16. Output integrity
# ---------------------------------------------------------------------------

def test_original_never_mutated(low_contrast):
    before = low_contrast.copy()
    enhance_image(low_contrast)
    assert (low_contrast == before).all()


def test_output_validity(low_contrast):
    res, _ = enhance_image(low_contrast)
    if res.enhanced_image is not None:
        e = res.enhanced_image
        assert e.shape == low_contrast.shape
        assert e.dtype == np.uint8
        assert np.isfinite(e.astype(np.float32)).all()
        assert e.min() >= 0 and e.max() <= 255


def test_no_nan_inf_on_degenerate_inputs():
    for img in (np.zeros((224, 224, 3), dtype=np.uint8),
                np.full((224, 224, 3), 255, dtype=np.uint8),
                np.full((224, 224, 3), 128, dtype=np.uint8)):
        res, _ = enhance_image(img)  # must not raise
        if res.enhanced_image is not None:
            assert np.isfinite(res.enhanced_image.astype(np.float32)).all()


# ---------------------------------------------------------------------------
# 17 + §19. Model integration
# ---------------------------------------------------------------------------

def test_model_receives_enhanced_only_on_success(low_contrast):
    seen = []

    def fake_model(img):
        seen.append(img.copy())
        return {"grade": 1}

    out = screen_enhance_infer(low_contrast, fake_model)
    res = out["enhancement"]
    if res["enhancement_successful"]:
        assert out["model_called"] is True and out["used_image"] == "enhanced"
        assert len(seen) == 1 and (seen[0] != low_contrast).any()
    else:
        assert out["model_called"] is False and seen == []


def test_borderline_fail_blocks_model():
    cfg = load_enhancement_config()
    cfg["sanity_checks"]["min_mask_fraction_ratio"] = 2.0  # force discard
    calls = []

    def fake_model(_img):
        calls.append(1)
        return {"grade": 1}

    f = _base().astype(np.float32)
    lowc = np.clip((f - f.mean()) * 0.25 + 110, 0, 255).astype(np.uint8)
    out = screen_enhance_infer(lowc, fake_model, enh_config=cfg)
    assert out["enhancement"]["enhancement_successful"] is False
    assert out["model_called"] is False and calls == []
    assert out["prediction"] is None
    assert out["recapture_feedback"], "failed rescue must explain next steps"


def test_good_uses_original_ungradable_blocks(sharp):
    got = []

    def fake_model(img):
        got.append(img)
        return {"grade": 0}

    out = screen_enhance_infer(sharp, fake_model)
    assert out["used_image"] == "original" and (got[0] == sharp).all()

    out2 = screen_enhance_infer(np.zeros((224, 224, 3), dtype=np.uint8), fake_model)
    assert out2["model_called"] is False and len(got) == 1


def test_no_labels_used_in_enhancement(low_contrast):
    """Enhancement path accepts no grade/label argument by construction."""
    import inspect

    from src.preprocessing import enhancement_pipeline as ep

    for fn in (ep.enhance_image, ep.screen_enhance_infer):
        params = inspect.signature(fn).parameters
        assert not any(k in params for k in ("grade", "label", "diagnosis")), fn


def test_parameters_configurable(low_contrast):
    c1 = load_enhancement_config()
    c2 = load_enhancement_config()
    c2["clahe"]["clip_limit"] = 8.0
    _, info = enhance_image(low_contrast)
    e1, _ = apply_operations(low_contrast, info["mask"], ["clahe"], c1)
    e2, _ = apply_operations(low_contrast, info["mask"], ["clahe"], c2)
    assert not (e1 == e2).all(), "clip_limit override must change output"
    assert c1["denoise"]["diameter"] == DEFAULTS["denoise"]["diameter"]
