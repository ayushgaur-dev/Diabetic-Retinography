"""Optic-disc tests — SIH26038 Phase 4B. Synthetic fixtures only (no dataset
dependency); Drishti-GS evaluation is covered by the evaluate CLI."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retina.optic_disc.candidates import generate_candidates
from src.retina.optic_disc.config import DEFAULTS, load_disc_config
from src.retina.optic_disc.metrics import (bbox_iou, center_error,
                                           normalized_error, summarize)
from src.retina.optic_disc.pipeline import localize_optic_disc
from src.retina.optic_disc.preprocessing import to_working_resolution
from src.retina.optic_disc.scoring import score_candidates
from src.retina.optic_disc.types import (DETECTED, LOW_CONFIDENCE,
                                         NOT_DETECTED)
from src.retina.optic_disc.vessel_convergence import compute_convergence


# ---------------------------------------------------------------------------
# Synthetic fixtures (seeded, deterministic)
# ---------------------------------------------------------------------------

def _fundus_with_disc(size=400, seed=9, disc_xy=None, disc_r=38,
                      distractors=(), vessels=True):
    """Fundus-like RGB + FOV mask. Disc: bright region with converging lines."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    fov = ((xx - c) ** 2 + (yy - c) ** 2 <= (size // 2 - 10) ** 2)
    img[fov] = [110, 70, 40]
    dx, dy = disc_xy or (c + 70, c - 10)
    disc = (xx - dx) ** 2 + (yy - dy) ** 2 <= disc_r ** 2
    img[disc] = [235, 205, 150]
    if vessels:
        for _ in range(10):  # lines converging toward the disc
            ang = rng.uniform(0, 2 * np.pi)
            r0 = disc_r + 5
            x0, y0 = dx + r0 * np.cos(ang), dy + r0 * np.sin(ang)
            x1, y1 = dx + (r0 + 60) * np.cos(ang), dy + (r0 + 60) * np.sin(ang)
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (60, 25, 15), 2)
    for (bx, by, br) in distractors:  # bright non-disc blobs, no vessels
        img[(xx - bx) ** 2 + (yy - by) ** 2 <= br ** 2] = [240, 220, 180]
    img = img.astype(np.float32) + rng.normal(0, 4, img.shape)
    return (np.clip(img, 0, 255).astype(np.uint8),
            (fov.astype(np.uint8)) * 255, (float(dx), float(dy), float(disc_r)))


@pytest.fixture(scope="module")
def cfg():
    return load_disc_config()


def test_fixture_a_detects_disc(cfg):
    """A. Bright disc with converging vessels -> DETECTED near GT."""
    img, fov, (dx, dy, dr) = _fundus_with_disc()
    res = localize_optic_disc(img, fov_mask=fov, config=cfg).to_dict()
    assert res["status"] == DETECTED and res["detected"] is True
    err = center_error(res["center"], (dx, dy)) / (2 * dr)
    assert err < 0.5, f"normalized error {err}"
    assert res["confidence"] > 0
    assert res["radius"] > 0 and res["bounding_box"][2] > res["bounding_box"][0]


def test_fixture_b_distractors_do_not_win(cfg):
    """B. Disc + two brighter-but-vesselless blobs -> still finds the disc."""
    img, fov, (dx, dy, dr) = _fundus_with_disc(
        distractors=[(90, 90, 26), (300, 320, 22)])
    res = localize_optic_disc(img, fov_mask=fov, config=cfg).to_dict()
    assert res["detected"] is True
    err = center_error(res["center"], (dx, dy)) / (2 * dr)
    assert err < 0.5, f"distractor won: {res['center']} vs GT {(dx, dy)}"


def test_fixture_c_no_disc_fails_explicitly(cfg):
    """C. Uniform field, no disc -> NOT_DETECTED, no invented coordinates."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = [110, 70, 40]
    fov = np.full((300, 300), 255, dtype=np.uint8)
    res = localize_optic_disc(img, fov_mask=fov, config=cfg).to_dict()
    assert res["status"] == NOT_DETECTED and res["detected"] is False
    assert res["center"] is None and res["radius"] is None
    assert res["bounding_box"] is None
    assert res["warnings"], "failure must carry warnings"


def test_fixture_d_resolution_independence(cfg):
    """D. Same scene at 300px and 600px -> consistent normalized centers."""
    small, fov_s, _ = _fundus_with_disc(size=300, disc_xy=(220, 140), disc_r=28)
    big, fov_b, _ = _fundus_with_disc(size=600, disc_xy=(440, 280), disc_r=56)
    rs = localize_optic_disc(small, fov_mask=fov_s, config=cfg).to_dict()
    rb = localize_optic_disc(big, fov_mask=fov_b, config=cfg).to_dict()
    assert rs["detected"] and rb["detected"]
    ns = (rs["center"][0] / 300, rs["center"][1] / 300)
    nb = (rb["center"][0] / 600, rb["center"][1] / 600)
    assert abs(ns[0] - nb[0]) < 0.05 and abs(ns[1] - nb[1]) < 0.05
    assert 0 < rs["radius_normalized"] < 0.25
    assert 0 < rb["radius_normalized"] < 0.25


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def test_candidate_generation_and_scoring(cfg):
    img, fov, _ = _fundus_with_disc()
    from src.retina.optic_disc.preprocessing import to_working_resolution as twr

    work, _ = twr(img, 800)
    cands = generate_candidates(work, fov, cfg)
    assert len(cands) >= 1
    ranked = score_candidates(cands, cfg, 70.0, 240.0)
    assert ranked[0].score >= ranked[-1].score
    assert set(ranked[0].contributions) == {"brightness", "contrast",
                                           "geometry", "convergence"}


def test_vessel_convergence_prefers_disc(cfg):
    img, fov, (dx, dy, dr) = _fundus_with_disc()
    from src.retina.vessels.pipeline import segment_vessels

    vessel = segment_vessels(img, fov).vessel_mask

    class C:
        pass

    at = C()
    at.area, at.centroid_x, at.centroid_y = np.pi * dr ** 2, dx, dy
    away = C()
    away.area, away.centroid_x, away.centroid_y = np.pi * dr ** 2, 90.0, 90.0
    assert (compute_convergence(at, vessel, fov, cfg)
            > compute_convergence(away, vessel, fov, cfg))


def test_convergence_fallback_without_vessels(cfg):
    img, fov, _ = _fundus_with_disc(vessels=False)
    res = localize_optic_disc(img, fov_mask=fov,
                              vessel_mask=np.zeros((400, 400), np.uint8),
                              config=cfg).to_dict()
    assert res["detected"] in (True, False)  # must not crash either way
    assert isinstance(res["confidence"], float)


# ---------------------------------------------------------------------------
# Schema / bounds / validity
# ---------------------------------------------------------------------------

def test_output_schema_and_bounds(cfg):
    img, fov, _ = _fundus_with_disc()
    d = localize_optic_disc(img, fov_mask=fov, config=cfg).to_dict()
    assert set(d) >= {"detected", "status", "center", "radius",
                      "radius_normalized", "bounding_box", "confidence",
                      "candidate_count", "selected_candidate_score",
                      "candidates", "method", "warnings"}
    h, w = img.shape[:2]
    x, y = d["center"]
    assert 0 <= x < w and 0 <= y < h
    assert d["radius"] > 0 and d["radius_normalized"] == pytest.approx(
        d["radius"] / w)
    x0, y0, x1, y1 = d["bounding_box"]
    assert x1 > x0 and y1 > y0
    assert 0.0 <= d["confidence"] <= 1.0
    assert np.isfinite(d["confidence"]) and np.isfinite(d["radius"])


def test_confidence_is_algorithmic_not_probability(cfg):
    img, fov, _ = _fundus_with_disc()
    d = localize_optic_disc(img, fov_mask=fov, config=cfg).to_dict()
    assert d["method"] == "bright_candidate_plus_vessel_convergence"
    # confidence must equal the winning candidate's exposed score
    assert d["confidence"] == pytest.approx(d["selected_candidate_score"])
    assert d["confidence"] == pytest.approx(max(c["score"] for c in d["candidates"]))


def test_invalid_input_rejected(cfg):
    with pytest.raises(ValueError):
        localize_optic_disc(np.zeros((100, 100), dtype=np.uint8), config=cfg)


def test_working_resize_documented(cfg):
    img, _, _ = _fundus_with_disc(size=1200)
    small, scale = to_working_resolution(img, 800)
    assert small.shape[1] == 800 and scale < 1.0
    res = localize_optic_disc(img, config=cfg).to_dict()
    assert any("working width" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_definitions():
    assert center_error((0, 0), (3, 4)) == 5.0
    assert normalized_error((0, 0), (3, 4), 10.0) == 0.5
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    recs = [
        {"detected": True, "center_error_px": 5.0, "normalized_error": 0.1,
         "bbox_iou": 0.8},
        {"detected": True, "center_error_px": 60.0, "normalized_error": 0.6,
         "bbox_iou": 0.3},
        {"detected": False, "center_error_px": None, "normalized_error": None,
         "bbox_iou": None},
    ]
    s = summarize(recs)
    assert s["n_images"] == 3 and s["n_detected"] == 2
    assert s["detection_at_0.25"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["detection_at_0.5"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["detection_at_1.0"] == pytest.approx(2 / 3, abs=1e-4)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_defaults_and_weights(cfg):
    assert cfg["decision"]["detect_threshold"] == \
        DEFAULTS["decision"]["detect_threshold"]
    assert sum(cfg["scoring"]["weights"].values()) == pytest.approx(1.0)
    import json

    p = Path(__file__).resolve().parents[1] / "configs" / "optic_disc_config.json"
    file_cfg = json.load(open(p))
    assert file_cfg["candidates"]["red_percentile"] == cfg["candidates"]["red_percentile"]


def test_parameters_configurable(cfg):
    import copy

    img, fov, _ = _fundus_with_disc()
    c2 = copy.deepcopy(cfg)
    c2["decision"]["detect_threshold"] = 0.999
    c2["decision"]["low_confidence_threshold"] = 0.999
    d = localize_optic_disc(img, fov_mask=fov, config=c2).to_dict()
    assert d["status"] == NOT_DETECTED and d["detected"] is False


# ---------------------------------------------------------------------------
# Integration: Phase 2 FOV -> localization; failure is downstream-safe
# ---------------------------------------------------------------------------

def test_integration_phase2_fov_to_disc(cfg):
    """Fundus -> Phase 2 FOV -> localization -> structured result."""
    from src.quality.field_of_view import detect_retinal_field
    from src.quality.config import load_config as load_qconfig

    img, _, (dx, dy, dr) = _fundus_with_disc()
    fov = detect_retinal_field(img, load_qconfig())["mask"]
    assert (fov > 0).any()
    d = localize_optic_disc(img, fov_mask=fov, config=cfg).to_dict()
    assert d["detected"] is True
    assert center_error(d["center"], (dx, dy)) / (2 * dr) < 0.5


def test_downstream_handles_missing_disc(cfg):
    """NOT_DETECTED result must be consumable without crashing."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = [110, 70, 40]
    d = localize_optic_disc(img, config=cfg).to_dict()

    def downstream_consumer(disc):
        if not disc["detected"]:
            return {"anatomy_available": False, "reason": disc["warnings"][0]}
        return {"anatomy_available": True, "center": disc["center"]}

    out = downstream_consumer(d)
    assert out == {"anatomy_available": False, "reason": d["warnings"][0]}
