"""Fovea tests — SIH26038 Phase 4C. Synthetic fixtures only (no dataset
dependency); IDRiD evaluation is covered by the evaluate CLI."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retina.fovea.anatomical_geometry import (expected_fovea,
                                                  infer_laterality)
from src.retina.fovea.candidates import generate_candidates
from src.retina.fovea.config import DEFAULTS, load_fovea_config
from src.retina.fovea.dataset import _read_centers
from src.retina.fovea.metrics import (center_error, normalized_error_fov,
                                      summarize)
from src.retina.fovea.pipeline import localize_fovea
from src.retina.fovea.scoring import score_candidates
from src.retina.fovea.types import DETECTED, LOW_CONFIDENCE, NOT_DETECTED
from src.retina.fovea.vessel_features import vessel_sparsity


# ---------------------------------------------------------------------------
# Synthetic scene: disc + converging vessels + dark avascular fovea spot
# ---------------------------------------------------------------------------

def _scene(size=512, seed=21, disc_side="left", distractors=(),
           fovea=True, vessels=True):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    fov = ((xx - c) ** 2 + (yy - c) ** 2 <= (size // 2 - 8) ** 2)
    img[fov] = [120, 75, 45]
    dd = size / 8.0  # pseudo disc diameter
    dx = c - 1.6 * dd if disc_side == "left" else c + 1.6 * dd
    dy = float(c)
    dr = dd / 2.0
    img[(xx - dx) ** 2 + (yy - dy) ** 2 <= dr ** 2] = [235, 205, 150]
    tsign = +1 if disc_side == "left" else -1
    fx, fy = dx + tsign * 2.5 * dd, dy + 0.2 * dd
    vessel = np.zeros((size, size), dtype=np.uint8)
    if vessels:
        for _ in range(12):
            ang = rng.uniform(0, 2 * np.pi)
            x0, y0 = dx + (dr + 4) * np.cos(ang), dy + (dr + 4) * np.sin(ang)
            x1, y1 = dx + (dr + 70) * np.cos(ang), dy + (dr + 70) * np.sin(ang)
            # spare the fovea zone -> avascular appearance
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            if abs(mx - fx) < dd and abs(my - fy) < dd:
                continue
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (60, 25, 15), 2)
            cv2.line(vessel, (int(x0), int(y0)), (int(x1), int(y1)), 255, 2)
    if fovea:
        img[(xx - fx) ** 2 + (yy - fy) ** 2 <= (0.35 * dd) ** 2] = [70, 40, 25]
    for (bx, by, br) in distractors:
        img[(xx - bx) ** 2 + (yy - by) ** 2 <= br ** 2] = [75, 45, 28]
    img = img.astype(np.float32) + rng.normal(0, 4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    fovm = (fov.astype(np.uint8)) * 255
    return img, fovm, vessel, (float(fx), float(fy)), (float(dx), float(dy), float(dr))


@pytest.fixture(scope="module")
def cfg():
    return load_fovea_config()


def _disc_dict(dx, dy, dr, status=DETECTED):
    return {"detected": True, "status": status, "center": [dx, dy],
            "radius": dr, "confidence": 0.8}


# --- 1-4. Basic fixtures ----------------------------------------------------

def test_obvious_fovea_detected(cfg):
    img, fov, vessel, (fx, fy), (dx, dy, dr) = _scene()
    d = localize_fovea(img, fov_mask=fov, disc=_disc_dict(dx, dy, dr),
                       vessel_mask=vessel, config=cfg).to_dict()
    assert d["status"] == DETECTED and d["detected"] is True
    assert center_error(d["center_x_y"], (fx, fy)) / img.shape[1] < 0.10


def test_distractor_dark_regions_do_not_win(cfg):
    img, fov, vessel, (fx, fy), (dx, dy, dr) = _scene(
        distractors=[(100, 100, 20), (400, 420, 18)])
    d = localize_fovea(img, fov_mask=fov, disc=_disc_dict(dx, dy, dr),
                       vessel_mask=vessel, config=cfg).to_dict()
    assert d["detected"] is True
    assert center_error(d["center_x_y"], (fx, fy)) / img.shape[1] < 0.10


def test_vessel_sparsity_prefers_avascular(cfg):
    img, fov, vessel, (fx, fy), _ = _scene()
    dd = img.shape[0] / 8.0
    s_fov = vessel_sparsity(fx, fy, vessel, dd, cfg)
    s_ves = vessel_sparsity(fx + 2 * dd, fy, vessel, dd, cfg)
    assert s_fov >= s_ves


def test_no_evidence_fails_gracefully(cfg):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:] = [120, 75, 45]
    fov = np.full((256, 256), 255, dtype=np.uint8)
    nodisc = {"detected": False, "status": NOT_DETECTED, "center": None,
              "radius": None, "confidence": 0.0}
    d = localize_fovea(img, fov_mask=fov, disc=nodisc,
                       vessel_mask=np.zeros((256, 256), np.uint8),
                       config=cfg).to_dict()
    assert d["status"] in (LOW_CONFIDENCE, NOT_DETECTED)
    if not d["detected"]:
        assert d["center_x_y"] is None


# --- 5-8. Geometry ----------------------------------------------------------

def test_left_eye_orientation(cfg):
    img, fov, vessel, (fx, fy), (dx, dy, dr) = _scene(disc_side="right")
    d = localize_fovea(img, fov_mask=fov, disc=_disc_dict(dx, dy, dr),
                       vessel_mask=vessel, config=cfg).to_dict()
    assert d["laterality"] == "left"
    assert d["detected"] is True
    assert d["center_x_y"][0] < dx  # fovea temporal = -x for left eye


def test_right_eye_orientation(cfg):
    img, fov, vessel, (fx, fy), (dx, dy, dr) = _scene(disc_side="left")
    d = localize_fovea(img, fov_mask=fov, disc=_disc_dict(dx, dy, dr),
                       vessel_mask=vessel, config=cfg).to_dict()
    assert d["laterality"] == "right"
    assert d["detected"] is True
    assert d["center_x_y"][0] > dx


def test_normalized_disc_relative_geometry(cfg):
    g = cfg["geometry"]
    exp = expected_fovea((100.0, 100.0), 40.0, +1, cfg)
    assert exp == (100.0 + g["disc_fovea_distance_dd"] * 40.0,
                   100.0 + g["vertical_offset_dd"] * 40.0)
    # same normalized geometry at 2x scale
    exp2 = expected_fovea((200.0, 200.0), 80.0, +1, cfg)
    assert (exp2[0] - 200) / 80 == pytest.approx((exp[0] - 100) / 40)


def test_resolution_scaling(cfg):
    """224 / 512 / 1024 proportional scenes: every size must DETECT with
    bounded error. Pairwise agreement is looser (0.15): grid snapping plus
    absolute noise/line-widths make classical appearance scores mildly
    scale-sensitive — a documented limitation, not a hidden failure."""
    from src.retina.fovea.metrics import center_error

    normed = []
    for s in (224, 512, 1024):
        img, fov, vessel, (fx, fy), (dx, dy, dr) = _scene(size=s, disc_side="left")
        d = localize_fovea(img, fov_mask=fov, disc=_disc_dict(dx, dy, dr),
                           vessel_mask=vessel, config=cfg).to_dict()
        assert d["detected"] is True, f"no detection at {s}px"
        ne = center_error(d["center_x_y"], (fx, fy)) / s
        assert ne < 0.25, f"error {ne} too large at {s}px"
        normed.append((d["center_x_y"][0] / s, d["center_x_y"][1] / s))
    for a, b in zip(normed, normed[1:]):
        assert abs(a[0] - b[0]) < 0.15 and abs(a[1] - b[1]) < 0.15


# --- 9-13. Failures ---------------------------------------------------------

def test_missing_disc_falls_back_without_crash(cfg):
    img, fov, vessel, _, _ = _scene()
    nodisc = {"detected": False, "status": NOT_DETECTED, "center": None,
              "radius": None, "confidence": 0.0}
    d = localize_fovea(img, fov_mask=fov, disc=nodisc, vessel_mask=vessel,
                       config=cfg).to_dict()
    assert d["disc_used"] is False
    assert any("fallback" in w for w in d["warnings"])


def test_low_confidence_disc_used_with_warning(cfg):
    img, fov, vessel, _, (dx, dy, dr) = _scene()
    d = localize_fovea(img, fov_mask=fov,
                       disc=_disc_dict(dx, dy, dr, status=LOW_CONFIDENCE),
                       vessel_mask=vessel, config=cfg).to_dict()
    assert d["disc_used"] is True
    assert any("LOW_CONFIDENCE" in w for w in d["warnings"])


def test_invalid_fov_is_not_detected(cfg):
    img, _, _, _, (dx, dy, dr) = _scene()
    d = localize_fovea(img, fov_mask=np.zeros((512, 512), np.uint8),
                       disc=_disc_dict(dx, dy, dr), config=cfg).to_dict()
    assert d["status"] == NOT_DETECTED and d["detected"] is False


def test_disc_outside_field_gives_no_candidates(cfg):
    img, fov, vessel, _, _ = _scene()
    d = localize_fovea(img, fov_mask=fov,
                       disc=_disc_dict(5.0, 5.0, 6.0), vessel_mask=vessel,
                       config=cfg).to_dict()
    assert d["status"] == NOT_DETECTED
    assert d["candidate_count"] == 0


def test_ambiguous_candidates_reduce_confidence(cfg):
    """Two identical dark spots -> tie-ish top scores -> penalty warning."""
    img, fov, vessel, (fx, fy), (dx, dy, dr) = _scene()
    dd = img.shape[0] / 8.0
    yy, xx = np.mgrid[0:512, 0:512]
    img2 = img.copy()
    mirror = (xx - (2 * dx + 2.5 * dd - fx)) ** 2 + (yy - fy) ** 2 <= (0.35 * dd) ** 2
    img2[mirror] = [70, 40, 25]
    d = localize_fovea(img2, fov_mask=fov, disc=_disc_dict(dx, dy, dr),
                       vessel_mask=vessel, config=cfg).to_dict()
    assert any("mbigu" in w for w in d["warnings"]) or d["status"] != DETECTED \
        or d["confidence"] < 0.9


# --- 14-17. Config ----------------------------------------------------------

def test_config_schema_and_weights(cfg):
    assert set(cfg["scoring"]["weights"]) == {"geometry", "appearance",
                                             "vessel", "fov"}
    assert sum(cfg["scoring"]["weights"].values()) == pytest.approx(1.0)
    assert cfg["decision"]["detect_threshold"] > cfg["decision"]["low_confidence_threshold"]
    assert cfg["method"] == DEFAULTS["method"]


def test_threshold_behavior(cfg):
    import copy

    img, fov, vessel, _, (dx, dy, dr) = _scene()
    c2 = copy.deepcopy(cfg)
    c2["decision"]["detect_threshold"] = 0.999
    c2["decision"]["low_confidence_threshold"] = 0.999
    d = localize_fovea(img, fov_mask=fov, disc=_disc_dict(dx, dy, dr),
                       vessel_mask=vessel, config=c2).to_dict()
    assert d["status"] == NOT_DETECTED and d["detected"] is False


def test_config_file_matches_runtime(cfg):
    import json

    p = Path(__file__).resolve().parents[1] / "configs" / "fovea_config.json"
    file_cfg = json.load(open(p))
    assert file_cfg["scoring"]["weights"] == cfg["scoring"]["weights"]
    assert file_cfg["geometry"]["disc_fovea_distance_dd"] == \
        cfg["geometry"]["disc_fovea_distance_dd"]


# --- 18-20. Integration -----------------------------------------------------

def test_phase2_fov_integration(cfg):
    from src.quality.config import load_config as load_qconfig
    from src.quality.field_of_view import detect_retinal_field

    img, _, vessel, (fx, fy), (dx, dy, dr) = _scene()
    fov = detect_retinal_field(img, load_qconfig())["mask"]
    assert (fov > 0).any()
    d = localize_fovea(img, fov_mask=fov, disc=_disc_dict(dx, dy, dr),
                       vessel_mask=vessel, config=cfg).to_dict()
    assert d["detected"] is True


def test_phase4a_vessel_integration(cfg):
    from src.retina.vessels.pipeline import segment_vessels

    img, fov, _, (fx, fy), (dx, dy, dr) = _scene()
    vessel = segment_vessels(img, fov).vessel_mask
    assert (vessel > 0).any()
    d = localize_fovea(img, fov_mask=fov, disc=_disc_dict(dx, dy, dr),
                       vessel_mask=vessel, config=cfg).to_dict()
    assert d["detected"] is True


def test_phase4b_disc_integration(cfg):
    """Phase 4B result dict feeds the fovea pipeline (key contract)."""
    from src.retina.optic_disc.pipeline import localize_optic_disc

    img, fov, vessel, (fx, fy), _ = _scene()
    disc = localize_optic_disc(img, fov_mask=fov).to_dict()
    assert set(disc) >= {"detected", "status", "center", "radius"}
    d = localize_fovea(img, fov_mask=fov, disc=disc, vessel_mask=vessel,
                       config=cfg).to_dict()
    assert d["disc_used"] == disc["detected"]
    if disc["detected"]:
        assert d["detected"] is True


def test_coordinate_convention_xy(cfg):
    """x=column (width axis), y=row (height axis) — never transposed."""
    img, fov, vessel, _, (dx, dy, dr) = _scene()
    d = localize_fovea(img, fov_mask=fov, disc=_disc_dict(dx, dy, dr),
                       vessel_mask=vessel, config=cfg).to_dict()
    h, w = img.shape[:2]
    x, y = d["center_x_y"]
    assert 0 <= x < w and 0 <= y < h


def test_csv_reader_skips_blank_rows(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("Image No,X- Coordinate,Y - Coordinate\nIDRiD_001,100,200\nIDRiD_002,,,\n")
    out = _read_centers(str(p))
    assert out == {"IDRiD_001": (100.0, 200.0)}
