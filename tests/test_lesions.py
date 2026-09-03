"""Lesion tests — SIH26038 Phase 4D. Synthetic fixtures only (no dataset
dependency); IDRiD evaluation is covered by the evaluate CLI."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retina.lesions.config import DEFAULTS, load_lesion_config
from src.retina.lesions.dataset import LESION_FILES, discover, load_mask
from src.retina.lesions.exudates import (detect_hard_exudates,
                                         detect_soft_exudates)
from src.retina.lesions.hemorrhage import detect_hemorrhages
from src.retina.lesions.microaneurysm import detect_microaneurysms
from src.retina.lesions.metrics import evaluate_mask, object_metrics
from src.retina.lesions.pipeline import detect_lesions
from src.retina.lesions.postprocessing import (circularity, cleanup_mask,
                                               components)
from src.retina.lesions.preprocessing import (lesion_preprocess,
                                              to_working_resolution)
from src.retina.lesions.types import (DETECTED, LESION_TYPES, LOW_CONFIDENCE,
                                      NOT_DETECTED)


# ---------------------------------------------------------------------------
# Synthetic fundus with painted lesions (x=column, y=row)
# ---------------------------------------------------------------------------

def _scene(size=512, seed=31, ma=(), he=(), ex=(), se=()):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    fov = ((xx - c) ** 2 + (yy - c) ** 2 <= (size // 2 - 8) ** 2)
    img[fov] = [120, 75, 45]
    vessel = np.zeros((size, size), dtype=np.uint8)
    for _ in range(10):
        x0, y0 = rng.integers(c - 150, c + 150, 2)
        cv2.line(img, (int(x0), int(y0)),
                 (int(x0 + rng.integers(-60, 60)), int(y0 + rng.integers(-60, 60))),
                 (60, 25, 15), 3)
        cv2.line(vessel, (int(x0), int(y0)),
                 (int(x0 + rng.integers(-60, 60)), int(y0 + rng.integers(-60, 60))),
                 255, 3)
    for (x, y, r) in ma:
        cv2.circle(img, (x, y), r, (45, 20, 12), -1)
    for (x, y, w, h) in he:
        cv2.ellipse(img, (x, y), (w, h), 30, 0, 360, (50, 18, 10), -1)
    for (x, y, r) in ex:
        cv2.circle(img, (x, y), r, (235, 215, 90), -1)
    for (x, y, w, h) in se:
        cv2.ellipse(img, (x, y), (w, h), 15, 0, 360, (225, 215, 200), -1)
    img = img.astype(np.float32) + rng.normal(0, 4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, (fov.astype(np.uint8)) * 255, vessel


@pytest.fixture(scope="module")
def cfg():
    return load_lesion_config()


@pytest.fixture(scope="module")
def scene():
    return _scene(ma=[(150, 150, 4), (350, 300, 5)],
                  he=[(300, 150, 14, 9)],
                  ex=[(180, 350, 10)],
                  se=[(350, 180, 22, 14)])


def _ctx(img, fov, vessel):
    ww = img.shape[1]
    green, _ = lesion_preprocess(img, fov)
    dist = cv2.distanceTransform((255 - vessel), cv2.DIST_L2, 3)
    return green, dist, ww


# --- 1-6. General -----------------------------------------------------------

def test_config_loads_with_required_sections(cfg):
    for section in ("microaneurysm", "hemorrhage", "hard_exudate",
                    "soft_exudate", "common", "optic_disc_exclusion",
                    "candidate_scoring"):
        assert section in cfg


def test_result_schema(cfg, scene):
    img, fov, vessel = scene
    out = detect_lesions(img, fov_mask=fov, vessel_mask=vessel,
                         disc={"detected": False, "status": NOT_DETECTED,
                               "center": None, "radius": None, "confidence": 0.0},
                         fovea={"center_x_y": None}, config=cfg)
    assert set(out) >= set(LESION_TYPES)
    for lt in LESION_TYPES:
        d = out[lt].to_dict()
        assert set(d) >= {"lesion_type", "status", "candidate_count",
                          "candidates", "total_evidence_area", "confidence",
                          "method", "warnings"}
        assert d["status"] in (DETECTED, LOW_CONFIDENCE, NOT_DETECTED)
        for c in d["candidates"]:
            assert set(c) >= {"candidate_id", "bbox", "centroid_x_y",
                              "area", "score", "features"}
            x, y = c["centroid_x_y"]
            assert 0 <= x < img.shape[1] and 0 <= y < img.shape[0]


def test_deterministic_output(cfg, scene):
    img, fov, vessel = scene
    kw = dict(fov_mask=fov, vessel_mask=vessel, config=cfg)
    a = {lt: detect_lesions(img, **kw)[lt].to_dict() for lt in LESION_TYPES}
    b = {lt: detect_lesions(img, **kw)[lt].to_dict() for lt in LESION_TYPES}
    assert a == b


def test_fov_masks_background(cfg):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = [120, 75, 45]
    fov = np.zeros((200, 200), np.uint8)
    fov[80:120, 80:120] = 255  # tiny field only
    out = detect_lesions(img, fov_mask=fov, vessel_mask=np.zeros((200, 200), np.uint8),
                         disc={"detected": False, "status": NOT_DETECTED,
                               "center": None, "radius": None, "confidence": 0.0},
                         fovea={"center_x_y": None}, config=cfg)
    for lt in LESION_TYPES:
        m = out[lt].evidence_mask
        assert (m[fov == 0] == 0).all(), f"{lt} leaks outside FOV"


def test_coordinate_convention_xy(cfg, scene):
    img, fov, vessel = scene
    out = detect_lesions(img, fov_mask=fov, vessel_mask=vessel, config=cfg)
    h, w = img.shape[:2]
    for lt in LESION_TYPES:
        for c in out[lt].to_dict()["candidates"]:
            (x0, y0, x1, y1), (x, y) = c["bbox"], c["centroid_x_y"]
            assert 0 <= x0 <= x <= x1 < w and 0 <= y0 <= y <= y1 < h


def test_working_resolution_preserves_detail():
    big = np.zeros((800, 1200, 3), np.uint8)
    small, scale = to_working_resolution(big, 1072)
    assert small.shape[1] == 1072 and scale == pytest.approx(1072 / 1200)
    tiny = np.zeros((100, 100, 3), np.uint8)
    same, s2 = to_working_resolution(tiny, 1072)
    assert s2 == 1.0 and same.shape == tiny.shape


# --- 7-17. Per-type detectors ------------------------------------------------

def test_microaneurysm_finds_dark_blobs(cfg, scene):
    img, fov, vessel = scene
    green, dist, w = _ctx(img, fov, vessel)
    mask, cands = detect_microaneurysms(img, fov, vessel, dist, green, cfg,
                                        cfg["common"], w)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 255}
    assert len(cands) > 0
    assert all("equivalent_diameter" in c["features"] for c in cands)


def test_microaneurysm_vessel_veto(cfg):
    img, fov, vessel = _scene(ma=[(256, 256, 5)])
    green, dist, w = _ctx(img, fov, vessel)
    c1 = cfg
    import copy

    c2 = copy.deepcopy(cfg)
    c2["microaneurysm"]["max_vessel_overlap"] = 0.0  # strict veto
    _, a = detect_microaneurysms(img, fov, vessel, dist, green, c1, c1["common"], w)
    _, b = detect_microaneurysms(img, fov, vessel, dist, green, c2, c2["common"], w)
    assert len(b) <= len(a)


def test_microaneurysm_size_filter(cfg, scene):
    img, fov, vessel = scene
    green, dist, w = _ctx(img, fov, vessel)
    _, cands = detect_microaneurysms(img, fov, vessel, dist, green, cfg,
                                     cfg["common"], w)
    from src.retina.lesions.config import scaled_area

    lo = scaled_area(cfg["microaneurysm"]["min_area"], w, 1072)
    hi = scaled_area(cfg["microaneurysm"]["max_area"], w, 1072)
    assert all(lo <= c["area"] <= hi for c in cands)


def test_hemorrhage_finds_large_dark(cfg, scene):
    img, fov, vessel = scene
    green, dist, w = _ctx(img, fov, vessel)
    mask, cands = detect_hemorrhages(img, fov, vessel, dist, green, cfg,
                                     cfg["common"], w)
    assert len(cands) > 0
    mas, _ = detect_microaneurysms(img, fov, vessel, dist, green, cfg,
                                   cfg["common"], w)
    he_areas = [c["area"] for c in cands]
    assert max(he_areas) > 100, "hemorrhage detector must admit large regions"


def test_hemorrhage_separate_from_ma(cfg):
    import copy

    assert copy.deepcopy(cfg)["hemorrhage"]["min_area"] > \
        cfg["microaneurysm"]["min_area"]
    assert cfg["hemorrhage"]["min_circularity"] < cfg["microaneurysm"]["min_circularity"]


def test_hard_exudate_finds_yellow(cfg, scene):
    img, fov, vessel = scene
    w = img.shape[1]
    _, cands = detect_hard_exudates(img, fov, None, None, None, cfg,
                                    cfg["common"], w)
    assert len(cands) > 0
    assert all("mean_yellowness" in c["features"] for c in cands)


def test_hard_exudate_disc_exclusion(cfg):
    img, fov, vessel = _scene(ex=[(256, 200, 12)])
    w = img.shape[1]
    excl = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.circle(excl, (256, 200), 40, 255, -1)  # cover the exudate fully
    _, with_excl = detect_hard_exudates(img, fov, excl, (256, 200), None, cfg,
                                        cfg["common"], w)
    _, without = detect_hard_exudates(img, fov, None, None, None, cfg,
                                      cfg["common"], w)
    assert len(with_excl) < len(without)


def test_soft_exudate_texture_path(cfg, scene):
    img, fov, vessel = scene
    w = img.shape[1]
    _, cands = detect_soft_exudates(img, fov, None, None, None, cfg,
                                    cfg["common"], w)
    assert all("mean_texture" in c["features"] for c in cands)


# --- 18-23. Integration ------------------------------------------------------

def test_vessel_context_flows(cfg, scene):
    img, fov, vessel = scene
    out = detect_lesions(img, fov_mask=fov, vessel_mask=vessel, config=cfg)
    assert out["microaneurysm"].to_dict()["candidates"] != []  # sanity
    feats = out["microaneurysm"].to_dict()["candidates"][0]["features"]
    assert "vessel_overlap" in feats and "distance_to_vessel" in feats


def test_missing_vessel_fallback(cfg, scene):
    img, fov, _ = scene
    out = detect_lesions(img, fov_mask=fov, vessel_mask=None, config=cfg)
    assert all(lt in out for lt in LESION_TYPES)  # neutral fallback, no crash


def test_missing_disc_fallback(cfg, scene):
    img, fov, vessel = scene
    out = detect_lesions(img, fov_mask=fov, vessel_mask=vessel,
                         disc={"detected": False, "status": NOT_DETECTED,
                               "center": None, "radius": None, "confidence": 0.0},
                         config=cfg)
    assert out["hard_exudate"].to_dict()["candidate_count"] >= 0
    assert any("disc" in w.lower() for w in out["hard_exudate"].warnings)


def test_missing_fovea_fallback(cfg, scene):
    img, fov, vessel = scene
    out = detect_lesions(img, fov_mask=fov, vessel_mask=vessel,
                         fovea={"center_x_y": None}, config=cfg)
    for lt in LESION_TYPES:
        for c in out[lt].to_dict()["candidates"]:
            assert c["features"]["distance_to_fovea"] == -1.0


def test_fovea_disc_distance_features(cfg, scene):
    img, fov, vessel = scene
    out = detect_lesions(img, fov_mask=fov, vessel_mask=vessel,
                         disc={"detected": True, "status": "DETECTED",
                               "center": [100, 100], "radius": 30, "confidence": 0.9},
                         fovea={"center_x_y": [400, 400]}, config=cfg)
    for lt in ("hard_exudate", "soft_exudate"):
        for c in out[lt].to_dict()["candidates"]:
            assert c["features"]["distance_to_disc"] >= 0
            assert c["features"]["distance_to_fovea"] >= 0


# --- 24-26. Resolution --------------------------------------------------------

@pytest.mark.parametrize("size", [224, 512, 1024])
def test_detector_runs_at_resolutions(cfg, size):
    img, fov, vessel = _scene(size=size, ma=[(size // 3, size // 3, max(2, size // 128))],
                              ex=[(2 * size // 3, 2 * size // 3, max(4, size // 51))])
    out = detect_lesions(img, fov_mask=fov, vessel_mask=vessel, config=cfg)
    for lt in LESION_TYPES:
        m = out[lt].evidence_mask
        assert m.shape == (fov.shape[0], fov.shape[1])
        assert m.dtype == np.uint8
        assert np.isfinite(m.astype(float)).all()


# --- 27-30. Failure ----------------------------------------------------------

def test_blank_image_no_crash(cfg):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    out = detect_lesions(img, config=cfg)
    for lt in LESION_TYPES:
        d = out[lt].to_dict()
        assert d["status"] in (LOW_CONFIDENCE, NOT_DETECTED, DETECTED)


def test_no_fov_no_crash(cfg, scene):
    img, _, vessel = scene
    out = detect_lesions(img, fov_mask=np.zeros((512, 512), np.uint8),
                         vessel_mask=vessel, config=cfg)
    assert set(out) >= set(LESION_TYPES)


def test_malformed_image_rejected(cfg):
    with pytest.raises(ValueError):
        detect_lesions(np.zeros((100, 100), dtype=np.uint8), config=cfg)
    with pytest.raises(ValueError):
        detect_lesions("not-an-image", config=cfg)


def test_empty_annotation_metrics():
    gt = np.zeros((50, 50), np.uint8)
    pred = np.zeros((50, 50), np.uint8)
    fov = np.full((50, 50), 255, np.uint8)
    m = evaluate_mask(pred, gt, fov)
    assert m["tn"] == 2500 and m["tp"] == 0
    from src.retina.lesions.metrics import object_metrics

    o = object_metrics(pred, gt, fov)
    assert o["n_gt_objects"] == 0 and o["n_pred_objects"] == 0


def test_object_matching_rule():
    from src.retina.lesions.metrics import object_metrics

    gt = np.zeros((40, 40), np.uint8)
    gt[10:13, 10:13] = 255  # GT object, centroid (11, 11)
    pred = np.zeros((40, 40), np.uint8)
    pred[11, 11] = 255  # single overlapping pixel
    fov = np.full((40, 40), 255, np.uint8)
    o = object_metrics(pred, gt, fov)
    assert o["object_recall"] == 1.0 and o["object_precision"] == 1.0
    pred2 = np.zeros((40, 40), np.uint8)
    pred2[30, 30] = 255
    o2 = object_metrics(pred2, gt, fov)
    assert o2["object_recall"] == 0.0 and o2["object_precision"] == 0.0


def test_circularity_clamped():
    assert circularity(10, 1.0) == 1.0  # degenerate tiny perimeter
    assert 0.0 <= circularity(100, 40.0) <= 1.0
    assert circularity(0, 0) == 0.0


def test_cleanup_keeps_fov_only():
    m = np.full((30, 30), 255, np.uint8)
    fov = np.zeros((30, 30), np.uint8)
    fov[5:25, 5:25] = 255
    out = cleanup_mask(m, fov)
    assert (out[fov == 0] == 0).all() and (out[5:25, 5:25] == 255).all()
