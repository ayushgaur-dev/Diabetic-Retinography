"""Vessel-segmentation tests — SIH26038 Phase 4A.

Unit tests use a deterministic synthetic vessel fixture (dark lines on a
red/orange disc) and NEVER require DRIVE. Dataset-dependent evaluation is
covered by the evaluate CLI, not by this suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retina.vessels import preprocessing as pre
from src.retina.vessels import segmentation as seg
from src.retina.vessels.config import DEFAULTS, load_vessel_config
from src.retina.vessels.dataset import discover_split
from src.retina.vessels.enhancement import enhance_vessels
from src.retina.vessels.metrics import aggregate, evaluate_mask
from src.retina.vessels.pipeline import segment_vessels


# ---------------------------------------------------------------------------
# Synthetic vessel fixture: dark branching lines on a fundus-like disc
# ---------------------------------------------------------------------------

def _synthetic_vessel_image(size=224, seed=5):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    disc = (xx - c) ** 2 + (yy - c) ** 2 <= (size // 2 - 12) ** 2
    img[disc] = [150, 100, 60]
    gt = np.zeros((size, size), dtype=np.uint8)
    for _ in range(14):
        x0, y0 = rng.integers(c - 70, c + 70, 2)
        x1, y1 = x0 + rng.integers(-60, 60), y0 + rng.integers(-60, 60)
        import cv2

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (60, 25, 15), 2)
        cv2.line(gt, (int(x0), int(y0)), (int(x1), int(y1)), 255, 2)
    img = img.astype(np.float32) + rng.normal(0, 4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    gt[~disc] = 0
    fov = (disc.astype(np.uint8)) * 255
    return img, gt, fov


@pytest.fixture(scope="module")
def fixture():
    return _synthetic_vessel_image()


# ---------------------------------------------------------------------------
# 1-3. Dataset discovery / pairing / missing files (fake DRIVE tree)
# ---------------------------------------------------------------------------

def _fake_drive(tmp_path):
    import PIL.Image as I

    d = tmp_path / "DRIVE"
    (d / "training" / "images").mkdir(parents=True)
    (d / "training" / "1st_manual").mkdir(parents=True)
    (d / "training" / "mask").mkdir(parents=True)
    (d / "test" / "images").mkdir(parents=True)
    (d / "test" / "mask").mkdir(parents=True)
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    I.fromarray(tiny).save(d / "training" / "images" / "21_training.tif")
    I.fromarray(tiny[:, :, 0]).save(d / "training" / "1st_manual" / "21_manual1.gif")
    I.fromarray(tiny[:, :, 0]).save(d / "training" / "mask" / "21_training_mask.gif")
    I.fromarray(tiny).save(d / "test" / "images" / "01_test.tif")
    I.fromarray(tiny[:, :, 0]).save(d / "test" / "mask" / "01_test_mask.gif")
    return d


def test_dataset_discovery_and_pairing(tmp_path):
    d = _fake_drive(tmp_path)
    train = discover_split(d, "training")
    test = discover_split(d, "test")
    assert len(train) == 1 and train[0]["id"] == "21"
    assert train[0]["manual"].endswith("21_manual1.gif")
    assert train[0]["fov"].endswith("21_training_mask.gif")
    assert len(test) == 1 and test[0]["manual"] is None  # no test GT in mirror


def test_missing_annotation_fails_clearly(tmp_path):
    import PIL.Image as I

    d = tmp_path / "DRIVE"
    (d / "training" / "images").mkdir(parents=True)
    (d / "training" / "1st_manual").mkdir(parents=True)
    (d / "training" / "mask").mkdir(parents=True)
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    I.fromarray(tiny).save(d / "training" / "images" / "21_training.tif")
    with pytest.raises(FileNotFoundError, match="21_manual1"):
        discover_split(d, "training")


def test_unknown_split_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown DRIVE split"):
        discover_split(tmp_path, "validation")


# ---------------------------------------------------------------------------
# 4-6. Green channel / enhancement / segmentation on synthetic fixture
# ---------------------------------------------------------------------------

def test_green_channel_properties(fixture):
    img, _, _ = fixture
    g = pre.extract_green(img)
    assert g.shape == img.shape[:2] and g.dtype == np.float32
    assert np.isfinite(g).all() and 0.0 <= g.min() and g.max() <= 1.0


def test_green_channel_rejects_bad_input():
    with pytest.raises(ValueError):
        pre.extract_green(np.zeros((10, 10), dtype=np.uint8))


def test_enhancement_response_nonempty(fixture):
    img, _, fov = fixture
    green = pre.preprocess_green(img, fov)
    resp = enhance_vessels(green, fov, scales=(3, 5, 9))
    assert resp.shape == img.shape[:2] and resp.dtype == np.float32
    assert np.isfinite(resp).all() and float(resp.max()) > 0


def test_segmentation_output_valid(fixture):
    img, _, fov = fixture
    green = pre.preprocess_green(img, fov)
    resp = enhance_vessels(green, fov)
    mask = seg.segment_response(resp, fov, "percentile", 89.0)
    assert mask.shape == img.shape[:2] and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert (mask > 0).any(), "synthetic vessels must produce a non-empty mask"
    assert (mask[fov == 0] == 0).all(), "no vessels outside FOV"


# ---------------------------------------------------------------------------
# 7-10. Pipeline integration + density
# ---------------------------------------------------------------------------

def test_pipeline_integration(fixture):
    img, _, fov = fixture
    res = segment_vessels(img, fov)
    assert res.vessel_mask.shape == img.shape[:2]
    assert res.vessel_mask.dtype == np.uint8
    assert 0.0 <= res.vessel_density <= 1.0
    assert res.vessel_area == int((res.vessel_mask > 0).sum())
    assert res.processing_time_ms > 0
    assert res.method == "multiscale_tophat_classical"
    assert res.fov_source == "provided"
    d = res.to_dict()
    assert d["response_finite"] and d["response_nonempty"]


def test_pipeline_finds_synthetic_vessels(fixture):
    img, gt, fov = fixture
    res = segment_vessels(img, fov)
    m = evaluate_mask(res.vessel_mask, gt, fov)
    assert m["sensitivity"] > 0.3, f"recall too low on synthetic vessels: {m}"
    assert m["specificity"] > 0.8


def test_vessel_density_fov_restricted(fixture):
    img, _, fov = fixture
    res = segment_vessels(img, fov)
    expected = (res.vessel_mask > 0).sum() / (fov > 0).sum()
    assert res.vessel_density == pytest.approx(expected)


def test_pipeline_fallback_fov_labelled(fixture):
    img, _, _ = fixture
    res = segment_vessels(img, None)
    assert res.fov_source == "phase2_fallback"
    assert any("fallback" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# 11-12. Metrics incl. FOV restriction
# ---------------------------------------------------------------------------

def test_metrics_known_case():
    pred = np.array([[255, 0], [255, 255]], dtype=np.uint8)
    gt = np.array([[255, 255], [0, 255]], dtype=np.uint8)
    fov = np.full((2, 2), 255, dtype=np.uint8)
    m = evaluate_mask(pred, gt, fov)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 0)
    assert m["recall"] == m["sensitivity"] == pytest.approx(2 / 3)
    assert m["dice"] == m["f1"] == pytest.approx(4 / 6)
    assert m["iou"] == pytest.approx(2 / 4)
    assert m["accuracy"] == pytest.approx(2 / 4)


def test_metrics_ignore_background():
    pred = np.full((4, 4), 255, dtype=np.uint8)
    gt = np.zeros((4, 4), dtype=np.uint8)
    fov = np.zeros((4, 4), dtype=np.uint8)
    fov[0:2, 0:2] = 255
    m = evaluate_mask(pred, gt, fov)
    assert m["n_fov"] == 4 and m["tn"] == 0 and m["fp"] == 4


def test_aggregate_micro_macro():
    per = [
        {"sensitivity": 0.5, "specificity": 1.0, "precision": 1.0, "f1": 0.667,
         "dice": 0.667, "iou": 0.5, "accuracy": 0.9, "tp": 5, "fp": 0,
         "fn": 5, "tn": 90},
        {"sensitivity": 1.0, "specificity": 1.0, "precision": 1.0, "f1": 1.0,
         "dice": 1.0, "iou": 1.0, "accuracy": 1.0, "tp": 10, "fp": 0,
         "fn": 0, "tn": 90},
    ]
    agg = aggregate(per)
    assert agg["n_images"] == 2
    assert agg["micro"]["sensitivity"] == pytest.approx(15 / 20)  # pooled
    assert agg["macro"]["sensitivity_mean"] == pytest.approx(0.75)  # averaged


# ---------------------------------------------------------------------------
# 14. Configuration
# ---------------------------------------------------------------------------

def test_config_defaults_and_override(tmp_path):
    cfg = load_vessel_config(path=tmp_path / "missing.json")
    assert cfg["segmentation"]["percentile"] == DEFAULTS["segmentation"]["percentile"]
    assert cfg["enhancement"]["scales"] == DEFAULTS["enhancement"]["scales"]
    import json

    p = tmp_path / "v.json"
    p.write_text(json.dumps({"segmentation": {"percentile": 50.0}}))
    cfg2 = load_vessel_config(path=p)
    assert cfg2["segmentation"]["percentile"] == 50.0
    assert cfg2["postprocessing"]["min_component_size"] == \
        DEFAULTS["postprocessing"]["min_component_size"]


def test_algorithm_parameters_configurable(fixture):
    img, _, fov = fixture
    c1 = load_vessel_config()
    c2 = load_vessel_config()
    c2["segmentation"]["percentile"] = 50.0
    r1 = segment_vessels(img, fov, config=c1)
    r2 = segment_vessels(img, fov, config=c2)
    assert r2.vessel_area > r1.vessel_area, "percentile must change output"
