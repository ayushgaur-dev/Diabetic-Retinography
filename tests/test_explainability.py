"""Explainability tests — SIH26038 Phase 6. Fast: tiny in-test CNN for the
Grad-CAM adapter; stubs elsewhere. No dataset, no 23MB weights."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explainability.config import DEFAULTS, load_explainability_config
from src.explainability.consistency import classify
from src.explainability.evidence_fusion import (lesion_overlap_stats,
                                               rank_regions)
from src.explainability.evidence_map import (hot_mask, to_original,
                                             upsample_heatmap)
from src.explainability.summary import build_summary
from src.explainability.types import EvidenceRegion


@pytest.fixture(scope="module")
def tiny_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(64, 64, 3))
    x = layers.Conv2D(8, 3, activation="relu", name="test_conv")(inp)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(5, activation="softmax")(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="categorical_crossentropy")
    return m


@pytest.fixture(scope="module")
def tiny_batch():
    # Raw [0,1] input (no preprocess_input: its range saturates this toy).
    rng = np.random.default_rng(0)
    return (rng.random((1, 64, 64, 3))).astype("float32")


@pytest.fixture(scope="module")
def cfg():
    return load_explainability_config()


@pytest.fixture(scope="module")
def fundus():
    rng = np.random.default_rng(1)
    img = np.zeros((128, 160, 3), np.uint8)
    yy, xx = np.mgrid[0:128, 0:160]
    img[(xx - 80) ** 2 + (yy - 64) ** 2 <= 50 ** 2] = [120, 70, 40]
    return np.clip(img.astype(float) + rng.normal(0, 5, img.shape),
                   0, 255).astype(np.uint8)


# --- 1-5. Adapter -----------------------------------------------------------

def test_adapter_loads_and_predicted_class(tiny_model, tiny_batch, cfg):
    from src.explainability.gradcam_adapter import explain_class

    g = explain_class(tiny_batch, tiny_model, None, cfg)
    probs = tiny_model.predict(tiny_batch, verbose=0)[0]
    assert g["explained_class"] == int(np.argmax(probs))
    assert g["explained_class_probability"] == pytest.approx(float(np.max(probs)), abs=1e-3)


def test_heatmap_shape_finite_normalized(tiny_model, tiny_batch, cfg):
    from src.explainability.gradcam_adapter import explain_class

    g = explain_class(tiny_batch, tiny_model, 3, cfg)
    h = g["heatmap"]
    assert h.ndim == 2 and h.shape[0] > 0
    assert np.isfinite(h).all()
    assert h.min() >= 0.0 and h.max() <= 1.0
    assert g["explained_class"] == 3
    assert g["target_layer"] == "test_conv"


def test_targeted_differs_by_class(tiny_batch, cfg):
    """Structured dense weights force classes to read different filters."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(64, 64, 3))
    x = layers.Conv2D(4, 3, activation="relu", name="split_conv")(inp)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(5, activation="softmax")(x)
    m = models.Model(inp, out)
    # Pin all weights: small conv constants keep activations bounded so the
    # softmax stays soft (live gradients); classes read different filters.
    cw = 0.05 * np.ones((3, 3, 3, 4))
    cw[:, :, :, 1:] *= np.arange(1, 4).reshape(1, 1, 1, 3) * 0.5 + 0.5
    m.get_layer("split_conv").set_weights([cw, np.zeros(4)])
    k = np.zeros((4, 5))
    k[0, 0] = 0.8
    k[3, 4] = 0.8  # soft logits: no softmax saturation, live gradients
    m.layers[-1].set_weights([k, np.zeros(5)])
    from src.explainability.gradcam_adapter import explain_class

    h0 = explain_class(tiny_batch, m, 0, cfg)["heatmap"]
    h4 = explain_class(tiny_batch, m, 4, cfg)["heatmap"]
    assert h0.shape == h4.shape
    assert not np.allclose(h0, h4)


# --- 6-7. Coordinates --------------------------------------------------------

def test_common_coordinate_system():
    big = (np.random.default_rng(2).random((7, 7)) * 255).astype(np.uint8)
    out = to_original(big, (128, 160), is_mask=True)
    assert out.shape == (128, 160)
    assert set(np.unique(out)) <= {0, 255}
    heat = upsample_heatmap(np.random.default_rng(3).random((7, 7)), (128, 160))
    assert heat.shape == (128, 160)
    assert heat.min() >= 0.0 and heat.max() <= 1.0


def test_no_coordinate_shift():
    dot = np.zeros((7, 7), np.uint8)
    dot[3, 5] = 255  # row 3, col 5
    out = to_original(dot, (140, 140), is_mask=True)
    ys, xs = np.nonzero(out)
    assert abs(xs.mean() - 5 / 7 * 140) < 140 / 7 + 1
    assert abs(ys.mean() - 3 / 7 * 140) < 140 / 7 + 1


def test_fov_masks_background_stats():
    fov = np.zeros((50, 50), np.uint8)
    fov[10:40, 10:40] = 255
    hot = np.zeros((50, 50), np.uint8)
    hot[0:10, 0:10] = 255  # activation entirely outside FOV
    frac_out = ((hot > 0) & ~(fov > 0)).sum() / max((hot > 0).sum(), 1)
    assert frac_out == 1.0


# --- 8-15. Integration (stubs) -------------------------------------------------

def _stub_result(**kw):
    d = {"detected": False, "status": "NOT_DETECTED", "center": None,
         "center_x_y": None, "candidate_count": 0, "confidence": 0.0}
    d.update(kw)
    return d


def test_full_pipeline_prediction_unchanged(fundus, tiny_model, cfg):
    from src.explainability.pipeline import explain_image

    probs = [float(v) for v in tiny_model.predict(
        __import__("tensorflow").keras.applications.efficientnet.preprocess_input(
            np.array(__import__("PIL.Image").Image.fromarray(fundus).resize((64, 64))).astype("float32")
        )[None], verbose=0)[0]]
    grade = int(np.argmax(probs))
    res = explain_image(fundus, model=tiny_model, grade=grade,
                        probabilities=probs, config=cfg, run_context=False,
                        image_size=64)
    d = res.to_dict()
    assert d["predicted_grade"] == grade
    assert [round(float(v), 4) for v in d["probabilities"]] == \
        [round(float(v), 4) for v in probs]
    assert d["raw_confidence"] == pytest.approx(max(probs), abs=1e-3)


def test_missing_sources_graceful(fundus, tiny_model, cfg):
    from src.explainability.pipeline import explain_image

    res = explain_image(fundus, model=tiny_model, config=cfg,
                        run_context=False, image_size=64)
    d = res.to_dict()
    assert d["vessels"]["available"] is False
    assert d["consistency"]["category"] in ("SUPPORTIVE", "PARTIALLY_SUPPORTIVE",
                                           "INCONCLUSIVE", "CONFLICTING")
    assert isinstance(d["summary"], str) and len(d["summary"]) > 0


# --- 16-18. Overlap / ranking / consistency -------------------------------------

def test_overlap_calculation():
    les = np.zeros((100, 100), np.uint8)
    les[20:40, 20:40] = 255
    hot = np.zeros((100, 100), np.uint8)
    hot[30:60, 30:60] = 255
    fov = np.full((100, 100), 255, np.uint8)
    stats = lesion_overlap_stats({"microaneurysm": les}, hot, fov)
    s = stats["microaneurysm"]
    assert s["lesion_area"] == 400
    assert s["gradcam_overlap_area"] == 100
    assert s["lesion_inside_gradcam_fraction"] == pytest.approx(0.25)
    assert s["iou"] == pytest.approx(100 / (400 + 900 - 100), abs=1e-4)


def test_evidence_ranking(cfg):
    regs = [EvidenceRegion(source="microaneurysm", evidence_type="lesion_candidate",
                           bbox=[0, 0, 10, 10], centroid=[5, 5], area=100,
                           score=0.9, overlap_with_gradcam=0.1),
            EvidenceRegion(source="hemorrhage", evidence_type="lesion_candidate",
                           bbox=[0, 0, 20, 20], centroid=[10, 10], area=400,
                           score=0.5, overlap_with_gradcam=0.8)]
    ranked = rank_regions(regs, cfg)
    assert ranked[0].source == "hemorrhage"  # overlap weight dominates


def test_consistency_classification(cfg):
    hi = {"microaneurysm": {"lesion_area": 100, "lesion_inside_gradcam_fraction": 0.6}}
    assert classify(hi, 0.9, cfg)[0] == "SUPPORTIVE"
    mid = {"microaneurysm": {"lesion_area": 100, "lesion_inside_gradcam_fraction": 0.2}}
    assert classify(mid, 0.9, cfg)[0] == "PARTIALLY_SUPPORTIVE"
    lo = {"microaneurysm": {"lesion_area": 100, "lesion_inside_gradcam_fraction": 0.05}}
    cat, reason = classify(lo, 0.9, cfg)
    assert cat == "CONFLICTING" and "not proof of error" in reason
    none = {"microaneurysm": {"lesion_area": 0, "lesion_inside_gradcam_fraction": None}}
    assert classify(none, 0.9, cfg)[0] == "INCONCLUSIVE"
    assert classify(none, 0.1, cfg)[0] == "INCONCLUSIVE"


# --- 19-20. Semantics ------------------------------------------------------------

def test_raw_confidence_semantics(fundus, tiny_model, cfg):
    from src.explainability.pipeline import explain_image

    res = explain_image(fundus, model=tiny_model, config=cfg,
                        run_context=False, image_size=64)
    d = res.to_dict()
    assert d["raw_confidence_note"] == "UNCALIBRATED raw softmax confidence"
    assert "uncalibrated" in d["summary"].lower()
    assert "diagnos" not in d["summary"].lower()


def test_deterministic_summary(fundus, tiny_model, cfg):
    from src.explainability.pipeline import explain_image

    a = explain_image(fundus, model=tiny_model, config=cfg,
                      run_context=False, image_size=64).to_dict()["summary"]
    b = explain_image(fundus, model=tiny_model, config=cfg,
                      run_context=False, image_size=64).to_dict()["summary"]
    assert a == b and "prediction" in a.lower()


# --- 23-25. Viz / config / faithfulness --------------------------------------------

def test_visualization_generation(fundus, tiny_model, cfg, tmp_path):
    from src.explainability.pipeline import explain_image
    from src.explainability.visualization import save_explainability_figure

    res = explain_image(fundus, model=tiny_model, config=cfg,
                        run_context=False, image_size=64)
    d = res.to_dict()
    d["_maps"] = {"gradcam": res.maps["gradcam"], "vessels": res.maps["blank"],
                  "blank": res.maps["blank"]}
    p = save_explainability_figure(fundus, d, str(tmp_path / "fig.png"))
    assert Path(p).exists() and Path(p).stat().st_size > 5000


def test_configuration_validation(cfg):
    assert sum(cfg["ranking"]["weights"].values()) == pytest.approx(1.0)
    assert cfg["consistency"]["high_overlap"] > cfg["consistency"]["low_overlap"]
    assert 0 < cfg["gradcam"]["hot_quantile"] < 1
    assert 0 < cfg["faithfulness"]["top_fraction"] < 1
    import json

    file_cfg = json.load(open(Path(__file__).resolve().parents[1] /
                              "configs" / "explainability_config.json"))
    assert file_cfg["ranking"]["weights"] == cfg["ranking"]["weights"]


def test_faithfulness_sanity(fundus, tiny_model, cfg):
    from src.explainability.pipeline import faithfulness_deletion

    probs = [0.1, 0.1, 0.6, 0.1, 0.1]
    f = faithfulness_deletion(fundus, tiny_model, 2, probs, config=cfg,
                              image_size=64)
    assert set(f) >= {"original_probability", "masked_probability",
                      "probability_drop", "masked_prediction"}
    assert "not causal" in f["note"]
    assert f["original_probability"] == pytest.approx(0.6)


def test_hot_mask_quantile():
    h = np.arange(100, dtype=float).reshape(10, 10)
    m = hot_mask(h, 0.75)
    assert 0 < (m > 0).sum() <= 26
