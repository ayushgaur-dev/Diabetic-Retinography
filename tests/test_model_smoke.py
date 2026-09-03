"""Baseline smoke tests — Phase 1 (SIH26038).

Verifies the vendored baseline end-to-end on a SYNTHETIC fundus-like image
(no clinical meaning; exercises plumbing only):
  weights file -> model loads -> 5-class output -> inference distribution
  -> Grad-CAM heatmap + overlay -> deterministic triage decision.

Skipped automatically when the (gitignored, Hugging Face-hosted) weights
file is absent. Does NOT retrain anything. Raw softmax is asserted as a
valid probability distribution only — it is NOT calibrated confidence
(known gap, SIH Phase 7).
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = REPO_ROOT / "models" / "efficientnetb0_finetuned_patched.keras"

needs_weights = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"weights not present at {MODEL_PATH} (gitignored; see docs/MODEL_ARTIFACTS.md)",
)


def _synthetic_fundus(seed=7):
    """Deterministic fundus-like RGB array (dark background, bright disc).
    SYNTHETIC — plumbing fixture only, no clinical content."""
    import numpy as np

    rng = np.random.default_rng(seed)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:224, 0:224]
    img[(xx - 112) ** 2 + (yy - 112) ** 2 <= 100 ** 2] = [120, 60, 30]
    img[(xx - 150) ** 2 + (yy - 112) ** 2 <= 18 ** 2] = [230, 200, 140]
    noisy = img.astype(np.float32) + rng.normal(0, 4, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def model():
    from tensorflow.keras.models import load_model

    return load_model(MODEL_PATH)


@needs_weights
def test_model_loads_with_five_class_output(model):
    assert model.output_shape == (None, 5)


@needs_weights
def test_inference_returns_valid_distribution(model):
    import numpy as np
    from tensorflow.keras.applications.efficientnet import preprocess_input

    batch = np.expand_dims(preprocess_input(_synthetic_fundus().astype("float32")), axis=0)
    probs = model.predict(batch, verbose=0)[0]
    assert probs.shape == (5,)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)
    assert 0 <= int(np.argmax(probs)) <= 4


@needs_weights
def test_gradcam_heatmap_and_overlay_valid(model):
    import numpy as np
    from tensorflow.keras.applications.efficientnet import preprocess_input

    from src.models.gradcam import make_gradcam_heatmap, overlay_gradcam

    batch = np.expand_dims(preprocess_input(_synthetic_fundus().astype("float32")), axis=0)
    heatmap, pred_idx = make_gradcam_heatmap(batch, model)
    assert heatmap.ndim == 2
    assert bool(np.isfinite(heatmap).all())
    assert float(heatmap.max()) > 0  # non-empty attention signal
    assert 0 <= pred_idx <= 4
    overlay = overlay_gradcam(_synthetic_fundus(), heatmap, img_size=224)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


@needs_weights
def test_triage_deterministic_on_model_output(model):
    import numpy as np
    from tensorflow.keras.applications.efficientnet import preprocess_input

    from src.rules.triage import triage_decision

    batch = np.expand_dims(preprocess_input(_synthetic_fundus().astype("float32")), axis=0)
    probs = model.predict(batch, verbose=0)[0]
    grade, conf = int(np.argmax(probs)), float(np.max(probs))
    params = {"age": 60.0, "hba1c": 7.8, "diabetes_years": 10,
              "visual_acuity": 0.75, "spherical_refraction": 0.0}
    assert triage_decision(grade, conf, params) == triage_decision(grade, conf, params)
