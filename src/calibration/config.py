"""Calibration config loading (SIH26038 Phase 7). JSON overlay on defaults."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "calibration_config.json"

DEFAULTS = {
    "random_seed": 42,
    "initial_temperature": 1.0,
    "optimizer": {"method": "golden_section_logT", "logT_bounds": [-3.0, 3.0],
                  "tolerance": 1e-6, "max_iterations": 200},
    "temperature_bounds": [0.05, 20.0],
    "logit_recovery": {"method": "log_probabilities", "eps": 1e-12},
    "ece_bins": [10, 15, 20],
    "ece_mode": "equal_width",
    "referable_threshold": 0.7,
    "referable_grade_min": 2,
    "bootstrap": {"n_resamples": 1000, "seed": 42},
    "model_artifact": "models/efficientnetb0_finetuned_patched.keras",
    "split_source": "Phase 5 notebook split",
}


def _deep_merge(base, override):
    merged = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
              for k, v in base.items()}
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_calibration_config(path=None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)
