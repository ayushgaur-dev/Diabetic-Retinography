"""Config loading for the quality subsystem (SIH26038 Phase 2).

Thresholds live in configs/quality_thresholds.json — never in Python.
load_config() overlays a user-supplied file on built-in defaults so the
pipeline always runs even if the config file is missing.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "quality_thresholds.json"

DEFAULTS = {
    "focus": {"log_lo": 1.6, "log_hi": 3.0, "good_var": 300.0, "borderline_var": 60.0},
    "illumination": {
        "median_good_lo": 50.0, "median_good_hi": 150.0,
        "median_borderline_lo": 30.0, "median_borderline_hi": 180.0,
        "nonuniformity_good": 18.0, "nonuniformity_borderline": 30.0,
    },
    "contrast": {
        "std_good": 15.0, "std_borderline": 8.0,
        "std_excessive": 70.0, "spread_borderline": 15.0,
    },
    "exposure": {
        "black_level": 5, "white_level": 250,
        "black_borderline": 0.08, "black_bad": 0.20,
        "white_borderline": 0.02, "white_bad": 0.08,
    },
    "field_of_view": {
        "red_factor": 0.35, "red_floor": 10.0,
        "mask_fraction_good": 0.40, "mask_fraction_borderline": 0.20,
        "min_plausible_fraction": 0.05,
    },
    "retinal_coverage": {"completeness_good": 0.75, "completeness_borderline": 0.55},
    "aggregation": {"weights": {
        "focus": 1.0, "illumination": 1.0, "contrast": 1.0,
        "exposure": 1.0, "field_of_view": 1.5, "retinal_coverage": 1.0,
    }},
}


def _deep_merge(base, override):
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path=None):
    """Load quality thresholds. `path=None` -> repo default JSON file.
    Missing file -> built-in DEFAULTS (pipeline never crashes on config)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)
