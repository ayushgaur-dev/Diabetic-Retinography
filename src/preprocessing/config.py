"""Enhancement config loading (SIH26038 Phase 3). Mirrors
src/quality/config.py: JSON file overlaid on built-in defaults so the
pipeline never crashes on a missing config. No thresholds in Python."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "enhancement_config.json"

DEFAULTS = {
    "version": "1.0.0",
    "clahe": {"clip_limit": 2.0, "tile_grid": [8, 8]},
    "illumination": {"median_kernel": 41, "gain_cap_lo": 0.5,
                     "gain_cap_hi": 2.0, "target": "channel_mean"},
    "denoise": {"diameter": 5, "sigma_color": 25.0, "sigma_space": 5.0},
    "color": {"trigger_spread": 25.0, "max_gain": 1.25},
    "selection": {
        "allow_unsharp": False,
        "contrast_operations": ["clahe", "denoise"],
        "illumination_operations": ["illumination_normalization", "color_normalization"],
        "exposure_dark_operations": ["illumination_normalization"],
        "exposure_bright_operations": [],
    },
    "sanity_checks": {
        "max_extra_clipped_fraction": 0.05,
        "max_mean_shift": 40.0,
        "max_channel_shift": 45.0,
        "min_mask_fraction_ratio": 0.8,
    },
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


def load_enhancement_config(path=None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)
