"""Vessel config loading (SIH26038 Phase 4A). JSON overlay on defaults —
same pattern as quality/enhancement configs. No thresholds in Python."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "vessel_segmentation_config.json"

DEFAULTS = {
    "method": "multiscale_tophat_classical",
    "tuning_images": ["21_training", "22_training", "23_training"],
    "preprocessing": {"stretch_low_percentile": 1.0, "stretch_high_percentile": 99.0},
    "enhancement": {"scales": [3, 5, 9, 15]},
    "segmentation": {"method": "percentile", "percentile": 89.0},
    "postprocessing": {"min_component_size": 25, "closing_kernel": 3, "apply_closing": True},
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


def load_vessel_config(path=None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)
