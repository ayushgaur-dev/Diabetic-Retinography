"""Explainability config loading (SIH26038 Phase 6). JSON overlay on defaults."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "explainability_config.json"

DEFAULTS = {
    "gradcam": {"target_layer": None, "normalization": "minmax", "hot_quantile": 0.75},
    "overlap": {"mask_interpolation": "nearest", "heatmap_interpolation": "bilinear"},
    "consistency": {"high_overlap": 0.40, "low_overlap": 0.15, "inactive_max": 0.30},
    "ranking": {"weights": {"gradcam_overlap": 0.5, "candidate_score": 0.3, "area": 0.2}},
    "faithfulness": {"top_fraction": 0.20, "fill": "blurred", "blur_kernel": 31},
    "visualization": {"dpi": 110, "figsize": [13, 8]},
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


def load_explainability_config(path=None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)
