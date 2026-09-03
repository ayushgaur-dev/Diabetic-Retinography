"""Fovea config loading (SIH26038 Phase 4C). JSON overlay on defaults —
same pattern as all other configs."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "fovea_config.json"

DEFAULTS = {
    "method": "disc_relative_grid_plus_appearance_and_vessel_sparsity",
    "tuning_ids": ["IDRiD_001", "IDRiD_002", "IDRiD_003", "IDRiD_004",
                   "IDRiD_005", "IDRiD_006", "IDRiD_007", "IDRiD_008"],
    "working_resolution": {"max_width": 800},
    "geometry": {
        "disc_fovea_distance_dd": 2.5, "vertical_offset_dd": 0.2,
        "search_radius_dd": 1.5, "grid_step_dd": 0.25, "geometry_sigma_dd": 1.0,
        "region_radius_dd": 0.5,
    },
    "laterality": {"min_margin_frac_fov": 0.03},
    "appearance": {
        "patch_radius_dd": 0.35, "ring_factor": 2.0,
        "darkness_range": 40.0, "contrast_range": 25.0,
    },
    "vessel_sparsity": {"inner_radius_dd": 0.4, "outer_radius_dd": 1.2, "ratio_cap": 4.0},
    "scoring": {"weights": {"geometry": 0.35, "appearance": 0.35,
                            "vessel": 0.20, "fov": 0.10}},
    "decision": {
        "detect_threshold": 0.45, "low_confidence_threshold": 0.30,
        "ambiguity_margin": 0.10, "ambiguity_penalty": 0.30,
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


def load_fovea_config(path=None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)
