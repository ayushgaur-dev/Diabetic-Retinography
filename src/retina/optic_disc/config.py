"""Optic-disc config loading (SIH26038 Phase 4B). JSON overlay on defaults —
same pattern as quality/enhancement/vessel configs."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "optic_disc_config.json"

DEFAULTS = {
    "method": "bright_candidate_plus_vessel_convergence",
    "tuning_ids": ["drishtiGS_002", "drishtiGS_004", "drishtiGS_008",
                   "drishtiGS_010", "drishtiGS_012"],
    "working_resolution": {"max_width": 800},
    "candidates": {
        "red_percentile": 97.0, "closing_kernel_frac": 0.02,
        "min_area_frac_fov": 0.002,
        "max_area_frac_fov": 0.05, "min_circularity": 0.25,
        "max_aspect_ratio": 2.2, "contrast_ring_width_frac": 0.5,
    },
    "convergence": {"ring_inner_factor": 1.0, "ring_outer_factor": 2.0},
    "scoring": {"weights": {"brightness": 0.30, "contrast": 0.25,
                            "geometry": 0.20, "convergence": 0.25}},
    "decision": {"detect_threshold": 0.45, "low_confidence_threshold": 0.30},
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


def load_disc_config(path=None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)
