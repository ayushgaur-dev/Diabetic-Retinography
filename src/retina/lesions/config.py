"""Lesion config loading (SIH26038 Phase 4D). JSON overlay on defaults —
same pattern as all other configs."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "lesion_config.json"

DEFAULTS = {
    "method": "classical_multichannel_candidates",
    "tuning_ids": ["IDRiD_01", "IDRiD_02", "IDRiD_03", "IDRiD_04",
                   "IDRiD_05", "IDRiD_06", "IDRiD_07", "IDRiD_08"],
    "common": {
        "working_width": 1072, "reference_width": 1072,
        "green_stretch_low": 1.0, "green_stretch_high": 99.0,
        "status_high_score": 0.55, "status_low_score": 0.35,
    },
    "microaneurysm": {
        "tophat_scales": [2, 3, 4, 6], "response_percentile": 98.5,
        "min_area": 3, "max_area": 220, "min_circularity": 0.45,
        "max_vessel_overlap": 0.6,
        "weights": {"contrast": 0.35, "shape": 0.35, "isolation": 0.30},
    },
    "hemorrhage": {
        "background_kernel": 51, "response_percentile": 97.0,
        "min_area": 8, "max_area": 3000, "min_circularity": 0.12,
        "max_vessel_overlap": 0.8,
        "weights": {"contrast": 0.4, "shape": 0.2, "isolation": 0.4},
    },
    "hard_exudate": {
        "brightness_percentile": 90.0, "yellow_threshold": 48.0,
        "min_area": 15, "max_area": 30000, "min_circularity": 0.2,
        "weights": {"brightness": 0.35, "color": 0.35, "shape": 0.30},
    },
    "soft_exudate": {
        "brightness_percentile": 90.0, "max_saturation": 190.0,
        "min_area": 30, "max_area": 80000, "min_circularity": 0.08,
        "weights": {"brightness": 0.3, "color": 0.3, "shape": 0.4},
    },
    "optic_disc_exclusion": {"dilation_factor": 1.6,
                             "low_confidence_dilation_factor": 1.2},
    "candidate_scoring": {},
    "postprocessing": {"fill_holes": True},
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


def load_lesion_config(path=None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)


def scaled_area(value, working_width, reference_width=1072):
    """Scale a reference-width area bound to the working resolution."""
    return value * (working_width / reference_width) ** 2
