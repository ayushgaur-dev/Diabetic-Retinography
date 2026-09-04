"""Eval config loading (SIH26038 Phase 5). JSON overlay on defaults."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "grading_evaluation_config.json"

DEFAULTS = {
    "dataset": {"label_files": ["train_1.csv", "valid.csv", "test.csv"],
                "image_dirs": ["train_images", "val_images", "test_images"],
                "id_field": "id_code", "grade_field": "diagnosis",
                "image_extension": ".png"},
    "split": {"train_frac": 0.70, "temp_test_frac": 0.50,
              "random_seed": 42, "stratify": True},
    "referable": {"positive_grade_min": 2,
                  "score_definition": "P2+P3+P4"},
    "decision_threshold": {"default": 0.5, "grid": [0.3, 0.35, 0.4, 0.45, 0.5,
                                                    0.55, 0.6, 0.65, 0.7]},
    "bootstrap": {"n_resamples": 1000, "seed": 42},
    "inference": {"image_size": 224, "batch_size": 32},
    "model_artifact": "models/efficientnetb0_finetuned_patched.keras",
    "quality_analysis": {"enabled": True},
    "enhancement_ablation": {"enabled": True},
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


def load_eval_config(path=None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)
