"""Reporting config loading (SIH26038 Phase 9). JSON overlay on defaults."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "reporting_config.json"

DEFAULTS = {
    "report_version": "1.0.0",
    "generator_version": "1.0.0",
    "grade_labels": {"0": "No DR", "1": "Mild", "2": "Moderate",
                     "3": "Severe", "4": "Proliferative"},
    "reason_translations": {},
    "workflow_text": {},
    "lesion_limitation": "Lesion findings are algorithmic evidence candidates.",
    "limitations": [],
    "provenance": {},
    "formatting": {"probability_decimals": 4, "confidence_decimals": 4},
    "banned_phrases": [],
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


def load_reporting_config(path=None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)
