"""Triage config loading (SIH26038 Phase 8). JSON overlay on defaults."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "triage_config.json"

DEFAULTS = {
    "decisions": ["UNGRADABLE", "ROUTINE", "REFER", "URGENT_REVIEW", "TECHNICAL_REVIEW"],
    "priorities": {"UNGRADABLE": "NORMAL", "ROUTINE": "LOW", "REFER": "HIGH",
                   "URGENT_REVIEW": "CRITICAL", "TECHNICAL_REVIEW": "HIGH"},
    "referable_grade_min": 2,
    "referable_threshold": 0.7,
    "severe_grade_threshold": 3,
    "low_confidence_threshold": 0.60,
    "rule_priority": ["INVALID_INPUT", "UNGRADABLE", "ENHANCEMENT_FAILED",
                      "PROCESSING_FAILURE", "HIGH_SEVERITY", "REFERABLE",
                      "EVIDENCE_CONFLICT", "LOW_CONFIDENCE", "ROUTINE"],
    "evidence": {
        "strong_min_candidates": {"microaneurysm": 5, "hemorrhage": 3,
                                  "hard_exudate": 3, "soft_exudate": 2},
        "conflict_action": "TECHNICAL_REVIEW",
        "escalate_patterns_to_refer": [],
    },
    "optional_evidence": ["vessels", "optic_disc", "fovea", "lesions", "gradcam"],
    "temperature": 0.9542,
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


def load_triage_config(path=None):
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user = {}
    return _deep_merge(DEFAULTS, user)
