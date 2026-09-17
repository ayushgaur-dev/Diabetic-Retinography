"""MATLAB integration structural checks — Phase 11. MATLAB itself is not
installed here, so these tests verify tree completeness, function/file
name contracts, fixture validity, and threshold-file sharing (the .m
suite in matlab/tests runs where MATLAB exists)."""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
MATLAB = REPO / "matlab"

EXPECTED = {
    "config/load_json_config.m": "load_json_config",
    "src/preprocessing/load_fundus.m": "load_fundus",
    "src/preprocessing/detect_fov.m": "detect_fov",
    "src/preprocessing/normalize_rgb.m": "normalize_rgb",
    "src/preprocessing/correct_illumination.m": "correct_illumination",
    "src/preprocessing/enhance_contrast_clahe.m": "enhance_contrast_clahe",
    "src/preprocessing/denoise_conservative.m": "denoise_conservative",
    "src/quality/assess_focus.m": "assess_focus",
    "src/quality/assess_illumination.m": "assess_illumination",
    "src/quality/assess_contrast.m": "assess_contrast",
    "src/quality/assess_exposure.m": "assess_exposure",
    "src/quality/assess_quality.m": "assess_quality",
    "src/visualization/plot_retina_overview.m": "plot_retina_overview",
    "src/visualization/vessel_response_demo.m": "vessel_response_demo",
    "src/evaluation/visualize_grading_report.m": "visualize_grading_report",
    "src/evaluation/visualize_calibration_report.m": "visualize_calibration_report",
    "src/workflow/screening_struct.m": "screening_struct",
    "src/workflow/run_engineering_workflow.m": "run_engineering_workflow",
    "src/integration/load_screening_json.m": "load_screening_json",
    "src/integration/validate_report.m": "validate_report",
    "src/integration/screening_summary.m": "screening_summary",
    "tests/run_all_tests.m": "run_all_tests",
}


def test_matlab_tree_complete():
    for rel in list(EXPECTED) + ["README.md", "startup.m",
                                 "outputs/.gitignore",
                                 "tests/TestLoading.m", "tests/TestPreprocessing.m",
                                 "tests/TestQuality.m", "tests/TestIntegration.m",
                                 "examples/example_quality_demo.m",
                                 "examples/example_screening_summary.m"]:
        assert (MATLAB / rel).exists(), f"missing matlab/{rel}"


def test_function_name_contract():
    for rel, func in EXPECTED.items():
        text = (MATLAB / rel).read_text()
        assert re.search(rf"^function\s.*\b{func}\b", text, re.M), \
            f"{rel} must define function {func}"


def test_balanced_control_flow():
    pairs = [("function", "end"), ("for", "end"), ("if", "end"),
             ("while", "end"), ("switch", "end"), ("try", "end")]
    for rel in EXPECTED:
        text = (MATLAB / rel).read_text()
        # Strings first (%s formats contain %), then comments, then indexing.
        code = re.sub(r"'[^'\n]*'", "''", text)
        code = "\n".join(l.split("%")[0] for l in code.splitlines())
        code = re.sub(r"[\(\{:]\s*end\b", "(IDX", code)  # indexing, not blocks
        toks = re.findall(r"\b(function|for|if|while|switch|try|end|elseif|else|catch)\b",
                          code)
        depth = 0
        for t in toks:
            if t in ("function", "for", "if", "while", "switch", "try"):
                depth += 1
            elif t == "end":
                depth -= 1
            assert depth >= 0, f"{rel}: unbalanced block"
        assert depth == 0, f"{rel}: unclosed block"


def test_shared_threshold_files_exist():
    for name in ("quality_thresholds", "calibration_config", "triage_config"):
        assert (REPO / "configs" / f"{name}.json").exists()
    qc = json.load(open(REPO / "configs" / "quality_thresholds.json"))
    # Real-fundus recalibrated (TRAIN/VAL; frozen test untouched); MATLAB
    # TestQuality.thresholdsAreShared asserts the same value.
    assert qc["focus"]["good_var"] == 120.0  # value MATLAB tests assert too
    assert qc["focus"]["borderline_var"] == 40.0


def test_matlab_fixtures_valid():
    for ex in ("example_refer", "example_ungradable"):
        p = REPO / "reports" / "reporting" / "examples" / ex / "report.json"
        assert p.exists(), f"MATLAB example input missing: {p}"
        rep = json.load(open(p))
        assert set(rep) >= {"report_version", "status", "sections"}
        assert "triage" in rep["sections"] and "summary" in rep["sections"]


def test_no_binaries_committed_under_matlab():
    bad = [p for p in MATLAB.rglob("*")
           if p.suffix.lower() in (".png", ".jpg", ".mat", ".slx")]
    assert bad == [], f"binaries under matlab/: {bad}"


def test_safety_wording_present():
    text = (REPO / "docs" / "MATLAB_WORKFLOW.md").read_text().lower()
    for phrase in ("not a diagnostic device", "human review",
                   "evidence", "uncalibrated"):
        assert phrase in text, phrase
