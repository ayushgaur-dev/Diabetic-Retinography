"""Simulink workflow structural checks — Phase 12. MATLAB/Simulink is not
installed here, so these verify tree completeness, function contracts,
scenario definitions, and documentation discipline (the .m suite in
matlab/tests runs where MATLAB exists). No execution is pretended."""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIM = Path(__file__).resolve().parents[1] / "matlab" / "src" / "simulink"

EXPECTED = {
    "config_simulation.m": "config_simulation",
    "validate_simulation_config.m": "validate_simulation_config",
    "create_scenario.m": "create_scenario",
    "arrival_rate_per_s.m": "arrival_rate_per_s",
    "simulate_workflow.m": "simulate_workflow",
    "collect_metrics.m": "collect_metrics",
    "run_scenario.m": "run_scenario",
    "run_all_scenarios.m": "run_all_scenarios",
    "plot_results.m": "plot_results",
    "plot_scalability.m": "plot_scalability",
    "build_simulink_model.m": "build_simulink_model",
}


def _code(rel):
    text = (SIM / rel).read_text()
    # Order matters: strip string literals FIRST (%s formats contain %),
    # then comments, then index-style end.
    code = re.sub(r"'[^'\n]*'", "''", text)
    code = "\n".join(l.split("%")[0] for l in code.splitlines())
    code = re.sub(r"[\(\{:]\s*end\b", "(IDX", code)
    return code


def test_simulink_tree_complete():
    for rel in list(EXPECTED) + ["../../../tests/TestSimulation.m"]:
        assert (SIM / rel).exists() if not rel.startswith("..") else \
            (Path(__file__).resolve().parents[1] / "matlab" / "tests" /
             "TestSimulation.m").exists(), rel


def test_function_contracts():
    for rel, func in EXPECTED.items():
        assert re.search(rf"^function\s.*\b{func}\b", _code(rel), re.M), rel


def test_balanced_blocks():
    for rel in EXPECTED:
        toks = re.findall(
            r"\b(function|for|if|while|switch|try|end|elseif|else|catch)\b",
            _code(rel))
        depth = 0
        for t in toks:
            if t in ("function", "for", "if", "while", "switch", "try"):
                depth += 1
            elif t == "end":
                depth -= 1
            assert depth >= 0, rel
        assert depth == 0, rel


def test_scenario_volumes_defined():
    text = (SIM / "create_scenario.m").read_text()
    for name, vol in (("A", 10000), ("B", 50000), ("C", 100000), ("D", 150000)):
        assert re.search(rf"'{name}'\s*,\s*{vol}", text), f"scenario {name}={vol}"


def test_seed_exposed_and_default():
    text = (SIM / "config_simulation.m").read_text()
    assert re.search(r"seed\s*=\s*26038", text)


def test_no_toolbox_only_calls():
    banned = ["exprnd", "poissrnd", "normrnd", "sim(", "simevents"]
    for rel in EXPECTED:
        code = _code(rel).lower()
        for b in banned:
            assert b not in code, f"{rel} uses toolbox-only {b}"


def test_docs_discipline():
    for doc in ("SIMULINK_WORKFLOW_AUDIT.md", "SIMULINK_WORKFLOW.md"):
        text = (Path(__file__).resolve().parents[1] / "docs" / doc).read_text()
        low = text.lower()
        assert "engineering" in low and "simulation" in low
        assert "not a clinical trial" in low or "not clinical" in low
    audit = (Path(__file__).resolve().parents[1] / "docs" /
             "SIMULINK_WORKFLOW_AUDIT.md").read_text()
    assert "100" in audit and "SimEvents" in audit


def test_no_claim_language_in_simulink_sources():
    bad = ["supports 100,000 patients/year", "clinically validated",
           "clinical trial", "proven capacity"]
    for rel in EXPECTED:
        text = (SIM / rel).read_text().lower()
        for b in bad:
            assert b not in text, f"{rel}: claim language '{b}'"


def test_no_binaries_under_simulink():
    bad = [p for p in SIM.rglob("*")
           if p.suffix.lower() in (".slx", ".mat", ".png", ".mdl")]
    assert bad == [], f"binaries under simulink/: {bad}"
