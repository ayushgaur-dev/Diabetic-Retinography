"""Reporting tests — SIH26038 Phase 9. Fixture-driven, no model/CV/API."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.config import DEFAULTS, load_reporting_config
from src.reporting.deterministic_generator import generate
from src.reporting.html_renderer import render_html
from src.reporting.input_adapter import adapt
from src.reporting.json_renderer import render_json
from src.reporting.markdown_renderer import render_markdown
from src.reporting.pdf_renderer import render_pdf
from src.reporting.pipeline import generate_report, render_all
from src.reporting.validation import validate

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "screening_input_example.json"


@pytest.fixture(scope="module")
def cfg():
    return load_reporting_config()


@pytest.fixture(scope="module")
def data():
    return json.load(open(FIXTURE, encoding="utf-8"))


@pytest.fixture(scope="module")
def report(data, cfg):
    return generate(copy.deepcopy(data), cfg).to_dict()


def test_valid_input(data, cfg):
    ok, missing = validate(data)
    assert ok and missing == []


def test_missing_required_grade():
    bad = {"quality": {"status": "GOOD"},
           "triage": {"decision": "REFER"},
           "grading": {"predicted_grade": 9, "raw_probabilities": [0.2] * 5,
                       "calibrated_confidence": 0.5}}
    ok, missing = validate(bad)
    assert not ok and "grading.predicted_grade" in missing


def test_malformed_probabilities():
    bad = {"quality": {"status": "GOOD"},
           "triage": {"decision": "REFER"},
           "grading": {"predicted_grade": 2, "raw_probabilities": [0.5, 0.5],
                       "calibrated_confidence": 0.5}}
    ok, missing = validate(bad)
    assert not ok and any("raw_probabilities" in m for m in missing)


def test_probabilities_must_sum():
    bad = {"quality": {"status": "GOOD"},
           "triage": {"decision": "REFER"},
           "grading": {"predicted_grade": 2,
                       "raw_probabilities": [0.5, 0.5, 0.5, 0.5, 0.5],
                       "calibrated_confidence": 0.5}}
    ok, missing = validate(bad)
    assert not ok and any("sum" in m for m in missing)


def test_optional_missing_fovea(data, cfg):
    d = copy.deepcopy(data)
    d["fovea"] = {}
    r = generate(d, cfg).to_dict()
    assert r["status"] == "COMPLETE"
    assert "unavailable" in json.dumps(r["sections"]["anatomy"]).lower()


def test_optional_missing_vessels(data, cfg):
    d = copy.deepcopy(data)
    d["vessels"] = {}
    assert generate(d, cfg).to_dict()["status"] == "COMPLETE"


def test_optional_missing_lesions(data, cfg):
    d = copy.deepcopy(data)
    d["lesions"] = {}
    r = generate(d, cfg).to_dict()
    assert r["status"] == "COMPLETE"
    assert len(r["sections"]["lesions"]["lesions"]) == 4


def test_ungradable_report_has_no_grade(cfg):
    d = {"image_id": "u", "quality": {"status": "UNGRADABLE"},
         "triage": {"decision": "UNGRADABLE", "priority": "NORMAL",
                    "reason_codes": ["IMAGE_UNGRADABLE"], "evidence_summary": {},
                    "safety_flags": {}, "predicted_grade": -1,
                    "calibrated_confidence": 0.0, "warnings": []},
         "grading": {}, "referable": {"score": None, "threshold": 0.7}}
    r = generate(d, cfg).to_dict()
    assert r["status"] == "COMPLETE"
    assert r["sections"]["grading"]["assessed"] is False
    assert "not assessed" in json.dumps(r["sections"]["summary"]).lower()


@pytest.mark.parametrize("decision", ["REFER", "URGENT_REVIEW", "TECHNICAL_REVIEW",
                                      "ROUTINE", "UNGRADABLE"])
def test_all_decisions_render(cfg, data, decision):
    d = copy.deepcopy(data)
    d["triage"]["decision"] = decision
    r = generate(d, cfg).to_dict()
    assert r["sections"]["triage"]["decision"] == decision
    md = render_markdown(r)
    assert decision in md


def test_reason_code_translation(data, cfg, report):
    tr = report["sections"]["triage"]
    assert tr["reason_translations"]["REFERABLE_DR_PREDICTION"] == \
        cfg["reason_translations"]["REFERABLE_DR_PREDICTION"]
    assert set(tr["reason_translations"]) == set(tr["reason_codes"])


def test_safety_flags_surfaced(report):
    assert "safety_flags" in report["sections"]["triage"]
    assert isinstance(report["sections"]["triage"]["safety_flags"], dict)


def test_calibrated_confidence_semantics(report):
    blob = json.dumps(report)
    assert "not clinical certainty" in blob
    assert report["sections"]["grading"]["confidence_sentence"] is not None


def test_probabilities_preserved(data, report):
    g = data["grading"]
    rg = report["sections"]["grading"]
    assert rg["raw_probabilities"] == [round(v, 4) for v in g["raw_probabilities"]]
    assert rg["calibrated_probabilities"] == [round(v, 4) for v in g["calibrated_probabilities"]]


def test_lesion_evidence_wording(report):
    blob = json.dumps(report["sections"]["lesions"])
    assert "evidence" in blob and "candidate" in blob
    assert "confirmed hemorrhage" not in blob.lower()
    assert "confirmed" not in blob.lower().replace("not clinically confirmed", "")


def test_no_diagnostic_language(report, cfg):
    blob = (render_markdown(report) + render_html(report)).lower()
    for phrase in cfg["banned_phrases"]:
        if phrase == "clinical certainty":
            # Required safety negation ("... is not clinical certainty.")
            # must exist; positive claims ("X% clinically certain") must not.
            assert "not clinical certainty" in blob
            assert "clinically certain" not in blob.replace(
                "not clinically certain", "")
            continue
        assert phrase not in blob, f"banned phrase present: {phrase}"


def test_deterministic_output(data, cfg):
    a = render_json(generate(copy.deepcopy(data), cfg).to_dict())
    b = render_json(generate(copy.deepcopy(data), cfg).to_dict())
    assert a == b


def test_json_schema(report, cfg):
    assert set(report) >= {"report_version", "generator_version", "status",
                           "missing_fields", "sections"}
    s = report["sections"]
    assert set(s) >= {"image_id", "summary", "quality", "enhancement", "grading",
                      "referable", "lesions", "anatomy", "explainability",
                      "triage", "limitations", "provenance", "warnings"}
    assert report["report_version"] == cfg["report_version"]


def test_markdown_rendering(report):
    md = render_markdown(report)
    for h in ("# AI-Assisted", "## Screening Summary", "## Image Quality",
              "## DR Grading", "## Referable", "## Lesion Evidence",
              "## Anatomical", "## Explainability", "## Workflow",
              "## Limitations", "## Provenance"):
        assert h in md


def test_html_rendering(report):
    h = render_html(report)
    assert "<!DOCTYPE html>" in h and "</html>" in h
    assert "cdn" not in h.lower() and "http://" not in h and "https://" not in h
    assert "REFER" in h


def test_pdf_rendering(report, tmp_path):
    p = render_pdf(report, str(tmp_path / "r.pdf"))
    assert Path(p).exists() and Path(p).stat().st_size > 3000
    with open(p, "rb") as f:
        assert f.read(5) == b"%PDF-"


def test_provenance(report):
    pv = report["sections"]["provenance"]
    assert pv["temperature"] == 0.9542
    assert "deterministic" in pv["generator"].lower()


def test_report_source_consistency(data, report):
    t = data["triage"]
    tr = report["sections"]["triage"]
    assert report["sections"]["grading"]["predicted_grade"] == t["predicted_grade"] == 2
    assert report["sections"]["grading"]["calibrated_confidence"] == \
        t["calibrated_confidence"] == 0.81
    assert tr["decision"] == t["decision"] == "REFER"
    assert tr["reason_codes"] == t["reason_codes"]


def test_no_fabricated_patient_metadata(data, cfg):
    r = generate(copy.deepcopy(data), cfg).to_dict()
    blob = json.dumps(r).lower()
    for field in ("patient_id", "patient_name", "age", "sex", "diabetes_years",
                  "hba1c", "visual_acuity", "symptom", "history", "medication"):
        assert f'"{field}"' not in blob


def test_incomplete_report(data, cfg):
    d = copy.deepcopy(data)
    d["grading"] = {}
    r = generate(d, cfg).to_dict()
    assert r["status"] == "REPORT_INCOMPLETE"
    assert r["missing_fields"]
    assert "fabricated" in json.dumps(r["sections"]).lower()


def test_adapter_builds_input():
    t = adapt(image_id="i1", quality={"status": "GOOD"},
              grade=1, raw_probs=[0.1, 0.6, 0.2, 0.05, 0.05],
              calibrated_probs=[0.1, 0.62, 0.18, 0.05, 0.05],
              calibrated_confidence=0.62, referable_score=0.28,
              triage={"decision": "ROUTINE"})
    d = t.to_dict()
    assert d["grading"]["predicted_grade"] == 1
    assert d["grading"]["raw_confidence"] == pytest.approx(0.6)


def test_render_all_writes_formats(report, tmp_path):
    paths = render_all(report, tmp_path, basename="r")
    assert set(paths) == {"json", "md", "html", "pdf"}
    for p in paths.values():
        assert Path(p).exists() and Path(p).stat().st_size > 0
