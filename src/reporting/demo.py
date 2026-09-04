"""Reporting demo: python -m src.reporting.demo [--input JSON] [--example NAME] [--out DIR]

--example refer: built-in synthetic REFER fixture (no inference).
--example ungradable: built-in synthetic UNGRADABLE fixture.
--input: path to a ScreeningReportInput JSON (e.g. from a Phase 8 run).
Writes report.json/.md/.html/.pdf. No model/CV reruns.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reporting.config import load_reporting_config  # noqa: E402
from src.reporting.deterministic_generator import generate  # noqa: E402
from src.reporting.pipeline import render_all  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ungradable_example():
    return {
        "image_id": "SYNTHETIC-EXAMPLE-UNGRADABLE",
        "quality": {"status": "UNGRADABLE", "overall_score": 0.12,
                    "focus": {"status": "BAD"},
                    "reasons": ["focus: Image appears out of focus."],
                    "recapture_feedback": [
                        "Image appears out of focus. Keep the camera steady and recapture.",
                        "This image cannot be graded. Please recapture before screening."]},
        "enhancement": {"rejected": True},
        "grading": {},
        "referable": {"score": None, "threshold": 0.7},
        "lesions": {}, "vessels": {}, "optic_disc": {"status": "UNKNOWN"},
        "fovea": {"status": "UNKNOWN"}, "explainability": {},
        "triage": {"decision": "UNGRADABLE", "priority": "NORMAL",
                   "predicted_grade": -1, "calibrated_confidence": 0.0,
                   "referable_score": 0.0,
                   "reason_codes": ["IMAGE_UNGRADABLE", "RECAPTURE_REQUIRED"],
                   "evidence_summary": {"lesion_flags": []},
                   "safety_flags": {"image_quality_issue": True},
                   "explanation": "Decision: UNGRADABLE",
                   "warnings": ["Technical recapture workflow recommended."]},
        "warnings": [],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 9 reporting demo")
    ap.add_argument("--input", default=None)
    ap.add_argument("--example", default="refer", choices=["refer", "ungradable"])
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "reporting" / "examples"))
    args = ap.parse_args(argv)
    if args.input:
        data = json.load(open(args.input, encoding="utf-8"))
        name = Path(args.input).stem
    elif args.example == "ungradable":
        data, name = _ungradable_example(), "example_ungradable"
    else:
        data = json.load(open(REPO_ROOT / "tests" / "fixtures" /
                              "screening_input_example.json", encoding="utf-8"))
        name = "example_refer"
    cfg = load_reporting_config()
    report = generate(data, cfg).to_dict()
    paths = render_all(report, Path(args.out) / name, basename="report")
    print(f"status: {report['status']}")
    print(f"decision: {report['sections'].get('triage', {}).get('decision')}")
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
