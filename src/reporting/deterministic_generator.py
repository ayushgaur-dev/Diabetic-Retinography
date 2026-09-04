"""Deterministic generator: validated input -> authoritative report dict.
No LLM, no network, no randomness. Same input -> byte-identical JSON."""

from src.reporting import sections as S
from src.reporting.types import REPORT_INCOMPLETE, ScreeningReport
from src.reporting.validation import validate


def generate(input_dict, cfg):
    ok, missing = validate(input_dict)
    report = ScreeningReport(report_version=cfg.get("report_version", "1.0.0"),
                             generator_version=cfg.get("generator_version", "1.0.0"))
    if not ok:
        report.status = REPORT_INCOMPLETE
        report.missing_fields = missing
        report.sections = {"image_id": input_dict.get("image_id", "unknown"),
                           "notice": "Report incomplete: required fields missing; "
                                     "nothing was fabricated."}
        return report
    tri = input_dict.get("triage", {})
    trec = {"decision": tri.get("decision"), "priority": tri.get("priority")}
    report.sections = {
        "image_id": input_dict.get("image_id", "unknown"),
        "summary": S.summary(input_dict, trec, cfg),
        "quality": S.quality_section(input_dict),
        "enhancement": S.enhancement_section(input_dict),
        "grading": S.grading_section(input_dict, cfg),
        "referable": S.referable_section(input_dict),
        "lesions": S.lesion_section(input_dict, cfg),
        "anatomy": S.anatomy_section(input_dict),
        "explainability": S.explainability_section(input_dict),
        "triage": S.triage_section(input_dict, cfg),
        "limitations": S.limitations_section(cfg),
        "provenance": S.provenance_section(cfg),
        "warnings": list(input_dict.get("warnings", [])),
    }
    return report
