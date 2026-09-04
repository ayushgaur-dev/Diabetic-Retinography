"""Error analysis categorization (SIH26038 Phase 5). Pure record-level rules."""

import numpy as np


def categorize(record, positive_grade_min=2):
    gt, pr = record.ground_truth, record.predicted_grade
    ref_t = gt >= positive_grade_min
    ref_p = pr >= positive_grade_min
    if gt == pr:
        base = "correct"
    elif abs(gt - pr) == 1:
        base = "neighboring-grade error"
    elif gt >= 3 and pr <= 1:
        base = "severe undercall"
    elif pr >= 3 and gt <= 1:
        base = "severe overcall"
    else:
        base = "one-or-more-grade error"
    suffix = ""
    if ref_t and not ref_p:
        suffix = " + referable false negative"
    elif ref_p and not ref_t:
        suffix = " + referable false positive"
    return base + suffix


def error_table(records, positive_grade_min=2):
    rows = []
    for r in records:
        rows.append({
            "image_id": r.image_id, "ground_truth": r.ground_truth,
            "prediction": r.predicted_grade,
            "raw_confidence": round(r.raw_confidence, 4),
            **{f"P{i}": round(float(p), 4) for i, p in enumerate(r.probabilities)},
            "referable_truth": int(r.ground_truth >= positive_grade_min),
            "referable_prediction": int(r.predicted_grade >= positive_grade_min),
            "error_category": categorize(r, positive_grade_min),
        })
    return rows


def worst_errors(rows, n=25):
    """Severe + referable-FN errors first, sorted by raw confidence desc —
    surfaces confidently-wrong cases."""
    def rank(row):
        cat = row["error_category"]
        sev = 0 if ("severe" in cat or "false negative" in cat) else 1
        return (sev, -row["raw_confidence"])

    return sorted(rows, key=rank)[:n]
