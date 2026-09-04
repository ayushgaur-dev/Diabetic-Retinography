"""Evaluation orchestrator (SIH26038 Phase 5). Deterministic, offline, no API."""

import time

import numpy as np

from src.evaluation import dataset as DS
from src.evaluation.calibration_placeholder import describe_raw_confidence
from src.evaluation.classification_metrics import (confusion_counts,
                                                  five_class_metrics)
from src.evaluation.confusion import adjacent_report, severe_report
from src.evaluation.error_analysis import error_table, worst_errors
from src.evaluation.inference import run_inference
from src.evaluation.quality_analysis import (analyze_quality, select_grading_image,
                                             stratify_metrics)
from src.evaluation.referable_metrics import (referable_labels, referable_metrics,
                                              referable_scores, roc_pr,
                                              sensitivity_at_specificity,
                                              threshold_grid)
from src.evaluation.referable_metrics import referable_metrics as _rm


def _records_to_arrays(records):
    y = np.array([r.ground_truth for r in records])
    p = np.array([r.predicted_grade for r in records])
    return y, p


def evaluate_split(records, cfg, label):
    y, p = _records_to_arrays(records)
    pgm = cfg["referable"]["positive_grade_min"]
    yb = referable_labels(y, pgm)
    pb = referable_labels(p, pgm)
    scores = referable_scores([r.probabilities for r in records], pgm)
    cm, norm = confusion_counts(y, p)
    out = {
        "split": label, "n": len(records),
        "five_class": five_class_metrics(y, p),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": norm.tolist(),
        "adjacent_pairs": adjacent_report(cm),
        "severe_mistakes": severe_report(cm),
        "referable": {**referable_metrics(yb, pb),
                      "definition": f"grade >= {pgm}",
                      "score": cfg["referable"]["score_definition"],
                      "threshold": cfg["decision_threshold"]["default"],
                      "targets": {"sensitivity": 0.90, "specificity": 0.85,
                                  "note": "TARGETS, not claims"}},
        "roc_pr": roc_pr(yb, scores),
        "referable_confusion": {"tn": int(((pb == 0) & (yb == 0)).sum()),
                                "fp": int(((pb == 1) & (yb == 0)).sum()),
                                "fn": int(((pb == 0) & (yb == 1)).sum()),
                                "tp": int(((pb == 1) & (yb == 1)).sum())},
        "raw_confidence": describe_raw_confidence(records),
        "errors": worst_errors(error_table(records, pgm)),
    }
    return out


def full_bootstrap(records, cfg):
    import numpy as np
    from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    n = len(records)
    pgm = cfg["referable"]["positive_grade_min"]
    acc, f1m, qwks, sens, spec, aucs = [], [], [], [], [], []
    for _ in range(cfg["bootstrap"]["n_resamples"]):
        idx = rng.integers(0, n, n)
        boot = [records[i] for i in idx]
        y = np.array([r.ground_truth for r in boot])
        p = np.array([r.predicted_grade for r in boot])
        from sklearn.metrics import precision_recall_fscore_support

        _, _, f1, _ = precision_recall_fscore_support(
            y, p, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)
        yb = (y >= pgm).astype(int)
        pb = (p >= pgm).astype(int)
        tp = int(((pb == 1) & (yb == 1)).sum())
        tn = int(((pb == 0) & (yb == 0)).sum())
        fp = int(((pb == 1) & (yb == 0)).sum())
        fn = int(((pb == 0) & (yb == 1)).sum())
        s = np.array([sum(r.probabilities[pgm:]) for r in boot])
        try:
            auc = float(roc_auc_score(yb, s)) if len(set(yb)) == 2 else float("nan")
        except Exception:
            auc = float("nan")
        acc.append(float(accuracy_score(y, p)))
        f1m.append(float(f1))
        qwks.append(float(cohen_kappa_score(y, p, weights="quadratic")))
        sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
        aucs.append(auc)
    def ci(v):
        a = np.array([x for x in v if not (isinstance(x, float) and np.isnan(x))])
        return [round(float(np.percentile(a, 2.5)), 4),
                round(float(np.percentile(a, 97.5)), 4)] if len(a) else [None, None]
    return {"accuracy_95ci": ci(acc), "macro_f1_95ci": ci(f1m),
            "qwk_95ci": ci(qwks), "referable_sensitivity_95ci": ci(sens),
            "referable_specificity_95ci": ci(spec), "auroc_95ci": ci(aucs),
            "n_resamples": cfg["bootstrap"]["n_resamples"],
            "seed": cfg["bootstrap"]["seed"]}
