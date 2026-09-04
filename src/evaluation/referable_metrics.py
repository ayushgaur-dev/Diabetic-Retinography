"""Referable-DR binary metrics (SIH26038 Phase 5).

Referable = grade >= positive_grade_min (default 2, configurable).
Referable score = P2+P3+P4 (configurable definition string recorded).
SIH targets (sens>90%, spec>85%) are TARGETS — reported against, never claimed.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def referable_labels(grades, positive_grade_min=2):
    return (np.asarray(grades, dtype=int) >= positive_grade_min).astype(int)


def referable_scores(probabilities, positive_grade_min=2):
    """P(grade >= min): row-wise sum of predicted probabilities."""
    p = np.asarray(probabilities, dtype=float)
    return p[:, positive_grade_min:].sum(axis=1)


def referable_metrics(y_true_bin, y_pred_bin):
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    y_pred_bin = np.asarray(y_pred_bin, dtype=int)
    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / max(len(y_true_bin), 1)
    bal = (sens + spec) / 2
    prev = (tp + fn) / max(len(y_true_bin), 1)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "precision_ppv": round(prec, 4), "npv": round(npv, 4),
        "f1": round(f1, 4), "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal), 4),
        "prevalence": round(float(prev), 4),
    }


def roc_pr(y_true_bin, scores):
    """AUROC + average precision + curves (downsampled to <=500 points)."""
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    scores = np.asarray(scores, dtype=float)
    auroc = float(roc_auc_score(y_true_bin, scores))
    ap = float(average_precision_score(y_true_bin, scores))
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, thr = roc_curve(y_true_bin, scores)
    prec, rec, thr_pr = precision_recall_curve(y_true_bin, scores)
    return {
        "auroc": round(auroc, 4), "average_precision": round(ap, 4),
        "roc": {"fpr": _thin(fpr), "tpr": _thin(tpr), "thresholds": _thin(thr)},
        "pr": {"precision": _thin(prec), "recall": _thin(rec),
               "thresholds": _thin(thr_pr)},
    }


def _thin(arr, n=500):
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= n:
        return [round(float(v), 4) for v in arr]
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return [round(float(arr[i]), 4) for i in idx]


def threshold_grid(y_true_bin, scores, grid):
    rows = []
    for t in grid:
        pred = (np.asarray(scores) >= t).astype(int)
        m = referable_metrics(y_true_bin, pred)
        rows.append({"threshold": t, "sensitivity": m["sensitivity"],
                     "specificity": m["specificity"],
                     "precision": m["precision_ppv"], "npv": m["npv"],
                     "f1": m["f1"]})
    return rows


def sensitivity_at_specificity(y_true_bin, scores, min_spec):
    """Max sensitivity with specificity >= min_spec (None if unachievable)."""
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(y_true_bin, dtype=int)[order]
    n_neg = (y == 0).sum()
    best = None
    for k in range(1, len(y) + 1):
        pred = np.zeros(len(y), dtype=int)
        pred[order[:k]] = 1
        m = referable_metrics(y_true_bin, pred)
        if m["specificity"] >= min_spec:
            best = {"threshold_rank": k, "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"]}
    return best
