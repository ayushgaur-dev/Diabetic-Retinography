"""Five-class classification metrics (SIH26038 Phase 5). Pure functions of
(ground-truth, prediction) integer arrays. QWK via sklearn quadratic kappa."""

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)


def five_class_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average="weighted", zero_division=0)
    per_p, per_r, per_f, per_s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(prec_m), 4),
        "macro_recall": round(float(rec_m), 4),
        "macro_f1": round(float(f1_m), 4),
        "weighted_precision": round(float(prec_w), 4),
        "weighted_recall": round(float(rec_w), 4),
        "weighted_f1": round(float(f1_w), 4),
        "cohens_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "qwk": round(float(cohen_kappa_score(y_true, y_pred, weights="quadratic")), 4),
        "per_class": {
            str(g): {"precision": round(float(per_p[g]), 4),
                     "recall": round(float(per_r[g]), 4),
                     "f1": round(float(per_f[g]), 4),
                     "support": int(per_s[g])} for g in range(5)},
    }


def confusion_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        norm = np.nan_to_num(norm)
    return cm, np.round(norm, 4)
