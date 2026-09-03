"""FOV-restricted segmentation metrics (SIH26038 Phase 4A).

All counts use ONLY pixels inside the FOV mask. Sensitivity and recall are
the same quantity in binary pixel classification — both keys are reported
with that documented, not as two different metrics. F1 and Dice coincide
for binary segmentation; both reported for readability.
"""

import numpy as np


def _counts(pred, gt, fov):
    f = np.asarray(fov) > 0
    p = (np.asarray(pred) > 0) & f
    g = (np.asarray(gt) > 0) & f
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g & f).sum())  # background outside FOV is never a TN
    return tp, fp, fn, tn


def evaluate_mask(pred, gt, fov):
    """Return per-image metric dict (FOV-restricted)."""
    tp, fp, fn, tn = _counts(pred, gt, fov)
    total = tp + fp + fn + tn
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    acc = (tp + tn) / total if total else 0.0
    return {
        "sensitivity": sens, "recall": sens,  # identical by definition (binary pixels)
        "specificity": spec, "precision": prec,
        "f1": f1, "dice": f1,  # identical by definition (binary segmentation)
        "iou": iou, "accuracy": acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_fov": total,
    }


def aggregate(per_image):
    """Dataset metrics: MICRO (pooled pixel counts — primary) + MACRO
    (mean±std across images). Returns dict with both, clearly labelled."""
    keys = ("sensitivity", "specificity", "precision", "f1", "dice", "iou", "accuracy")
    tp = sum(m["tp"] for m in per_image)
    fp = sum(m["fp"] for m in per_image)
    fn = sum(m["fn"] for m in per_image)
    tn = sum(m["tn"] for m in per_image)
    # Closed forms from pooled counts (exact, dependency-free):
    micro = {
        "sensitivity": tp / (tp + fn),
        "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "dice": 2 * tp / (2 * tp + fp + fn),
        "iou": tp / (tp + fp + fn),
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
    }
    macro = {}
    for k in keys:
        vals = [m[k] for m in per_image]
        macro[f"{k}_mean"] = float(np.mean(vals))
        macro[f"{k}_std"] = float(np.std(vals))
    return {"micro": {k: round(float(v), 4) for k, v in micro.items()},
            "macro": {k: round(float(v), 4) for k, v in macro.items()},
            "n_images": len(per_image)}
