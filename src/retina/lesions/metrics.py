"""Lesion metrics (SIH26038 Phase 4D).

Pixel-level: reused FOV-restricted evaluate_mask from Phase 4A (same
binary-pixel semantics; sensitivity==recall, F1==Dice documented there).
Object-level: connected components on both masks; a GT object counts as
recalled if its CENTROID falls on a predicted pixel; a predicted object
counts as precise if it overlaps any GT pixel. Generous, documented, and
fixed before any test evaluation — never tuned on held-out data.
"""

import numpy as np

from src.retina.vessels.metrics import evaluate_mask  # noqa: F401 (re-export)

from src.retina.lesions.postprocessing import components


def object_metrics(pred, gt, fov):
    """Return dict with object precision/recall/F1 + counts."""
    p = ((np.asarray(pred) > 0).astype(np.uint8)) * 255
    g = ((np.asarray(gt) > 0).astype(np.uint8)) * 255
    f = (np.asarray(fov) > 0)
    p[~f] = 0
    g[~f] = 0
    gt_objs = components(g)
    pred_objs = components(p)
    recalled = 0
    for o in gt_objs:
        cx, cy = int(round(o["centroid"][0])), int(round(o["centroid"][1]))
        if 0 <= cy < p.shape[0] and 0 <= cx < p.shape[1] and p[cy, cx] > 0:
            recalled += 1
    precise = sum(1 for o in pred_objs
                  if (g[o["mask"]]).any()) if pred_objs else 0
    n_gt, n_pred = len(gt_objs), len(pred_objs)
    rec = recalled / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    prec = precise / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"object_recall": round(rec, 4), "object_precision": round(prec, 4),
            "object_f1": round(f1, 4), "n_gt_objects": n_gt,
            "n_pred_objects": n_pred}
