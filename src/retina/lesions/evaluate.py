"""IDRiD lesion evaluation command (SIH26038 Phase 4D).

Protocol: develop on 8 TRAINING images (tuning_ids), freeze config, then
evaluate ONCE on the TEST split (27 images). Per lesion type: FOV-restricted
pixel metrics (micro over the split) + object-level metrics + positive-image
detection rate. Missing mask files count as all-negative GT (flagged).
Writes reports/lesions/{metrics.json,per_image.csv,config.json}.

Usage: python -m src.retina.lesions.evaluate [--data-root R] [--split test] [--figures]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.retina.lesions import dataset as DS  # noqa: E402
from src.retina.lesions.config import load_lesion_config  # noqa: E402
from src.retina.lesions.dataset import load_mask  # noqa: E402
from src.retina.lesions.metrics import evaluate_mask, object_metrics  # noqa: E402
from src.retina.lesions.pipeline import detect_lesions  # noqa: E402
from src.retina.lesions.visualization import save_lesion_figure  # noqa: E402
from src.retina.lesions.types import LESION_TYPES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "reports" / "lesions"


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 4D IDRiD evaluation")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test", choices=["training", "test", "all"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_lesion_config()
    entries = DS.discover(args.data_root, split=args.split)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = {lt: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "obj_rec": [], "obj_prec": [],
                "pos_images": 0, "det_images": 0, "n": 0} for lt in LESION_TYPES}
    rows, times = [], []
    for e in entries:
        rgb = np.array(Image.open(e["image"]).convert("RGB"))
        t0 = time.perf_counter()
        res = detect_lesions(rgb, config=cfg)
        times.append((time.perf_counter() - t0) * 1000.0)
        info = res.pop("_info")
        h, w = rgb.shape[:2]
        fov_full = np.full((h, w), 255, np.uint8)  # GT masks are FOV-full already
        for lt in LESION_TYPES:
            d = res[lt].to_dict()
            pm = res[lt].evidence_mask
            pm_full = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
            if e["masks"][lt] is not None:
                gt = load_mask(e["masks"][lt], (h, w))
            else:
                gt = np.zeros((h, w), np.uint8)  # flagged missing -> negative
            m = evaluate_mask(pm_full, gt, fov_full)
            o = object_metrics(pm_full, gt, fov_full)
            a = agg[lt]
            for k in ("tp", "fp", "fn", "tn"):
                a[k] += m[k]
            a["obj_rec"].append(o["object_recall"])
            a["obj_prec"].append(o["object_precision"])
            if (gt > 0).any():
                a["pos_images"] += 1
            if d["candidate_count"] > 0:
                a["det_images"] += 1
            a["n"] += 1
            rows.append({"id": e["id"], "lesion": lt, "status": d["status"],
                         "candidates": d["candidate_count"], "precision": round(m["precision"], 4),
                         "recall": round(m["sensitivity"], 4), "f1": round(m["f1"], 4),
                         "iou": round(m["iou"], 4),
                         "object_recall": o["object_recall"],
                         "object_precision": o["object_precision"],
                         "gt_missing": not e["gt_present"][lt]})
        if args.figures:
            save_lesion_figure(rgb, {lt: res[lt].to_dict() for lt in LESION_TYPES},
                               out_dir / f"{e['id']}_figure.png")

    summary = {"method": cfg.get("method"), "tuning_ids": cfg.get("tuning_ids"),
               "eval_split": f"IDRiD Segmentation {args.split}",
               "mean_ms": round(float(np.mean(times)), 1),
               "median_ms": round(float(np.median(times)), 1),
               "image_size": "4288x2848", "per_lesion": {}}
    for lt in LESION_TYPES:
        a = agg[lt]
        tp, fp, fn, tn = a["tp"], a["fp"], a["fn"], a["tn"]
        micro = {
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
            "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
            "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        }
        summary["per_lesion"][lt] = {
            "n_images": a["n"], "positive_images": a["pos_images"],
            "images_with_candidates": a["det_images"],
            "pixel_micro": {k: round(float(v), 4) for k, v in micro.items()},
            "object_recall_mean": round(float(np.mean(a["obj_rec"])), 4),
            "object_precision_mean": round(float(np.mean(a["obj_prec"])), 4),
        }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lesion", "status", "candidates",
                                          "precision", "recall", "f1", "iou",
                                          "object_recall", "object_precision",
                                          "gt_missing"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for lt in LESION_TYPES:
        p = summary["per_lesion"][lt]
        print(f"{lt:13s} pos={p['positive_images']:3d}/{p['n_images']} det={p['images_with_candidates']:3d} "
              f"pixP={p['pixel_micro']['precision']:.3f} pixR={p['pixel_micro']['recall']:.3f} "
              f"F1={p['pixel_micro']['f1']:.3f} IoU={p['pixel_micro']['iou']:.3f} "
              f"objR={p['object_recall_mean']:.3f} objP={p['object_precision_mean']:.3f}")
    print(f"mean {summary['mean_ms']} ms/image | wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
