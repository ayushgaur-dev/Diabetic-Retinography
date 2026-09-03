"""Drishti-GS evaluation command (SIH26038 Phase 4B).

Protocol: parameters tuned on 5 TRAINING images (see tuning_ids in config),
frozen, then evaluated ONCE on the TEST split (independent of tuning).
Writes reports/optic_disc/{metrics.json,per_image.csv,config.json}.

Usage: python -m src.retina.optic_disc.evaluate [--data-root R] [--split test] [--figures]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.retina.optic_disc import drishti as D  # noqa: E402
from src.retina.optic_disc.config import load_disc_config  # noqa: E402
from src.retina.optic_disc.metrics import (bbox_iou, center_error,  # noqa: E402
                                           normalized_error, summarize)
from src.retina.optic_disc.pipeline import localize_optic_disc  # noqa: E402
from src.retina.optic_disc.visualization import save_disc_figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "reports" / "optic_disc"


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 4B Drishti-GS evaluation")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test", choices=["training", "test", "all"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_disc_config()
    entries = D.discover(args.data_root, split=args.split)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for e in entries:
        rgb = np.array(Image.open(e["image"]).convert("RGB"))
        gt = D.gt_in_image_pixels(e["gt"], rgb.shape)
        t0 = time.perf_counter()
        res = localize_optic_disc(rgb, config=cfg)
        ms = (time.perf_counter() - t0) * 1000.0
        d = res.to_dict()
        if d["detected"]:
            ce = center_error(d["center"], gt["center"])
            ne = normalized_error(d["center"], gt["center"], gt["diameter"])
            gr = gt["diameter"] / 2.0
            gx, gy = gt["center"]
            iou = bbox_iou(d["bounding_box"], [gx - gr, gy - gr, gx + gr, gy + gr])
        else:
            ce, ne, iou = None, None, None
        records.append({"id": e["id"], "detected": d["detected"],
                        "status": d["status"], "center_error_px": ce,
                        "normalized_error": ne, "bbox_iou": iou,
                        "confidence": d["confidence"],
                        "candidates": d["candidate_count"],
                        "gt_diameter": round(gt["diameter"], 1), "ms": round(ms, 1)})
        if args.figures:
            save_disc_figure(rgb, d, out_dir / f"{e['id']}_figure.png")

    summary = summarize(records)
    summary["method"] = cfg.get("method")
    summary["tuning_ids"] = cfg.get("tuning_ids")
    summary["eval_split"] = f"Drishti-GS {args.split} (GT: diskCenter + OD softmap)"
    summary["mean_ms"] = round(float(np.mean([r["ms"] for r in records])), 1)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "detected", "status", "center_error_px",
                                          "normalized_error", "bbox_iou", "confidence",
                                          "candidates", "gt_diameter", "ms"])
        w.writeheader()
        for r in records:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                        for k in w.fieldnames})
    print(f"images: {summary['n_images']} detected: {summary['n_detected']} "
          f"({summary['detection_rate']:.0%})")
    print(f"median norm err: {summary['median_normalized_error']} | "
          f"det@0.25/0.5/1.0: {summary['detection_at_0.25']}/{summary['detection_at_0.5']}/"
          f"{summary['detection_at_1.0']} | mean IoU: {summary['mean_bbox_iou']}")
    print(f"mean {summary['mean_ms']} ms/image | wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
