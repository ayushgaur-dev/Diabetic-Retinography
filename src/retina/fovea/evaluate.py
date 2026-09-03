"""IDRiD fovea evaluation command (SIH26038 Phase 4C).

Protocol: tune on 8 TRAINING images (tuning_ids in config), freeze, then
evaluate ONCE on the TEST split (103 images, independent of tuning).
GT laterality is derived from GT disc vs fovea x-order (disc nasal).
Writes reports/fovea/{metrics.json,per_image.csv,config.json}.

Usage: python -m src.retina.fovea.evaluate [--data-root R] [--split test] [--figures]
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

from src.retina.fovea import dataset as DS  # noqa: E402
from src.retina.fovea.config import load_fovea_config  # noqa: E402
from src.retina.fovea.metrics import (center_error, normalized_error_fov,  # noqa: E402
                                      summarize)
from src.retina.fovea.pipeline import localize_fovea  # noqa: E402
from src.retina.fovea.visualization import save_fovea_figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "reports" / "fovea"


def _gt_laterality(fovea_xy, disc_xy):
    if disc_xy is None:
        return "unknown"
    return "right" if fovea_xy[0] > disc_xy[0] else "left"  # disc nasal


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 4C IDRiD evaluation")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test", choices=["training", "test", "all"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_fovea_config()
    entries = DS.discover(args.data_root, split=args.split)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for e in entries:
        rgb = np.array(Image.open(e["image"]).convert("RGB"))
        # End-to-end: NO supplied disc — the pipeline runs Phase 4B itself.
        # GT disc centers serve only for laterality_gt bookkeeping.
        t0 = time.perf_counter()
        res = localize_fovea(rgb, config=cfg)
        ms = (time.perf_counter() - t0) * 1000.0
        d = res.to_dict()
        fov_w = float(rgb.shape[1])
        if d["detected"]:
            ce = center_error(d["center_x_y"], e["fovea"])
            ne = normalized_error_fov(d["center_x_y"], e["fovea"], fov_w)
        else:
            ce, ne = None, None
        records.append({"id": e["id"], "detected": d["detected"], "status": d["status"],
                        "px_err": ce, "norm_err": ne,
                        "disc_used": d["disc_used"], "laterality": d["laterality"],
                        "laterality_gt": _gt_laterality(e["fovea"], e["disc"]),
                        "confidence": d["confidence"], "candidates": d["candidate_count"],
                        "ms": round(ms, 1)})
        if args.figures:
            save_fovea_figure(rgb, d, out_dir / f"{e['id']}_figure.png",
                              disc_xy=e["disc"])

    summary = summarize(records)
    summary["method"] = cfg.get("method")
    summary["tuning_ids"] = cfg.get("tuning_ids")
    summary["eval_split"] = f"IDRiD Localization {args.split} (GT fovea centers)"
    summary["normalization"] = ("center error / image (FOV) width; GT disc "
                                "diameters unavailable in IDRiD Localization")
    summary["mean_ms"] = round(float(np.mean([r["ms"] for r in records])), 1)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "detected", "status", "px_err", "norm_err",
                                          "disc_used", "laterality", "laterality_gt",
                                          "confidence", "candidates", "ms"])
        w.writeheader()
        for r in records:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                        for k in w.fieldnames})
    print(f"images: {summary['n_images']} DETECTED: {summary['n_detected']} "
          f"LOW: {summary['n_low_confidence']} NOT: {summary['n_not_detected']}")
    print(f"median norm err: {summary['median_normalized_error']} | "
          f"det@0.025/0.05/0.1: {summary['detection_at_0.025']}/"
          f"{summary['detection_at_0.05']}/{summary['detection_at_0.1']}")
    print(f"laterality L/R: {summary['laterality_left']} | {summary['laterality_right']}")
    print(f"mean {summary['mean_ms']} ms/image | wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
