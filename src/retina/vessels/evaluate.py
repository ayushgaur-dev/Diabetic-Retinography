"""DRIVE evaluation command (SIH26038 Phase 4A).

Protocol (also see docs/VESSEL_SEGMENTATION.md):
  1. Parameters were tuned on DRIVE training images 21-23 ONLY, then frozen.
  2. This command evaluates the FROZEN config on all 20 training images
     against 1st_manual (FOV-restricted). Official test annotations are not
     shipped in this mirror, so the test split gets a runtime/mask sanity
     run only — NO metrics are reported where no ground truth exists.
  3. Writes reports/vessel_segmentation/{metrics.json,per_image_metrics.csv,config.json}.

Usage:
  python -m src.retina.vessels.evaluate --data-root <DRIVE root> [--out DIR]
  (DRIVE_DATA_ROOT env var is used when --data-root is omitted.)
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

from src.retina.vessels import dataset as ds  # noqa: E402
from src.retina.vessels.config import load_vessel_config  # noqa: E402
from src.retina.vessels.metrics import aggregate, evaluate_mask  # noqa: E402
from src.retina.vessels.pipeline import load_fov_mask, segment_vessels  # noqa: E402
from src.retina.vessels.visualization import save_vessel_figure  # noqa: E402
from src.retina.vessels.preprocessing import extract_green  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "reports" / "vessel_segmentation"


def _read_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def _read_gray(path):
    return np.array(Image.open(path).convert("L"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 4A DRIVE evaluation")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--figures", action="store_true",
                    help="also save per-image debug figures")
    args = ap.parse_args(argv)

    cfg = load_vessel_config()
    entries = ds.discover(args.data_root)
    train = [e for e in entries if e["split"] == "training"]
    test = [e for e in entries if e["split"] == "test"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image, total_ms = [], 0.0
    for e in train:
        rgb = _read_rgb(e["image"])
        gt = _read_gray(e["manual"])
        fov = load_fov_mask(e["fov"], rgb.shape[:2])
        t0 = time.perf_counter()
        res = segment_vessels(rgb, fov, config=cfg)
        total_ms += (time.perf_counter() - t0) * 1000.0
        m = evaluate_mask(res.vessel_mask, gt, fov)
        m["id"] = e["id"]
        m["density"] = round(res.vessel_density, 4)
        m["ms"] = round(res.processing_time_ms, 1)
        per_image.append(m)
        if args.figures:
            save_vessel_figure(rgb, extract_green(rgb), res.vessel_response,
                               res.vessel_mask,
                               out_dir / f"{e['id']}_figure.png",
                               gt_mask=gt, title=e["id"])

    metrics = aggregate([{k: m[k] for k in
                          ("sensitivity", "specificity", "precision", "f1",
                           "dice", "iou", "accuracy", "tp", "fp", "fn", "tn")}
                         for m in per_image])
    metrics["method"] = cfg.get("method")
    metrics["tuning_images"] = cfg.get("tuning_images")
    metrics["eval_split"] = "DRIVE training (20 images) vs 1st_manual, FOV-restricted"
    metrics["test_split_note"] = ("official test annotations not shipped in this "
                                  "mirror — no test metrics reported")
    metrics["mean_ms_per_image"] = round(total_ms / max(len(train), 1), 1)
    metrics["image_size"] = "565x584"

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(out_dir / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "sensitivity", "specificity",
                                          "precision", "f1", "dice", "iou",
                                          "accuracy", "density", "ms"])
        w.writeheader()
        for m in per_image:
            w.writerow({k: (round(m[k], 4) if isinstance(m[k], float) else m[k])
                        for k in w.fieldnames})

    # Test split: runtime + mask sanity only (no ground truth -> no metrics).
    test_ms = []
    for e in test:
        rgb = _read_rgb(e["image"])
        fov = load_fov_mask(e["fov"], rgb.shape[:2])
        res = segment_vessels(rgb, fov, config=cfg)
        test_ms.append(res.processing_time_ms)
    print(f"train images: {len(train)} | micro F1={metrics['micro']['f1']:.4f} "
          f"acc={metrics['micro']['accuracy']:.4f} sens={metrics['micro']['sensitivity']:.4f} "
          f"spec={metrics['micro']['specificity']:.4f}")
    print(f"test images (no GT): {len(test)} sanity-checked, "
          f"mean {np.mean(test_ms):.1f} ms/image")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
