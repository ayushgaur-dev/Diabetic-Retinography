"""Explainability subset evaluation (SIH26038 Phase 6, offline).

Runs the full pipeline + faithfulness on a small image list (default: 10
APTOS test images, 2 per grade) and writes reports/explainability/
{metrics.json,per_image.csv,config.json}. Descriptive statistics only —
NOT clinical explanation validation.

Usage: python -m src.explainability.evaluate [--data-root R] [--n-per-grade 2] [--out DIR]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import dataset as DS  # noqa: E402
from src.evaluation.config import load_eval_config  # noqa: E402
from src.evaluation.inference import get_model  # noqa: E402
from src.explainability.config import load_explainability_config  # noqa: E402
from src.explainability.pipeline import explain_image, faithfulness_deletion  # noqa: E402
from src.explainability.visualization import save_explainability_figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 6 subset evaluation")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n-per-grade", type=int, default=2)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "explainability"))
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_explainability_config()
    ecfg = load_eval_config()
    labels = DS.load_labels(DS.resolve_root(args.data_root), ecfg)
    splits = DS.notebook_split(labels, ecfg)
    test = splits["test"]
    chosen = []
    for g in range(5):
        sub = test[test["diagnosis"] == g].head(args.n_per_grade)
        chosen += sub.to_dict("records")
    model = get_model(str(REPO_ROOT / ecfg["model_artifact"]))

    rows, overlaps, cats, times, drops = [], [], {}, [], []
    for row in chosen:
        rgb = np.array(Image.open(row["image_path"]).convert("RGB"))
        t0 = time.perf_counter()
        res = explain_image(rgb, model=model, config=cfg)
        ms = (time.perf_counter() - t0) * 1000.0
        d = res.to_dict()
        f = faithfulness_deletion(rgb, model, d["predicted_grade"], d["probabilities"],
                                  config=cfg)
        cats[d["consistency"]["category"]] = cats.get(d["consistency"]["category"], 0) + 1
        times.append(ms)
        drops.append(f["probability_drop"])
        for lt, s in d["consistency"]["lesion_overlap"].items():
            overlaps.append(s["lesion_inside_gradcam_fraction"])
        rows.append({"image_id": row["id_code"], "grade_gt": row["diagnosis"],
                     "grade_pred": d["predicted_grade"],
                     "raw_conf": d["raw_confidence"],
                     "consistency": d["consistency"]["category"],
                     "regions": len(d["evidence_regions"]),
                     "prob_drop": f["probability_drop"],
                     "masked_pred": f["masked_prediction"], "ms": round(ms, 0)})
        if args.figures:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            d["_maps"] = {"gradcam": res.maps["gradcam"],
                          "vessels": (res.maps["vessel"]
                                      if res.maps["vessel"] is not None
                                      else res.maps["blank"]),
                          "blank": res.maps["blank"]}
            save_explainability_figure(rgb, d, out / "figures" / f"{row['id_code']}.png")

    summary = {
        "n_images": len(rows),
        "grade_distribution": {str(g): sum(1 for r in rows if r["grade_gt"] == g)
                               for g in range(5)},
        "consistency_distribution": cats,
        "mean_regions": round(float(np.mean([r["regions"] for r in rows])), 1),
        "mean_overlap": round(float(np.nanmean(
            [o for o in overlaps if o is not None])), 4),
        "mean_prob_drop": round(float(np.mean(drops)), 4),
        "median_prob_drop": round(float(np.median(drops)), 4),
        "mean_ms": round(float(np.mean(times)), 0),
        "note": "Descriptive explainability statistics on a 10-image subset; "
                "not clinical validation.",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(out / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
