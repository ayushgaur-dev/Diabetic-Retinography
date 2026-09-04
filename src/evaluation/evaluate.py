"""Held-out grading evaluation CLI (SIH26038 Phase 5, offline, no API).

Pipeline: load labels -> reconstruct notebook split -> leakage audit ->
val inference (+threshold grid on VAL, frozen) -> test inference ->
metrics/ROC/PR/bootstrap/error/quality/enhancement-ablation -> reports.

Usage: python -m src.evaluation.evaluate --data-root <APTOS> [--split test]
       [--output-dir reports/grading] [--no-quality] [--no-bootstrap] [--seed 42]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import dataset as DS  # noqa: E402
from src.evaluation.config import load_eval_config  # noqa: E402
from src.evaluation.evaluator import (evaluate_split, full_bootstrap)  # noqa: E402
from src.evaluation.inference import run_inference  # noqa: E402
from src.evaluation.referable_metrics import (referable_labels, referable_scores,  # noqa: E402
                                              threshold_grid)
from src.evaluation.visualization import (save_confidence_hist, save_confusion,  # noqa: E402
                                          save_pr, save_roc, save_threshold_curve)

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 5 grading evaluation")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test", choices=["test", "val", "all"])
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "reports" / "grading"))
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    t_all = time.perf_counter()
    cfg = load_eval_config()
    if args.seed is not None:
        cfg["split"]["random_seed"] = args.seed
        cfg["bootstrap"]["seed"] = args.seed
    root = DS.resolve_root(args.data_root)

    t0 = time.perf_counter()
    labels = DS.load_labels(root, cfg)
    splits = DS.notebook_split(labels, cfg)
    load_s = time.perf_counter() - t0
    audit = DS.leakage_audit(splits)
    assert not audit["train_test_overlap"] and not audit["validation_test_overlap"], \
        "Split overlap detected — aborting."

    model_path = str(REPO_ROOT / cfg["model_artifact"])
    dist = {k: v["diagnosis"].value_counts().sort_index().to_dict()
            for k, v in splits.items()}

    # --- validation: inference + frozen threshold choice (test labels unseen) ---
    t0 = time.perf_counter()
    val_records = run_inference(splits["val"], model_path,
                                cfg["inference"]["image_size"],
                                cfg["inference"]["batch_size"], split="val")
    infer_val_s = time.perf_counter() - t0
    pgm = cfg["referable"]["positive_grade_min"]
    val_scores = referable_scores([r.probabilities for r in val_records], pgm)
    val_true = referable_labels([r.ground_truth for r in val_records], pgm)
    grid = threshold_grid(val_true, val_scores, cfg["decision_threshold"]["grid"])
    frozen = max(grid, key=lambda r: (r["f1"], r["sensitivity"]))
    frozen_threshold = float(frozen["threshold"])

    # --- test: single evaluation run ---
    target = {"test": ["test"], "val": ["val"], "all": ["test", "val"]}[args.split]
    all_records = val_records if args.split in ("val", "all") else []
    t0 = time.perf_counter()
    test_records = []
    if "test" in target:
        test_records = run_inference(splits["test"], model_path,
                                     cfg["inference"]["image_size"],
                                     cfg["inference"]["batch_size"], split="test")
        all_records = all_records + test_records
    infer_test_s = time.perf_counter() - t0

    eval_recs = test_records if args.split == "test" else all_records
    summary = evaluate_split(eval_recs, cfg, args.split)
    summary["threshold_analysis"] = {
        "grid_validation": grid, "frozen_threshold": frozen_threshold,
        "rule": "max F1 on validation (tie: sensitivity); applied once to test",
    }
    # frozen-threshold referable metrics on the evaluated split
    from src.evaluation.referable_metrics import referable_metrics as rm

    yb = referable_labels([r.ground_truth for r in eval_recs], pgm)
    sc = referable_scores([r.probabilities for r in eval_recs], pgm)
    summary["referable_at_frozen_threshold"] = {
        "threshold": frozen_threshold,
        **rm(yb, (sc >= frozen_threshold).astype(int)),
    }
    if not args.no_bootstrap:
        summary["bootstrap"] = full_bootstrap(eval_recs, cfg)
    if args.split == "test":
        summary["dedup_sensitivity"] = _dedup_sensitivity(eval_recs, splits, cfg)

    quality = {}
    if not args.no_quality:
        from src.evaluation.quality_analysis import analyze_quality, stratify_metrics

        qrows = analyze_quality(splits["test"] if args.split == "test"
                                else splits["val"], {r.image_id: r for r in eval_recs}, cfg)
        qmap = {r["image_id"]: r["quality_status"] for r in qrows}
        counts = {s: sum(1 for r in qrows if r["quality_status"] == s) for s in
                  ("GOOD", "BORDERLINE", "UNGRADABLE", "ERROR")}
        from src.evaluation.classification_metrics import five_class_metrics

        quality = {"counts": counts, "blocked": counts["UNGRADABLE"],
                   "by_state_five_class": stratify_metrics(
                       [{"image_id": r.image_id, "ground_truth": r.ground_truth,
                         "prediction": r.predicted_grade} for r in eval_recs],
                       qmap, lambda yt, yp: five_class_metrics(yt, yp)),
                   "note": "UNGRADABLE blocked from grading (not scored)."}
        # enhancement ablation on BORDERLINE-accepted images
        if cfg["enhancement_ablation"]["enabled"]:
            quality["enhancement_ablation"] = _enhancement_ablation(
                splits["test"] if args.split == "test" else splits["val"],
                qmap, eval_recs, cfg, model_path)
    summary["quality_analysis"] = quality
    summary["dataset"] = {"name": "APTOS 2019 (mirror mariaherrerot/aptos2019)",
                          "root": str(root), "n_labels": len(labels),
                          "class_distribution": {k: {str(g): int(c) for g, c in v.items()}
                                                 for k, v in dist.items()}}
    summary["split"] = {"procedure": "notebook 70/15/15 stratified seed "
                                     f"{cfg['split']['random_seed']}",
                        "scope": "IN-DATASET held-out (not external validation)",
                        "leakage_audit": {**audit, "duplicate_hashes":
                                          f"{len(audit['duplicate_hashes'])} duplicates",
                                          "duplicate_hash_detail": audit["duplicate_hashes"][:5]}}
    summary["model"] = {"artifact": cfg["model_artifact"], "frozen": True,
                        "note": "weights/architecture untouched"}
    summary["runtime"] = {"dataset_load_s": round(load_s, 1),
                          "inference_val_s": round(infer_val_s, 1),
                          "inference_test_s": round(infer_test_s, 1),
                          "total_s": round(time.perf_counter() - t_all, 1),
                          "batch_size": cfg["inference"]["batch_size"],
                          "device": "CPU"}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize(summary), f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        recs = [r.to_dict() for r in eval_recs]
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    cm = np.array(summary["confusion_matrix"])
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"Pred_{i}" for i in range(5)])
        for i in range(5):
            w.writerow([f"True_{i}"] + list(cm[i]))
    fig_dir = Path(args.figures_dir) if args.figures_dir else out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_confusion(cm, fig_dir / "confusion_matrix.png", "Confusion matrix (held-out test)")
    save_confusion(np.array(summary["confusion_matrix_normalized"]),
                   fig_dir / "confusion_matrix_normalized.png", "Normalized confusion matrix")
    save_roc(summary["roc_pr"]["roc"]["fpr"], summary["roc_pr"]["roc"]["tpr"],
             summary["roc_pr"]["auroc"], fig_dir / "roc.png")
    save_pr(summary["roc_pr"]["pr"]["recall"], summary["roc_pr"]["pr"]["precision"],
            summary["roc_pr"]["average_precision"], fig_dir / "pr.png")
    save_confidence_hist(summary["raw_confidence"]["histogram_10bin"],
                         fig_dir / "raw_confidence.png")
    save_threshold_curve(grid, fig_dir / "threshold_curve.png")

    r = summary["referable"]
    print(f"n={summary['n']} acc={summary['five_class']['accuracy']} "
          f"QWK={summary['five_class']['qwk']}")
    print(f"referable sens={r['sensitivity']} spec={r['specificity']} "
          f"AUROC={summary['roc_pr']['auroc']} AP={summary['roc_pr']['average_precision']}")
    print(f"frozen threshold={frozen_threshold} | quality={quality.get('counts', 'skipped')}")
    print(f"wrote {out_dir}")
    return 0


def _sanitize(obj):
    """NaN/Inf -> None (JSON-safe). Single-class subsets leave kappa undefined."""
    import math

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _dedup_sensitivity(eval_recs, splits, cfg):
    """Strict-clean test subset: drop every test image with a byte-twin in
    train/val, and dedupe within-test groups to one copy. Bounds leakage
    impact without changing the primary (exact notebook test_df) result."""
    from src.evaluation.dataset import duplicate_groups
    from src.evaluation.evaluator import evaluate_split as ev

    groups = duplicate_groups(splits)
    drop = set()
    for members in groups.values():
        tests = sorted(m.split(":")[1] for m in members if m.startswith("test:"))
        others = [m for m in members if not m.startswith("test:")]
        if others:
            drop.update(tests)  # twin outside test -> drop all test copies
        else:
            drop.update(tests[1:])  # within-test dup -> keep first copy
    filt = [r for r in eval_recs if r.image_id not in drop]
    sub = ev(filt, cfg, "test_dedup")
    return {"n_full": len(eval_recs), "n_dedup": len(filt),
            "dropped_ids": sorted(drop),
            "five_class": sub["five_class"],
            "referable": sub["referable"],
            "auroc": sub["roc_pr"]["auroc"]}


def _enhancement_ablation(df, qmap, eval_recs, cfg, model_path):
    """BORDERLINE images only: grade enhanced-vs-original inputs, compare to GT."""
    from src.evaluation.inference import get_model, preprocess_image

    import numpy as np

    from src.evaluation.quality_analysis import select_grading_image

    model = get_model(model_path)
    size = cfg["inference"]["image_size"]
    rec_by_id = {r.image_id: r for r in eval_recs}
    n_better = n_worse = n_same = n_blocked = 0
    for _, row in df.iterrows():
        if qmap.get(row["id_code"]) != "BORDERLINE":
            continue
        sel = select_grading_image({"image_path": row["image_path"],
                                    "quality_status": "BORDERLINE"}, size)
        if sel["array"] is None:
            n_blocked += 1
            continue
        from tensorflow.keras.applications.efficientnet import preprocess_input

        enh = model.predict(np.expand_dims(preprocess_input(
            sel["array"].astype("float32")), 0), verbose=0)[0]
        gt = rec_by_id[row["id_code"]].ground_truth
        base_correct = rec_by_id[row["id_code"]].predicted_grade == gt
        enh_correct = int(np.argmax(enh)) == gt
        if enh_correct and not base_correct:
            n_better += 1
        elif base_correct and not enh_correct:
            n_worse += 1
        else:
            n_same += 1
    return {"borderline_enhanced_ok": n_better + n_worse + n_same,
            "blocked": n_blocked, "enhanced_better": n_better,
            "enhanced_worse": n_worse, "unchanged": n_same,
            "note": "ablation only; base predictions grade ORIGINAL inputs"}


if __name__ == "__main__":
    raise SystemExit(main())
