"""Triage evaluation (SIH26038 Phase 8, offline).

Full test split (550): cached Phase 5 predictions + fresh quality +
temperature application (no model rerun, no lesions — recorded missing).
Evidence subset (30, stratified): full lesion/vessel/disc/fovea context.
Writes reports/triage/{metrics.json,per_image.csv,config.json} (+cards).

Usage: python -m src.triage.evaluate [--reports-grading DIR] [--out DIR] [--subset N]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.calibration.temperature_scaling import apply_temperature  # noqa: E402
from src.preprocessing.enhancement_pipeline import enhance_image  # noqa: E402
from src.quality.quality_pipeline import assess_image  # noqa: E402
from src.triage.config import load_triage_config  # noqa: E402
from src.triage.decision import decide  # noqa: E402
from src.triage.types import TriageInput  # noqa: E402
from src.triage.visualization import save_triage_card  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _rgb224(path):
    return np.array(Image.open(path).convert("RGB").resize((224, 224)))


def _quality_of(image_path):
    try:
        r, _ = assess_image(_rgb224(image_path))
        return r.status, round(r.overall_score, 3)
    except Exception as e:
        return "ERROR", 0.0


def _enhancement_status(arr):
    try:
        res, _ = enhance_image(arr)
        if res.bypassed:
            return "none"
        return "success" if res.enhancement_successful else "failed"
    except Exception:
        return "failed"


def _input_from_row(row, probs, quality, enh, lesions, vessel, disc, fovea, cfg):
    cal = apply_temperature(np.array(probs), cfg["temperature"])
    lev = {}
    for lt, d in (lesions or {}).items():
        dd = d.to_dict() if hasattr(d, "to_dict") else d
        lev[lt] = {"status": dd.get("status", "UNKNOWN"),
                   "candidate_count": dd.get("candidate_count", 0),
                   "confidence": dd.get("confidence", 0.0)}
    disc_d = disc.to_dict() if hasattr(disc, "to_dict") else (disc or {})
    fov_d = fovea.to_dict() if hasattr(fovea, "to_dict") else (fovea or {})
    return TriageInput(
        quality_status=quality[0], quality_score=quality[1],
        enhancement_status=enh, predicted_grade=int(np.argmax(probs)),
        raw_probabilities=[float(v) for v in probs],
        calibrated_probabilities=[float(v) for v in cal],
        calibrated_confidence=float(cal.max()),
        referable_score=float(cal[2:].sum()),
        lesion_evidence=lev, vessel_available=vessel is not None,
        optic_disc_status=str(disc_d.get("status", "UNKNOWN")),
        fovea_status=str(fov_d.get("status", "UNKNOWN")),
        consistency="UNKNOWN", processing_errors=[])


def main(argv=None):
    ap = argparse.ArgumentParser(description="SIH26038 Phase 8 triage evaluation")
    ap.add_argument("--reports-grading", default=str(REPO_ROOT / "reports" / "grading"))
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "triage"))
    ap.add_argument("--subset", type=int, default=30)
    ap.add_argument("--aptos-root", default=None)
    args = ap.parse_args(argv)

    t_all = time.perf_counter()
    cfg = load_triage_config()
    per_image = pd.read_csv(Path(args.reports_grading) / "per_image.csv")
    img_root = Path(args.aptos_root) if args.aptos_root else None
    if img_root is None:
        from src.evaluation import dataset as DS
        from src.evaluation.config import load_eval_config

        ecfg = load_eval_config()
        labels = DS.load_labels(DS.resolve_root(None), ecfg)
        splits = DS.notebook_split(labels, ecfg)
        paths = {r["id_code"]: r["image_path"] for _, r in splits["test"].iterrows()}
    else:
        paths = {}
        for _, r in per_image.iterrows():
            cands = list(Path(img_root).rglob(r["image_id"] + ".png"))
            paths[r["image_id"]] = str(cands[0]) if cands else None

    rows, dist, reasons = [], {}, {}
    for _, r in per_image.iterrows():
        probs = np.array([r[f"probability_{i}"] for i in range(5)])
        q = _quality_of(paths[r["image_id"]])
        enh = _enhancement_status(_rgb224(paths[r["image_id"]])) \
            if q[0] == "BORDERLINE" else "none"
        t = _input_from_row(r, probs, q, enh, None, None, None, None, cfg)
        d = decide(t, cfg).to_dict()
        dist[d["decision"]] = dist.get(d["decision"], 0) + 1
        for rc in d["reason_codes"]:
            reasons[rc] = reasons.get(rc, 0) + 1
        rows.append({"image_id": r["image_id"], "gt": r["ground_truth"],
                     "grade": d["predicted_grade"], "decision": d["decision"],
                     "priority": d["priority"], "reasons": "|".join(d["reason_codes"]),
                     "quality": q[0], "evidence": "none"})

    # evidence subset: stratified sample with FULL lesion context
    rng = np.random.default_rng(42)
    sub = []
    for g, n in ((0, 12), (1, 3), (2, 8), (3, 3), (4, 4)):
        pool = per_image[per_image["ground_truth"] == g]
        sub += pool.sample(min(n, len(pool)), random_state=42).to_dict("records")
    try:
        from src.retina.lesions.pipeline import detect_lesions
        from src.retina.vessels.pipeline import segment_vessels
        from src.retina.optic_disc.pipeline import localize_optic_disc
        from src.retina.fovea.pipeline import localize_fovea
        from src.quality.field_of_view import detect_retinal_field
        from src.quality.config import load_config as load_qconfig

        full_ctx = True
    except ImportError:
        full_ctx = False
    sub_rows = []
    if full_ctx:
        for r in sub:
            rgb = np.array(Image.open(paths[r["image_id"]]).convert("RGB"))
            try:
                fov = detect_retinal_field(rgb, load_qconfig())["mask"]
                ves = segment_vessels(rgb, fov).vessel_mask
                dsc = localize_optic_disc(rgb, fov_mask=fov)
                fov2 = localize_fovea(rgb, fov_mask=fov, disc=dsc.to_dict())
                les = detect_lesions(rgb)
                les.pop("_info", None)
                vessel, disc, fovea = ves, dsc, fov2
            except Exception:
                vessel, disc, fovea, les = None, None, None, None
            probs = np.array([r[f"probability_{i}"] for i in range(5)])
            q = _quality_of(paths[r["image_id"]])
            enh = _enhancement_status(_rgb224(paths[r["image_id"]])) \
                if q[0] == "BORDERLINE" else "none"
            t = _input_from_row(r, probs, q, enh, les, vessel, disc, fovea, cfg)
            d = decide(t, cfg).to_dict()
            sub_rows.append({"image_id": r["image_id"], "gt": r["ground_truth"],
                             "grade": d["predicted_grade"], "decision": d["decision"],
                             "reasons": "|".join(d["reason_codes"]), "quality": q[0],
                             "evidence": "full"})
    # safety analysis on full run
    fn = [r for r in rows if r["gt"] >= 2 and r["decision"] == "ROUTINE"]
    severe = [r for r in rows if r["gt"] >= 3 and r["grade"] <= 1
              and r["decision"] == "ROUTINE"]
    summary = {
        "n": len(rows), "distribution": dist, "reason_frequencies": reasons,
        "subset_n": len(sub_rows),
        "subset_distribution": {d: sum(1 for r in sub_rows if r["decision"] == d)
                                for d in set(r["decision"] for r in sub_rows)} if sub_rows else {},
        "safety": {
            "referable_false_negatives": len(fn),
            "referable_fn_ids": [r["image_id"] for r in fn][:20],
            "severe_undercalls_routine": len(severe),
            "severe_ids": [r["image_id"] for r in severe][:20],
            "escalated_referable_frac": round(
                sum(1 for r in rows if r["gt"] >= 2
                    and r["decision"] in ("REFER", "URGENT_REVIEW")) / max(
                    sum(1 for r in rows if r["gt"] >= 2), 1), 4),
        },
        "runtime_s": round(time.perf_counter() - t_all, 1),
        "note": "Full run grades without lesion evidence (recorded missing); "
                "subset adds full evidence. Descriptive workflow statistics.",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows + [{**r, "evidence": "full"} for r in sub_rows])
    # example cards (one per decision observed)
    seen = set()
    cards = out / "figures"
    cards.mkdir(parents=True, exist_ok=True)
    for r in rows + sub_rows:
        if r["decision"] in seen:
            continue
        seen.add(r["decision"])
        rgb = np.array(Image.open(paths[r["image_id"]]).convert("RGB"))
        if max(rgb.shape[:2]) > 1600:
            rgb = np.array(Image.fromarray(rgb).resize((800, 600)))
        probs = per_image.set_index("image_id").loc[r["image_id"]]
        cal = __import__("src.calibration.temperature_scaling",
                         fromlist=["apply_temperature"]).apply_temperature(
            np.array([probs[f"probability_{i}"] for i in range(5)]),
            cfg["temperature"])
        save_triage_card(rgb, {"predicted_grade": r["grade"],
                               "calibrated_confidence": round(float(cal.max()), 3),
                               "decision": r["decision"],
                               "priority": {"UNGRADABLE": "NORMAL", "ROUTINE": "LOW",
                                            "REFER": "HIGH", "URGENT_REVIEW": "CRITICAL",
                                            "TECHNICAL_REVIEW": "HIGH"}[r["decision"]],
                               "referable": r["decision"] in ("REFER", "URGENT_REVIEW"),
                               "reason_codes": r["reasons"].split("|"),
                               "evidence_summary": {}, "explanation": ""},
                         cards / f"{r['decision']}_{r['image_id']}.png", r["quality"])
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("n", "distribution", "safety")}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
