"""Quality-stratified analysis (SIH26038 Phase 5).

Phase 2 gate is NEVER altered: UNGRADABLE images are NOT sent to the model
here either — they are counted as blocked. BORDERLINE images go through
the Phase 3 pipeline (enhance -> reassess -> accepted ? enhanced :
original pipeline continues UNCHANGED for grading comparability... no:
per spec, grading runs on accepted-enhanced, recapture otherwise).

Concretely per image: assess -> GOOD: grade original | BORDERLINE: enhance;
accepted -> grade enhanced (flagged) | discarded: blocked (recapture) |
UNGRADABLE: blocked. The model itself is untouched; only the input image
varies, and every choice is recorded per image.
"""

import numpy as np
from PIL import Image

from src.preprocessing.enhancement_pipeline import enhance_image
from src.quality.quality_pipeline import assess_image


def _rgb_array(path, size=224):
    return np.array(Image.open(path).convert("RGB").resize((size, size)))


def analyze_quality(df, records_by_id, cfg):
    """Return (per-image quality rows, summary). No model calls here —
    grading of each image happens in evaluator.py via image_selector()."""
    rows = []
    for _, row in df.iterrows():
        arr = _rgb_array(row["image_path"])
        try:
            qres, _ = assess_image(arr)
            status = qres.status
        except Exception as e:
            rows.append({"image_id": row["id_code"], "quality_status": "ERROR",
                         "eligible": False, "used_image": None, "note": str(e)[:120]})
            continue
        rows.append({"image_id": row["id_code"], "quality_status": status,
                     "eligible": status in ("GOOD", "BORDERLINE"),
                     "used_image": None, "note": ""})
    return rows


def select_grading_image(row, image_size=224):
    """Decide the grading input for one quality row. Returns dict with
    used_image ('original'|'enhanced'|None), array (or None if blocked),
    and enhancement info. Model preprocessing happens downstream, once."""
    from src.preprocessing.config import load_enhancement_config

    arr = _rgb_array(row["image_path"], image_size)
    if row["quality_status"] == "GOOD":
        return {"used_image": "original", "array": arr, "enhancement": None}
    if row["quality_status"] == "BORDERLINE":
        res, _ = enhance_image(arr, enh_config=load_enhancement_config())
        if res.enhancement_successful:
            return {"used_image": "enhanced", "array": res.enhanced_image,
                    "enhancement": res.to_dict()}
        return {"used_image": None, "array": None,
                "enhancement": res.to_dict(), "blocked": "enhancement_discarded"}
    return {"used_image": None, "array": None, "enhancement": None,
            "blocked": "ungradable"}


def stratify_metrics(per_image_rows, quality_by_id, metric_fn):
    """metric_fn(y_true, y_pred) on eligible subsets per quality state."""
    out = {}
    for state in ("GOOD", "BORDERLINE"):
        sub = [r for r in per_image_rows if quality_by_id.get(r["image_id"]) == state]
        if sub:
            out[state] = {"n": len(sub), **metric_fn(
                [r["ground_truth"] for r in sub], [r["prediction"] for r in sub])}
        else:
            out[state] = {"n": 0}
    return out
