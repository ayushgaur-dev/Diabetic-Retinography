"""Drishti-GS dataset interface (SIH26038 Phase 4B).

Mirror layout (Kaggle lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation):

  <root>/Training-*/Training/Images/{GLAUCOMA,NORMAL}/drishtiGS_NNN.png
  <root>/Training-*/Training/GT/drishtiGS_NNN/AvgBoundary/drishtiGS_NNN_diskCenter.txt  ("x y")
  <root>/Training-*/Training/GT/drishtiGS_NNN/SoftMap/drishtiGS_NNN_ODsegSoftmap.png
  (same shape under Test-*/Test/...)

GT used: disc center from diskCenter.txt, cross-checked against the OD
softmap centroid; disc diameter from the softmap majority mask
(agreement >= 128/255) as 2*sqrt(area/pi). Softmaps are ~4px larger than
the images (2049x1751 vs 2045x1752) — GT coords are scaled to image pixels.

Root resolution: --data-root > DRISHTI_DATA_ROOT env > data/drishti/
(empty placeholder; dataset NEVER committed).
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_root(root=None):
    if root:
        return Path(root)
    env = os.environ.get("DRISHTI_DATA_ROOT")
    if env:
        return Path(env)
    return REPO_ROOT / "data" / "drishti"


def _find_dirs(root, kind):
    """kind: 'Training' | 'Test'. Returns (images_dir, gt_dir)."""
    cands = sorted(Path(root).glob(f"{kind}-*"))
    if not cands:
        raise FileNotFoundError(f"No {kind} directory under {root}.")
    base = cands[0] / kind
    gt_dir = base / "GT"
    if not gt_dir.is_dir():  # test split in this mirror uses Test_GT/
        gt_dir = base / "Test_GT"
    return base / "Images", gt_dir


def _gt_for(gt_dir, _id):
    gdir = gt_dir / _id
    if not gdir.is_dir():
        return None
    dc = gdir / "AvgBoundary" / f"{_id}_diskCenter.txt"
    sm = gdir / "SoftMap" / f"{_id}_ODsegSoftmap.png"
    if not (dc.exists() and sm.exists()):
        return None
    cx, cy = (float(v) for v in dc.read_text().split()[:2])
    # diskCenter.txt stores MATRIX order (row, col): first value is the
    # vertical (y) coordinate, second is horizontal (x). Verified: reading
    # them as (x, y) puts GT ~200px from both the prediction and the
    # softmap centroid; swapped, all three agree within tens of px.
    cx, cy = cy, cx
    soft = np.array(Image.open(sm).convert("L")).astype(np.float32)
    maj = soft >= 128
    area = float(maj.sum())
    ys, xs = np.nonzero(maj)
    centroid = (float(xs.mean()), float(ys.mean())) if area else (cx, cy)
    diameter = 2.0 * float(np.sqrt(area / np.pi)) if area else 0.0
    return {"center": (cx, cy), "softmap_shape": soft.shape,
            "softmap_centroid": centroid, "diameter_softmap_px": diameter}


def discover(root=None, split="test"):
    """Manifest entries: id, image path, gt dict (or None), split.
    split in {'training', 'test', 'all'}."""
    base = resolve_root(root)
    splits = ("training", "test") if split == "all" else (split,)
    entries = []
    for sp in splits:
        kind = "Training" if sp == "training" else "Test"
        img_dir, gt_dir = _find_dirs(base, kind)
        for img_path in sorted(img_dir.rglob("drishtiGS_*.png")):
            _id = img_path.stem
            gt = _gt_for(gt_dir, _id)
            if gt is None:
                raise FileNotFoundError(f"Drishti GT missing for {_id} under {gt_dir}.")
            entries.append({"id": _id, "split": sp, "image": str(img_path), "gt": gt})
    if not entries:
        raise FileNotFoundError(f"No Drishti-GS {split} images under {base}.")
    return entries


def gt_in_image_pixels(gt, image_shape):
    """Scale GT (diskCenter frame ~= image frame; softmap ~4px larger) to image pixels."""
    h, w = image_shape[:2]
    sh, sw = gt["softmap_shape"]
    sx, sy = w / sw, h / sh
    cx, cy = gt["center"]
    scx, scy = gt["softmap_centroid"]
    return {
        "center": (cx * sx, cy * sy),
        "softmap_centroid": (scx * sx, scy * sy),
        "diameter": gt["diameter_softmap_px"] * (sx + sy) / 2.0,
    }
