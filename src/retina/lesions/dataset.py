"""IDRiD Segmentation dataset interface (SIH26038 Phase 4D).

Source: IDRiD 'A. Segmentation.zip' (~557 MB) via Zenodo mirror — 54
training + 27 testing JPGs (4288x2848) with per-lesion TIF masks:

  1. Original Images/{a. Training Set,b. Testing Set}/IDRiD_NN.jpg
  2. All Segmentation Groundtruths/{...}/{1. Microaneurysms/IDRiD_NN_MA.tif,
     2. Haemorrhages/IDRiD_NN_HE.tif, 3. Hard Exudates/IDRiD_NN_EX.tif,
     4. Soft Exudates/IDRiD_NN_SE.tif, 5. Optic Disc/IDRiD_NN_OD.tif}

Mask loader is defensive: any shape (H,W)/(H,W,C) and any values are
binarized (>0 on the first channel). Test IDs run IDRiD_55..81.

Known gaps (explicit, never silent): train IDRiD_43 has no HE file; SE
files exist only for 26/54 train and 14/27 test images. A missing mask
file is recorded as gt_present=False and treated as all-negative GT with
an explicit flag (standard IDRiD reading: unshipped = no such lesion
annotated). Masks are pixel-aligned with images (verified identical
shapes), so no row/column ambiguity exists.

Root: --data-root > IDRID_SEG_DATA_ROOT (falls back to IDRID_DATA_ROOT) >
data/idrid-seg/. Dataset NEVER committed.
"""

import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]

LESION_FILES = {
    "microaneurysm": ("1. Microaneurysms", "MA"),
    "hemorrhage": ("2. Haemorrhages", "HE"),
    "hard_exudate": ("3. Hard Exudates", "EX"),
    "soft_exudate": ("4. Soft Exudates", "SE"),
    "optic_disc": ("5. Optic Disc", "OD"),
}


def resolve_root(root=None):
    if root:
        return Path(root)
    for var in ("IDRID_SEG_DATA_ROOT", "IDRID_DATA_ROOT"):
        env = os.environ.get(var)
        if env:
            return Path(env)
    return REPO_ROOT / "data" / "idrid-seg"


def _seg_base(root):
    cands = sorted(Path(root).rglob("A. Segmentation"))
    if not cands:
        raise FileNotFoundError(
            f"IDRiD 'A. Segmentation' not found under {root}. "
            "Set IDRID_SEG_DATA_ROOT or see docs/LESION_EVIDENCE.md.")
    return cands[0]


def load_mask(path, shape=None):
    """Robust binarization: any channels/values -> uint8 0/255. Shape-checked."""
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[:, :, 0]
    m = ((a > 0).astype(np.uint8)) * 255
    if shape is not None and m.shape != shape:
        raise ValueError(f"Mask {path} shape {m.shape} != image shape {shape}.")
    return m


def discover(root=None, split="test"):
    """Manifest entries: id, split, image, masks {lesion: path|None},
    gt_present {lesion: bool}. split in {'training','test','all'}."""
    base = _seg_base(resolve_root(root))
    kind = {"training": "a. Training Set", "test": "b. Testing Set"}
    splits = ("training", "test") if split == "all" else (split,)
    entries = []
    for sp in splits:
        img_dir = base / "1. Original Images" / kind[sp]
        gt_dir = base / "2. All Segmentation Groundtruths" / kind[sp]
        for img_path in sorted(img_dir.glob("IDRiD_*.jpg")):
            _id = img_path.stem
            masks, present = {}, {}
            for lesion, (sub, suffix) in LESION_FILES.items():
                cands = sorted((gt_dir / sub).glob(f"{_id}_{suffix}.tif"))
                masks[lesion] = str(cands[0]) if cands else None
                present[lesion] = masks[lesion] is not None
            entries.append({"id": _id, "split": sp, "image": str(img_path),
                            "masks": masks, "gt_present": present})
    if not entries:
        raise FileNotFoundError(f"No IDRiD segmentation {split} images found.")
    return entries
