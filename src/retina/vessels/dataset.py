"""DRIVE dataset loader (SIH26038 Phase 4A).

Expected layout (DRIVE grand-challenge + the Kaggle mirror used here):

  <root>/DRIVE/training/images/{NN}_training.tif      (NN = 21..40)
  <root>/DRIVE/training/1st_manual/{NN}_manual1.gif   (vessel ground truth)
  <root>/DRIVE/training/mask/{NN}_training_mask.gif   (FOV mask)
  <root>/DRIVE/test/images/{NN}_test.tif              (NN = 01..20)
  <root>/DRIVE/test/mask/{NN}_test_mask.gif           (FOV mask, no GT here)

Dataset root resolution (no hard-coded absolute path):
  1. explicit `root` argument, 2. DRIVE_DATA_ROOT env var,
  3. repo-local data/drive (empty by default; dataset is NEVER committed).

Pairing is validated strictly: an image without its annotation/FOV partner
raises FileNotFoundError naming the missing file — never a silent mismatch.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_ROOT = REPO_ROOT / "data" / "drive"


def resolve_root(root=None):
    if root:
        return Path(root)
    env = os.environ.get("DRIVE_DATA_ROOT")
    if env:
        return Path(env)
    return DEFAULT_LOCAL_ROOT


def _require(path, what):
    if not Path(path).exists():
        raise FileNotFoundError(f"DRIVE {what} missing: {path}")
    return str(path)


def discover_split(drive_dir, split):
    """Return manifest entries for one split. Each entry: id, split, image,
    manual (None for test — this mirror ships no test annotations), fov."""
    drive_dir = Path(drive_dir)
    entries = []
    if split == "training":
        img_dir, man_dir, mask_dir = (drive_dir / "training" / "images",
                                      drive_dir / "training" / "1st_manual",
                                      drive_dir / "training" / "mask")
        ids = sorted(p.stem.replace("_training", "")
                     for p in img_dir.glob("*_training.tif"))
        for _id in ids:
            image = _require(img_dir / f"{_id}_training.tif", "image")
            manual = _require(man_dir / f"{_id}_manual1.gif", "vessel annotation")
            fov = _require(mask_dir / f"{_id}_training_mask.gif", "FOV mask")
            entries.append({"id": _id, "split": split, "image": image,
                            "manual": manual, "fov": fov})
    elif split == "test":
        img_dir, mask_dir = drive_dir / "test" / "images", drive_dir / "test" / "mask"
        ids = sorted(p.stem.replace("_test", "") for p in img_dir.glob("*_test.tif"))
        for _id in ids:
            image = _require(img_dir / f"{_id}_test.tif", "image")
            fov = _require(mask_dir / f"{_id}_test_mask.gif", "FOV mask")
            entries.append({"id": _id, "split": split, "image": image,
                            "manual": None, "fov": fov})
    else:
        raise ValueError(f"Unknown DRIVE split: {split!r} (expected 'training'/'test').")
    if not entries:
        raise FileNotFoundError(f"No DRIVE {split} images found under {drive_dir}.")
    return entries


def discover(root=None):
    """Discover both splits. `root` may point at the dataset root itself or
    at its inner DRIVE/ directory (mirror nests one extra level)."""
    base = resolve_root(root)
    drive_dir = base / "DRIVE" if (base / "DRIVE").is_dir() else base
    if not drive_dir.is_dir():
        raise FileNotFoundError(
            f"DRIVE dataset not found at {base} (nor {base / 'DRIVE'}). "
            "Set DRIVE_DATA_ROOT or see docs/VESSEL_SEGMENTATION.md.")
    return discover_split(drive_dir, "training") + discover_split(drive_dir, "test")
