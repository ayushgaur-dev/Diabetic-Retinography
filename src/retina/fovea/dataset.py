"""IDRiD Localization dataset interface (SIH26038 Phase 4C).

Source: IDRiD (Indian Diabetic Retinopathy Image Dataset, Nanded,
Maharashtra; 50-degree FOV, 4288x2848 JPG). Acquired here as
'C. Localization.zip' (~203 MB) via a Zenodo mirror of the IEEE Dataport
release — dataset NEVER committed.

Layout under the data root:

  C. Localization/1. Original Images/a. Training Set/IDRiD_NNN.jpg (413)
  C. Localization/1. Original Images/b. Testing Set/IDRiD_NNN.jpg  (103)
  C. Localization/2. Groundtruths/1. Optic Disc Center Location/*OD_Center*Markups.csv
  C. Localization/2. Groundtruths/2. Fovea Center Location/*Fovea_Center*Markups.csv

Annotation CSVs: one row per dataset image (515 rows each); only the rows
of the matching split carry coordinates ('X- Coordinate' = COLUMN x,
'Y - Coordinate' = ROW y — verified: GT points land on anatomy, and
fovea/disc relative positions match expected laterality geometry).
Blank rows are skipped, never fabricated.

Root resolution: --data-root > IDRID_DATA_ROOT env > data/idrid/ (empty
placeholder). Usable annotated images: 413 train + 103 test (no missing
annotations within either split).
"""

import csv
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_root(root=None):
    if root:
        return Path(root)
    env = os.environ.get("IDRID_DATA_ROOT")
    if env:
        return Path(env)
    return REPO_ROOT / "data" / "idrid"


def _localization_base(root):
    cands = sorted(Path(root).rglob("C. Localization"))
    if not cands:
        raise FileNotFoundError(
            f"IDRiD 'C. Localization' not found under {root}. "
            "Set IDRID_DATA_ROOT or see docs/FOVEA_LOCALIZATION.md.")
    return cands[0]


def _read_centers(csv_path):
    out = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        xcol = next(c for c in reader.fieldnames if "X" in c)
        ycol = next(c for c in reader.fieldnames if "Y" in c)
        idcol = next(c for c in reader.fieldnames if "Image" in c)
        for row in reader:
            _id = (row[idcol] or "").strip()
            try:
                x, y = float(row[xcol]), float(row[ycol])
            except (ValueError, TypeError):
                continue  # blank non-split row — skip, never fabricate
            if _id:
                out[_id] = (x, y)  # (column, row)
    return out


def discover(root=None, split="test", with_disc=True):
    """Manifest entries: id, split, image path, fovea (x,y), disc (x,y|None).
    split in {'training', 'test', 'all'}. Raises on missing files."""
    base = _localization_base(resolve_root(root))
    splits = ("training", "test") if split == "all" else (split,)
    kind = {"training": "a. Training Set", "test": "b. Testing Set"}
    gt = base / "2. Groundtruths"
    fovea_files = {
        "training": next(gt.rglob("*Fovea*Training*.csv")),
        "test": next(gt.rglob("*Fovea*Test*.csv")),
    }
    disc_files = {
        "training": next(gt.rglob("*OD_Center*Training*.csv")),
        "test": next(gt.rglob("*OD_Center*Test*.csv")),
    }
    entries = []
    for sp in splits:
        fovea_gt = _read_centers(fovea_files[sp])
        disc_gt = _read_centers(disc_files[sp]) if with_disc else {}
        img_dir = base / "1. Original Images" / kind[sp]
        for img_path in sorted(img_dir.glob("IDRiD_*.jpg")):
            _id = img_path.stem
            if _id not in fovea_gt:
                raise FileNotFoundError(f"IDRiD fovea GT missing for {_id}.")
            entries.append({"id": _id, "split": sp, "image": str(img_path),
                            "fovea": fovea_gt[_id],
                            "disc": disc_gt.get(_id)})
    if not entries:
        raise FileNotFoundError(f"No IDRiD {split} images found.")
    return entries
