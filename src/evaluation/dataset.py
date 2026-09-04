"""APTOS dataset loading + notebook-split reconstruction (SIH26038 Phase 5).

Mirror layout (mariaherrerot/aptos2019): train_1.csv + valid.csv + test.csv
(id_code,diagnosis) plus nested train_images/val_images/test_images PNG
dirs. The three CSVs are COMBINED into one label table (= original
train.csv content: 3662 rows, verified) and re-split by the EXACT
notebook procedure (02[6]: 70/30 stratified seed 42, then temp 50/50
stratified seed 42). The mirror's own pre-split is not trusted.

Split integrity: test_df images were used for NEITHER training (train_ds)
NOR model/early-stopping decisions (val_ds) in the notebooks — test_ds
was constructed but never evaluated there (Phase 0/1 audit). This is an
IN-DATASET held-out test, not external validation (documented; APTOS has
no patient IDs, so image-level only).
"""

import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_root(root=None, env_var="APTOS_DATA_ROOT",
                 default=REPO_ROOT / "data" / "aptos"):
    import os

    if root:
        return Path(root)
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    return Path(default)


def load_labels(root, cfg):
    """Combine label CSVs; validate schema, grades, duplicates, file presence."""
    d = cfg["dataset"]
    frames = []
    for f in d["label_files"]:
        p = Path(root) / f
        if not p.exists():
            raise FileNotFoundError(f"APTOS label file missing: {p}.")
        df = pd.read_csv(p)
        for col in (d["id_field"], d["grade_field"]):
            if col not in df.columns:
                raise ValueError(f"{p} lacks column {col!r}.")
        frames.append(df[[d["id_field"], d["grade_field"]]])
    labels = pd.concat(frames, ignore_index=True)
    labels.columns = ["id_code", "diagnosis"]
    _validate_labels(labels)
    image_index = _index_images(root, cfg)
    missing = [i for i in labels["id_code"] if i not in image_index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} label IDs lack image files "
                                f"(e.g. {missing[:3]}).")
    labels["image_path"] = labels["id_code"].map(image_index)
    return labels


def _validate_labels(labels):
    dup = int(labels["id_code"].duplicated().sum())
    if dup:
        raise ValueError(f"{dup} duplicate image IDs in labels.")
    bad = labels[~labels["diagnosis"].isin([0, 1, 2, 3, 4])]
    if len(bad):
        raise ValueError(f"{len(bad)} malformed grade rows (e.g. {bad.head(3).to_dict()}).")
    if labels[["id_code", "diagnosis"]].isna().any().any():
        raise ValueError("Missing values in id_code/diagnosis.")


def _index_images(root, cfg):
    d = cfg["dataset"]
    index = {}
    for sub in d["image_dirs"]:
        for p in Path(root, sub).rglob(f"*{d['image_extension']}"):
            index.setdefault(p.stem, str(p))
    return index


def notebook_split(labels, cfg):
    """Exact notebook reconstruction. Returns dict train/val/test DataFrames."""
    from sklearn.model_selection import train_test_split

    s = cfg["split"]
    strat = labels["diagnosis"] if s["stratify"] else None
    train_df, temp_df = train_test_split(
        labels, test_size=round(1 - s["train_frac"], 10),
        stratify=strat, random_state=s["random_seed"])
    strat_t = temp_df["diagnosis"] if s["stratify"] else None
    val_df, test_df = train_test_split(
        temp_df, test_size=s["temp_test_frac"],
        stratify=strat_t, random_state=s["random_seed"])
    return {"train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True)}


def leakage_audit(splits, hash_audit=True):
    """IDs + file-hash audit. Returns dict (hashes cover all val+test images)."""
    ids = {k: set(v["id_code"]) for k, v in splits.items()}
    out = {
        "train_images": len(ids["train"]),
        "validation_images": len(ids["val"]),
        "test_images": len(ids["test"]),
        "train_test_overlap": sorted(ids["train"] & ids["test"]),
        "validation_test_overlap": sorted(ids["val"] & ids["test"]),
        "train_validation_overlap": sorted(ids["train"] & ids["val"]),
        "patient_overlap_status": ("UNKNOWN — APTOS train.csv carries no "
                                   "patient IDs; splits are image-level "
                                   "stratified (documented limitation)"),
    }
    out["duplicate_hashes"] = _hash_audit(splits) if hash_audit else "skipped"
    if hash_audit:
        groups = duplicate_groups(splits)
        cross = {h: v for h, v in groups.items()
                 if len({m.split(":")[0] for m in v}) > 1}
        out["duplicate_groups_all_splits"] = len(groups)
        out["cross_split_duplicate_groups"] = cross
        out["twin_grade_agreement"] = _twin_grades(splits, groups)
    return out


def _twin_grades(splits, groups):
    """Do byte-identical twins carry the same DR grade? (label-noise check)."""
    grade_of = {}
    for split in ("train", "val", "test"):
        for _, row in splits[split].iterrows():
            grade_of[f"{split}:{row['id_code']}"] = int(row["diagnosis"])
    agree = sum(1 for v in groups.values()
                if len({grade_of[m] for m in v}) == 1)
    return {"groups": len(groups), "unanimous_grade": agree,
            "mixed_grade": len(groups) - agree}


def _hash_audit(splits, which=("val", "test"), chunk=65536):
    """Hash-based duplicate detection. Returns list of duplicate PAIRS.
    Groups (all members per hash) come from duplicate_groups()."""
    seen, dups = {}, []
    for split in which:
        for _, row in splits[split].iterrows():
            digest = _file_hash(row["image_path"], chunk)
            if digest in seen:
                dups.append({"hash": digest, "first": seen[digest],
                             "second": f"{split}:{row['id_code']}"})
            else:
                seen[digest] = f"{split}:{row['id_code']}"
    return dups


def _file_hash(path, chunk=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def duplicate_groups(splits, which=("train", "val", "test"), chunk=65536):
    """hash -> sorted list of 'split:id' sharing identical file bytes."""
    groups = {}
    for split in which:
        for _, row in splits[split].iterrows():
            groups.setdefault(_file_hash(row["image_path"], chunk), []).append(
                f"{split}:{row['id_code']}")
    return {h: sorted(v) for h, v in groups.items() if len(v) > 1}
