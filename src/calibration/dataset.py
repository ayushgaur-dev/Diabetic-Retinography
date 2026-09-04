"""Phase 5 split reuse + verification (SIH26038 Phase 7).

No new random split: the exact Phase 5 procedure is reconstructed and the
resulting test IDs are verified against reports/grading/per_image.csv
(the Phase 5 test artifact). Any mismatch aborts — silent split drift
would invalidate the held-out claim.
"""

from src.evaluation import dataset as DS
from src.evaluation.config import load_eval_config


def load_phase5_splits(data_root):
    """Return (splits, eval_cfg) using the Phase 5 procedure verbatim."""
    ecfg = load_eval_config()
    labels = DS.load_labels(DS.resolve_root(data_root), ecfg)
    return DS.notebook_split(labels, ecfg), ecfg


def verify_against_phase5_artifact(splits, artifact_path):
    """Assert reconstructed test IDs == IDs in the Phase 5 per_image.csv."""
    import pandas as pd

    saved = pd.read_csv(artifact_path)
    a = sorted(splits["test"]["id_code"].tolist())
    b = sorted(saved["image_id"].tolist())
    if a != b:
        raise ValueError(
            f"Split drift: reconstructed test ({len(a)} IDs) != Phase 5 "
            f"artifact ({len(b)} IDs). Aborting.")
    return {"n_test": len(a), "match": True}
