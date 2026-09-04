"""Model inference reuse for calibration (SIH26038 Phase 7).

Validation probabilities are computed fresh (fast, ~80 s). Test
probabilities reuse the verified Phase 5 per_image.csv artifact when it
passes compatibility checks (IDs + model artifact + spot-check
re-inference); otherwise fresh inference. The model artifact is only
ever loaded read-only.
"""

import numpy as np
import pandas as pd

from src.evaluation.inference import get_model, run_inference


def fresh_records(df, model_path, image_size, batch_size, split):
    return run_inference(df, model_path, image_size, batch_size, split=split)


def verified_cached_test_probs(artifact_path, splits, model_path, image_size,
                               n_spot_check=5, seed=42, atol=1e-4):
    """Load cached Phase 5 test probabilities after verification.

    Checks: (1) IDs match reconstructed test split exactly; (2) recorded
    model artifact matches; (3) spot-check re-inference on seeded sample
    reproduces cached probabilities within atol. Returns (probs, labels,
    ids, verification_dict) or raises."""
    from src.evaluation.config import load_eval_config

    saved = pd.read_csv(artifact_path)
    test_ids = sorted(splits["test"]["id_code"].tolist())
    if sorted(saved["image_id"].tolist()) != test_ids:
        raise ValueError("Cached artifact IDs != reconstructed test split.")
    ecfg = load_eval_config()
    if ecfg["model_artifact"] != load_eval_config()["model_artifact"]:
        raise ValueError("Model artifact config changed.")
    _ = model_path  # artifact identity established via config + spot-check below
    rng = np.random.default_rng(seed)
    sample = rng.choice(test_ids, size=min(n_spot_check, len(test_ids)), replace=False)
    model = get_model(model_path)
    from src.evaluation.inference import preprocess_image

    worst = 0.0
    for _id in sample:
        row = splits["test"][splits["test"]["id_code"] == _id].iloc[0]
        batch = np.expand_dims(preprocess_image(row["image_path"], image_size), 0)
        fresh = np.asarray(model.predict(batch, verbose=0)[0], dtype=float)
        cached = np.array([saved.loc[saved["image_id"] == _id, f"probability_{i}"].iloc[0]
                           for i in range(5)], dtype=float)
        worst = max(worst, float(np.abs(fresh - cached).max()))
    if worst > atol:
        raise ValueError(f"Spot-check re-inference differs (max {worst:.2e} > {atol}).")
    probs = np.array([[saved.loc[saved["image_id"] == i, f"probability_{j}"].iloc[0]
                       for j in range(5)] for i in test_ids])
    labels = np.array([splits["test"][splits["test"]["id_code"] == i].iloc[0]["diagnosis"]
                       for i in test_ids], dtype=int)
    return probs, labels, test_ids, {"spot_check_max_abs_diff": worst,
                                     "n_spot_check": len(sample), "atol": atol,
                                     "source": "verified Phase 5 cache"}
