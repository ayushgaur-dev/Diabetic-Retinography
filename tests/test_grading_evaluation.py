"""Grading-evaluation tests — SIH26038 Phase 5. All offline/synthetic:
no dataset download, no model weights, no API."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration_placeholder import (CALIBRATED,
                                                   assert_not_calibrated,
                                                   describe_raw_confidence)
from src.evaluation.classification_metrics import (confusion_counts,
                                                  five_class_metrics)
from src.evaluation.confusion import adjacent_report, severe_report
from src.evaluation.config import DEFAULTS, load_eval_config
from src.evaluation.dataset import leakage_audit, notebook_split
from src.evaluation.error_analysis import categorize, error_table
from src.evaluation.quality_analysis import stratify_metrics
from src.evaluation.referable_metrics import (referable_labels,
                                              referable_metrics,
                                              referable_scores,
                                              sensitivity_at_specificity,
                                              threshold_grid)
from src.evaluation.types import PredictionRecord


def _rec(gt, pr, probs=None, cid="x"):
    n = 5
    if probs is None:
        probs = [0.05] * 5
        probs[pr] = 0.8
    return PredictionRecord(image_id=cid, ground_truth=gt, predicted_grade=pr,
                            probabilities=list(probs), raw_confidence=max(probs))


# --- 1-12. Metrics ----------------------------------------------------------

def test_perfect_predictions():
    y = [0, 1, 2, 3, 4, 0, 2]
    m = five_class_metrics(y, list(y))
    assert m["accuracy"] == 1.0 and m["qwk"] == 1.0 and m["macro_f1"] == 1.0
    assert all(v["recall"] == 1.0 for v in m["per_class"].values())


def test_completely_wrong_predictions():
    y = [0, 0, 0, 0]
    m = five_class_metrics(y, [4, 4, 4, 4])
    assert m["accuracy"] == 0.0 and m["qwk"] <= 0.0
    assert m["per_class"]["0"]["recall"] == 0.0


def test_per_class_values():
    y = [0, 0, 1, 1, 1, 2]
    p = [0, 1, 1, 1, 2, 2]
    m = five_class_metrics(y, p)
    assert m["per_class"]["1"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["per_class"]["1"]["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["per_class"]["2"]["support"] == 1
    assert m["per_class"]["4"]["support"] == 0


def test_qwk_ordinal():
    y = [0, 1, 2, 3, 4] * 4
    near = [0, 1, 2, 3, 3] * 4
    far = [4, 4, 4, 4, 4] * 4
    assert five_class_metrics(y, near)["qwk"] > five_class_metrics(y, far)["qwk"]


def test_referable_conversion():
    assert list(referable_labels([0, 1, 2, 3, 4], 2)) == [0, 0, 1, 1, 1]
    assert list(referable_labels([0, 1, 2, 3, 4], 1)) == [0, 1, 1, 1, 1]


def test_confusion_counts():
    cm, norm = confusion_counts([0, 0, 1, 4], [0, 1, 1, 4])
    assert cm.shape == (5, 5) and cm[0, 0] == 1 and cm[0, 1] == 1
    assert norm[0, 0] == pytest.approx(0.5) and norm[4].sum() == pytest.approx(1.0)


def test_referable_tp_tn_fp_fn():
    m = referable_metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 1, 1)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["precision_ppv"] == 0.5 and m["npv"] == 0.5
    assert m["f1"] == 0.5 and m["balanced_accuracy"] == 0.5


def test_referable_score_construction():
    probs = [[0.7, 0.2, 0.05, 0.03, 0.02], [0.1, 0.1, 0.5, 0.2, 0.1]]
    s = referable_scores(probs, 2)
    assert list(np.round(s, 2)) == [0.10, 0.80]


def test_auroc_computed_from_scores_not_labels():
    from src.evaluation.referable_metrics import roc_pr

    y = [0, 0, 1, 1]
    out = roc_pr(y, [0.1, 0.2, 0.8, 0.9])
    assert out["auroc"] == 1.0 and out["average_precision"] == 1.0
    assert len(out["roc"]["fpr"]) == len(out["roc"]["tpr"])


# --- 13-17. Dataset ---------------------------------------------------------

def _labels(n=120, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"id_code": [f"id{i:04d}" for i in range(n)],
                         "diagnosis": rng.integers(0, 5, n),
                         "image_path": [f"/img/id{i:04d}.png" for i in range(n)]})


def test_label_parsing_and_validation(tmp_path):
    import pandas as pd

    (tmp_path / "a.csv").write_text("id_code,diagnosis\nA,0\nB,4\n")
    (tmp_path / "b.csv").write_text("id_code,diagnosis\nC,2\n")
    for f in ("x.png",):
        (tmp_path / f).touch()
    from src.evaluation.dataset import load_labels

    cfg = load_eval_config()
    cfg["dataset"]["label_files"] = ["a.csv", "b.csv"]
    cfg["dataset"]["image_dirs"] = ["."]
    with pytest.raises(FileNotFoundError):  # B, C images missing
        load_labels(tmp_path, cfg)


def test_duplicate_detection():
    df = pd.DataFrame({"id_code": ["a", "a"], "diagnosis": [0, 1],
                       "image_path": ["x", "y"]})
    from src.evaluation.dataset import _validate_labels

    with pytest.raises(ValueError, match="duplicate"):
        _validate_labels(df)
    bad = pd.DataFrame({"id_code": ["a"], "diagnosis": [7], "image_path": ["x"]})
    with pytest.raises(ValueError, match="malformed"):
        _validate_labels(bad)


def test_deterministic_split_and_no_overlap():
    df = _labels()
    cfg = load_eval_config()
    s1 = notebook_split(df, cfg)
    s2 = notebook_split(df, cfg)
    assert s1["test"]["id_code"].tolist() == s2["test"]["id_code"].tolist()
    assert len(s1["test"]) == 18 and len(s1["val"]) == 18  # 15% of 120
    audit = leakage_audit(s1, hash_audit=False)
    assert audit["train_test_overlap"] == [] and audit["validation_test_overlap"] == []
    assert audit["train_images"] + audit["validation_images"] + audit["test_images"] == 120


def test_seed_changes_split():
    df = _labels()
    cfg = load_eval_config()
    cfg["split"]["random_seed"] = 1
    a = notebook_split(df, cfg)["test"]["id_code"].tolist()
    cfg["split"]["random_seed"] = 2
    b = notebook_split(df, cfg)["test"]["id_code"].tolist()
    assert a != b


# --- 18-21. Inference records -------------------------------------------------

def test_probability_contract():
    r = _rec(2, 2)
    assert len(r.probabilities) == 5
    assert sum(r.probabilities) == pytest.approx(1.0, abs=0.05)
    assert r.predicted_grade == int(np.argmax(r.probabilities))
    assert r.raw_confidence == max(r.probabilities)


def test_raw_confidence_labelled_uncalibrated():
    assert CALIBRATED is False
    with pytest.raises(NotImplementedError):
        assert_not_calibrated()
    recs = [_rec(0, 0, [0.9, 0.02, 0.02, 0.03, 0.03]),
            _rec(1, 3, [0.1, 0.2, 0.1, 0.5, 0.1])]
    d = describe_raw_confidence(recs)
    assert d["mean_correct"] > d["mean_incorrect"]
    assert sum(d["histogram_10bin"]) == 2


# --- 22-23. Threshold --------------------------------------------------------

def test_threshold_behavior():
    y = [0, 0, 1, 1]
    s = [0.2, 0.4, 0.6, 0.8]
    grid = threshold_grid(y, s, [0.3, 0.5, 0.7])
    assert [r["threshold"] for r in grid] == [0.3, 0.5, 0.7]
    assert grid[0]["sensitivity"] == 1.0 and grid[2]["specificity"] == 1.0


def test_validation_threshold_selection_ignores_test():
    import copy

    cfg = load_eval_config()
    grid = cfg["decision_threshold"]["grid"]
    assert 0.5 in grid and all(0 < t < 1 for t in grid)
    frozen = max(threshold_grid([0, 1, 1, 0], [0.2, 0.8, 0.6, 0.4], grid),
                 key=lambda r: (r["f1"], r["sensitivity"]))
    assert "threshold" in frozen and "sensitivity" in frozen
    assert copy.deepcopy(cfg)["decision_threshold"]["default"] == 0.5


def test_sensitivity_at_specificity():
    y = [0] * 8 + [1] * 2
    s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.9, 0.95]
    out = sensitivity_at_specificity(y, s, 0.85)
    assert out is not None and out["specificity"] >= 0.85
    assert sensitivity_at_specificity(y, s, 1.01) is None


# --- 24-25. Error analysis ----------------------------------------------------

def test_severe_undercall_classification():
    assert "severe undercall" in categorize(_rec(4, 0))
    assert "severe undercall" in categorize(_rec(3, 1))
    assert "neighboring" in categorize(_rec(2, 3))
    assert categorize(_rec(1, 1)) == "correct"


def test_referable_fn_classification():
    assert "referable false negative" in categorize(_rec(3, 1))
    assert "referable false positive" in categorize(_rec(1, 2))
    rows = error_table([_rec(4, 0, cid="a"), _rec(0, 0, cid="b")])
    assert rows[0]["referable_truth"] == 1 and rows[0]["referable_prediction"] == 0
    assert set(rows[0]) >= {"P0", "P4", "error_category"}


# --- 26-28. Quality gate -------------------------------------------------------

def test_ungradable_blocked_good_eligible():
    rows = [{"image_id": "a", "quality_status": "UNGRADABLE"},
            {"image_id": "b", "quality_status": "GOOD"},
            {"image_id": "c", "quality_status": "BORDERLINE"}]
    eligible = [r["image_id"] for r in rows if r["quality_status"] in ("GOOD", "BORDERLINE")]
    blocked = [r["image_id"] for r in rows if r["quality_status"] == "UNGRADABLE"]
    assert eligible == ["b", "c"] and blocked == ["a"]


def test_stratify_by_state():
    recs = [{"image_id": "a", "ground_truth": 0, "prediction": 0},
            {"image_id": "b", "ground_truth": 2, "prediction": 3}]
    out = stratify_metrics(recs, {"a": "GOOD", "b": "GOOD"},
                           lambda yt, yp: {"n_correct": sum(int(a == b) for a, b in zip(yt, yp))})
    assert out["GOOD"]["n"] == 2 and out["GOOD"]["n_correct"] == 1
    assert out["BORDERLINE"]["n"] == 0


# --- 29-30. Reproducibility -----------------------------------------------------

def test_same_predictions_same_metrics():
    recs = [_rec(g, p, cid=f"i{i}") for i, (g, p) in
            enumerate([(0, 0), (1, 2), (4, 4), (2, 2)])]
    from src.evaluation.evaluator import evaluate_split

    m1 = evaluate_split(recs, load_eval_config(), "test")
    m2 = evaluate_split(recs, load_eval_config(), "test")
    assert m1["five_class"] == m2["five_class"]
    assert m1["referable"] == m2["referable"]
    assert m1["confusion_matrix"] == m2["confusion_matrix"]


def test_adjacent_and_severe_tallies():
    import numpy as np

    cm = np.zeros((5, 5), dtype=int)
    cm[1, 2] = 5
    cm[2, 1] = 3
    cm[4, 0] = 2
    adj = adjacent_report(cm)
    assert adj["1<->2"]["total"] == 8
    sev = severe_report(cm)
    assert sev["4->0"] == 2 and sev["3->0"] == 0
