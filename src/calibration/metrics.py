"""Calibration metrics: NLL, Brier, ECE, MCE (SIH26038 Phase 7).

ECE: equal-width bins over max-probability confidence (default 10;
15/20 supported). Per-bin (mean confidence, empirical accuracy, count).
Classwise: one-vs-rest Brier + ECE per grade (minority classes flagged
unstable when support is small).
"""

import numpy as np


def brier_score(probs, labels):
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return float(((p - oh) ** 2).mean())


def _bin_stats(conf, correct, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        n = int(sel.sum())
        bins.append({
            "bin": [round(float(lo), 3), round(float(hi), 3)],
            "count": n,
            "mean_confidence": round(float(conf[sel].mean()), 4) if n else None,
            "accuracy": round(float(correct[sel].mean()), 4) if n else None,
        })
    return bins


def ece_metrics(probs, labels, n_bins=10):
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = _bin_stats(conf, correct, n_bins)
    n = len(y)
    ece = sum(b["count"] / n * abs(b["accuracy"] - b["mean_confidence"])
              for b in bins if b["count"])
    mce = max((abs(b["accuracy"] - b["mean_confidence"]) for b in bins if b["count"]),
              default=0.0)
    return {"ece": round(float(ece), 4), "mce": round(float(mce), 4),
            "n_bins": n_bins, "mode": "equal_width", "bins": bins}


def classwise_metrics(probs, labels, n_bins=10, min_support=20):
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    out = {}
    for g in range(p.shape[1]):
        yg = (y == g).astype(int)
        pg = np.stack([1 - p[:, g], p[:, g]], axis=1)
        brier = float(((pg[:, 1] - yg) ** 2).mean())
        conf = pg[:, 1]
        bins = _bin_stats(conf, yg.astype(float), n_bins)
        n = len(y)
        ece = sum(b["count"] / n * abs(b["accuracy"] - b["mean_confidence"])
                  for b in bins if b["count"])
        out[str(g)] = {"support": int(yg.sum()),
                       "one_vs_rest_brier": round(brier, 4),
                       "one_vs_rest_ece": round(float(ece), 4),
                       "stable": bool(yg.sum() >= min_support)}
    return out


def binary_metrics(scores, labels, n_bins=10):
    """Referable-score calibration: binary Brier + ECE on P(grade>=2)."""
    s = np.clip(np.asarray(scores, dtype=float), 0, 1)
    y = np.asarray(labels, dtype=int)
    brier = float(((s - y) ** 2).mean())
    bins = _bin_stats(s, y.astype(float), n_bins)
    n = len(y)
    ece = sum(b["count"] / n * abs(b["accuracy"] - b["mean_confidence"])
              for b in bins if b["count"])
    return {"brier": round(brier, 4), "ece": round(float(ece), 4),
            "n_bins": n_bins, "bins": bins}


def confidence_distribution(probs, labels):
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == np.asarray(labels, dtype=int))
    q = lambda a: [round(float(v), 4) for v in np.percentile(a, [5, 25, 50, 75, 95])] \
        if len(a) else None
    return {
        "mean": round(float(conf.mean()), 4),
        "median": round(float(np.median(conf)), 4),
        "std": round(float(conf.std()), 4),
        "quantiles_5_25_50_75_95": q(conf),
        "correct": {"n": int(correct.sum()), "mean": round(float(conf[correct].mean()), 4)
                    if correct.any() else None},
        "incorrect": {"n": int((~correct).sum()),
                      "mean": round(float(conf[~correct].mean()), 4)
                      if (~correct).any() else None},
    }


def bootstrap_cis(values_fn, records_probs, records_labels, n_resamples=1000,
                  seed=42):
    """Generic deterministic bootstrap over per-resample metric dicts."""
    rng = np.random.default_rng(seed)
    n = len(records_labels)
    acc = {}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        m = values_fn(np.asarray(records_probs)[idx], np.asarray(records_labels)[idx])
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    return {k: [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)] for k, v in acc.items()}
