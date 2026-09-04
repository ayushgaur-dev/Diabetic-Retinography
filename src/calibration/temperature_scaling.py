"""Temperature scaling (SIH26038 Phase 7). Numpy-only, single scalar T.

Logit recovery: the artifact exposes softmax probabilities only.
logits = log(p) is EXACT for temperature scaling (softmax shift
invariance: adding a constant to all logits leaves softmax(z/T)
unchanged). float64 + eps floor for stability. The artifact is untouched;
true pre-softmax logits would be preferred if ever exposed.

Optimization: deterministic golden-section search on t = log(T) minimizing
validation multiclass NLL. No random init, no scipy dependency.
"""

import numpy as np


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def recover_logits(probs, eps=1e-12):
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    return np.log(p)


def apply_temperature(probs, temperature, eps=1e-12):
    """Calibrated probabilities = softmax(log(p) / T). T=1 is identity."""
    t = float(temperature)
    if not t > 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}.")
    return softmax(recover_logits(probs, eps) / t)


def multiclass_nll(probs, labels, eps=1e-12):
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    y = np.asarray(labels, dtype=int)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def _nll_at_logT(logT, logits, labels):
    return multiclass_nll(softmax(logits / np.exp(logT)), labels)


def fit_temperature(val_probs, val_labels, initial_temperature=1.0,
                    logT_bounds=(-3.0, 3.0), tolerance=1e-6, max_iterations=200):
    """Golden-section search on log(T). VALIDATION arrays only — structural
    isolation: this function never sees test data (tested). Returns dict
    with fitted T, NLLs, and iteration count."""
    logits = recover_logits(val_probs)
    nll_before = multiclass_nll(np.asarray(val_probs, dtype=float),
                                 np.asarray(val_labels, dtype=int))
    lo, hi = logT_bounds
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc, fd = _nll_at_logT(c, logits, val_labels), _nll_at_logT(d, logits, val_labels)
    it = 0
    while (hi - lo) > tolerance and it < max_iterations:
        if fc < fd:
            hi, fd = d, fc
            d = c
            c = hi - inv_phi * (hi - lo)
            fc = _nll_at_logT(c, logits, val_labels)
        else:
            lo, fc = c, fd
            c = d
            d = lo + inv_phi * (hi - lo)
            fd = _nll_at_logT(d, logits, val_labels)
        it += 1
    logT = (lo + hi) / 2.0
    fitted = float(np.exp(logT))
    nll_after = _nll_at_logT(logT, logits, val_labels)
    return {"temperature": fitted,
            "initial_temperature": float(initial_temperature),
            "nll_before": round(float(nll_before), 4),
            "nll_after": round(float(nll_after), 4),
            "iterations": it, "tolerance": tolerance,
            "logT_bounds": list(logT_bounds)}
