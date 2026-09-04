"""Confusion helpers: adjacent-pair and severe-mistake tallies (Phase 5)."""

ADJACENT_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4)]
SEVERE_MISTAKES = [(3, 0), (4, 0), (4, 1)]  # (truth, prediction)


def adjacent_report(cm):
    return {f"{a}<->{b}": {"truth_a_pred_b": int(cm[a, b]),
                            "truth_b_pred_a": int(cm[b, a]),
                            "total": int(cm[a, b] + cm[b, a])}
            for a, b in ADJACENT_PAIRS}


def severe_report(cm):
    return {f"{t}->{p}": int(cm[t, p]) for t, p in SEVERE_MISTAKES}
