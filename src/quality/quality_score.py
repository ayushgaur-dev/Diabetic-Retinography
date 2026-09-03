"""Quality aggregation rules (SIH26038 Phase 2).

Explicit, auditable rules — no silent averaging:
  IF any component is BAD    -> UNGRADABLE
  ELSE IF any is BORDERLINE  -> BORDERLINE (enhancement_required=True; Phase 3 hook)
  ELSE                       -> GOOD

overall_score is the config-weighted mean of component scores and is
ADVISORY ONLY — the gate acts on status, never on the score.
"""

from src.quality.types import BAD, BORDERLINE, GOOD, UNGRADABLE


def aggregate(components, cfg):
    """components: dict name -> ComponentResult. Returns (status, overall_score, reasons)."""
    weights = cfg.get("aggregation", {}).get("weights", {})
    total_w, weighted = 0.0, 0.0
    for name, comp in components.items():
        w = float(weights.get(name, 1.0))
        weighted += w * comp.score
        total_w += w
    overall = weighted / total_w if total_w else 0.0
    bad = [c for c in components.values() if c.status == BAD]
    borderline = [c for c in components.values() if c.status == BORDERLINE]
    if bad:
        status = UNGRADABLE
        reasons = [f"{c.name}: {c.explanation}" for c in bad]
    elif borderline:
        status = BORDERLINE
        reasons = [f"{c.name}: {c.explanation}" for c in borderline]
    else:
        status = GOOD
        reasons = []
    return status, overall, reasons
