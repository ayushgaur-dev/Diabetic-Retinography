"""Evidence triage engine (SIH26038 Phase 8). Deterministic workflow
recommendations; model prediction immutable. Legacy src/rules/triage.py
is preserved untouched (different schema); safety concepts reused."""

from src.triage.decision import decide
from src.triage.pipeline import build_input, triage_from_phases

__all__ = ["decide", "build_input", "triage_from_phases"]
