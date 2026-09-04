"""Deterministic screening reports (SIH26038 Phase 9). Read-only over
phase outputs; authoritative JSON + Markdown/HTML/PDF renderings."""

from src.reporting.pipeline import generate_report, render_all

__all__ = ["generate_report", "render_all"]
