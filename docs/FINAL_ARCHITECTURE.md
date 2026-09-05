# Final Architecture (SIH26038 — Phase 14)

```
FUNDUS IMAGE (bytes, SHA-256)
  │
  ▼
QUALITY ASSESSMENT ── UNGRADABLE ──► recapture (grading never called)
  │  (src/quality, shared JSON thresholds)
  ▼
ENHANCEMENT (BORDERLINE only; accepted/rejected recorded)
  │  (src/preprocessing)
  ▼
DR GRADING ── EfficientNetB0, frozen .keras ──► grade 0-4 + raw probs
  │  (src/evaluation/inference; test QWK 0.811)
  ▼
CALIBRATION ── temperature scaling, T=0.9542 ──► calibrated probs
  │  (src/calibration; argmax provably unchanged)
  ▼
ANATOMICAL ANALYSIS ── vessels / optic disc / fovea (independent modules)
  │  (src/retina/*; heuristic localization, evidence wording)
  ▼
LESION EVIDENCE ── MA / HE / EX / SE candidate maps
  │  (src/retina/lesions; candidates, never confirmations)
  ▼
GRAD-CAM ── activation heatmap + evidence-overlap consistency
  │  (src/explainability; attribution aid, not proof)
  ▼
TRIAGE ── UNGRADABLE / ROUTINE / REFER / URGENT_REVIEW / TECHNICAL_REVIEW
  │  (src/triage deterministic rules; grade immutable)
  ▼
REPORT ── JSON (authoritative) / Markdown / HTML / PDF
   (src/reporting deterministic generator)
```

## Interfaces

- Python Core: `app/screening_pipeline.run_screening()` orchestrates
  everything above; per-stage try/except; content-hash invalidation.
- Streamlit Reference (`app/streamlit_app.py`): calls the orchestrator.
- FastAPI (`backend/`): job queue over the orchestrator; PNG data-URI
  layers; Phase 9 report rendering from stored results.
  - Next.js Frontend (`frontend/`): presentation only; derives API host
    at runtime; never computes clinical values.
- Phase 9 JSON ─► MATLAB (`load_screening_json` + `validate_report`;
  demonstration re-implementations share `configs/*.json`).
  - MATLAB structs ─► Simulink DES (scenario/report structs; operational
    tokens, no clinical data).

## Authoritative sources

Python pipeline = AI/clinical-result source. Phase 8 = triage source.
Phase 9 JSON = report source. MATLAB = integration/visualization.
Simulink = workflow simulation. Frontend renders; never decides.
