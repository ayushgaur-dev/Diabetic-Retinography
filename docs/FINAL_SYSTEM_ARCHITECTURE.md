# Final System Architecture (SIH26038 — Phase 13)

## Component diagram (data flow)

```
fundus image (bytes)
  -> app/screening_pipeline.run_screening()            [orchestrator, UI-side]
  -> src/quality.assess_image()                        [Phase 2 gate]
  -> src/preprocessing.enhance_image() (BORDERLINE)    [Phase 3]
  -> EfficientNetB0 (.keras, frozen)                   [Phase 1/5]
  -> src/calibration.apply_temperature()               [Phase 7, T=0.9542]
  -> src/retina/{vessels,disc->fovea,lesions}          [Phase 4A-D]
  -> src/explainability.explain_image()                [Phase 6]
  -> src/triage.decide()                               [Phase 8]
  -> src/reporting.generate()                          [Phase 9]
  -> Streamlit app/  |  FastAPI backend/  |  Next.js frontend/
  -> MATLAB import (Phase 9 JSON)  |  Simulink (workflow simulation)
```

## Authoritative source per value

| Value | Source of truth | Consumers (read-only) |
|---|---|---|
| Grade, raw probabilities | EfficientNetB0 inference | calibration, triage, reports, UIs, API, MATLAB import |
| Calibrated probabilities | `apply_temperature` | triage, reports, UIs, API |
| Referable score/decision | calibrated P2+P3+P4 @ frozen 0.7 | triage, reports, UIs |
| Quality state | `assess_image` + shared JSON thresholds | gate, triage, reports, UIs |
| Lesion/vessel/disc/fovea | Phase 4 modules | explainability, triage flags, reports, UIs |
| Grad-CAM/consistency | Phase 6 (original path preserved) | reports, UIs |
| Triage decision | `src/triage.decision.decide` | reports, UIs, API (verbatim) |
| Report content | `src/reporting` deterministic generator | downloads, MATLAB import |

## API boundaries

- Python→UI: `run_screening()` plain-dict contract (numbers + display arrays).
- HTTP: `backend/schemas.py` (pydantic v2); frontend `lib/types.ts` mirrors it.
- MATLAB: Phase 9 `report.json` via `load_screening_json` + `validate_report`.
- Simulink: scenario/report structs (no clinical data crosses).

## Safety gates (in pipeline order)

1. UNGRADABLE → grading never called (`gate_routing`, no fabricated grade).
2. Enhancement failure → grading withheld.
3. Grading exception → TECHNICAL_REVIEW routing.
4. Invalid inputs → TECHNICAL_REVIEW fallback.
5. Evidence conflict → human review (grade preserved).
6. Low calibrated confidence → review for ROUTINE-bound cases.

## Failure propagation

Per-stage try/except in the orchestrator records `errors{stage}` and
either routes (blocked paths) or continues with missing-optional
warnings (evidence stages). UIs render errors as messages, never
tracebacks; stale cases are invalidated by content hash.

## Boundaries

- Frontend/backend: JSON over HTTP; frontend never computes clinical values.
- MATLAB: read-only import + demonstration re-implementations (shared thresholds).
- Simulink: operational simulation over measured mixes; no AI internals.
