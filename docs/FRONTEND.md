# Frontend Website (SIH26038 — Phase 10B)

## Architecture

```
browser (Next.js 14, React 18, Tailwind 3, TypeScript)
   │  POST /api/screen (multipart) → {job_id, image_hash}
   │  GET  /api/jobs/{id}          → {state, stage, result, layers} (poll)
   │  GET  /api/report/{hash}/{json|md|html|pdf}
   │  POST /api/demo               → synthetic fixture job
   ▼
FastAPI (backend/) — thin adapter, no pipeline logic
   │  calls app/screening_pipeline.run_screening() (frozen)
   ▼
Phases 2–9 (unchanged, authoritative)
```

The Streamlit app (Phase 10A) remains the reference implementation and
is untouched. The frontend is a presentation layer: it never computes
grades, probabilities, calibration, triage, or metrics.

## Design system

Editorial + scientific: oversized Didot/Bodoni/Georgia display type,
system sans body, warm paper `#FAF6EF`, near-black `#12100D`, restrained
arterial-red `#B3352B` accent, technical grids, uppercase micro-labels,
asymmetric layouts, light/dark chapter transitions. Original generative
retinal artwork (`public/hero-retina.png`, seeded script — non-PHI).
Reduced-motion respected. Product name: **RetinaLens** (temporary;
VISION/EVIDENCE/CLARITY identity; no existing brand was defined).

## Frontend/backend communication

`frontend/lib/api.ts` (base URL via `NEXT_PUBLIC_API_URL`, default
`http://127.0.0.1:8000`). Job lifecycle: queued → running (stage from
the pipeline's real `on_stage` hook: quality/enhancement/grading/
evidence/triage) → done|error. Result screen renders backend values
verbatim, including UNGRADABLE (no grade shown) and BORDERLINE states.
Report buttons hit the backend's Phase 9 renderers directly. Demo mode
runs a seeded synthetic fundus through the real pipeline, always
labelled synthetic.

## Local setup

Backend (from repo root, venv active):
`pip install -r backend/requirements.txt`
`.\venv\Scripts\python.exe -m uvicorn backend.api:app --port 8001`
(npm scripts are `.cmd`-invoked on Windows: execution policy blocks `.ps1`.)

Frontend:
`cd frontend; npm.cmd install`
dev: `node node_modules/next/dist/bin/next dev`
prod: `node node_modules/next/dist/bin/next build` then `.../next start -p 3001`
Set `NEXT_PUBLIC_API_URL` when the API is not on 127.0.0.1:8000.

## Production build

`next build` passes (tsc + lint + static prerender: `/` 99.6 kB,
`/screen` 102 kB first-load JS). Verified served pages return HTTP 200
with full content.

## API contract

See `backend/schemas.py` (pydantic v2). Key invariants (tested):
unreadable/oversize/wrong-type uploads rejected; unknown jobs 404;
report bytes reproduce the stored triage decision; blocked results carry
no grade; missing evidence degrades to warnings.

## Privacy

No PHI collected (no names/ages/history fields exist); images processed
in-memory, results held in a process-local job store (no DB, no logs of
pixels); no analytics/telemetry/CDN; no external AI services; CORS
limited to localhost:3000.

## Demo mode

POST /api/demo → deterministic synthetic fundus (seeded, textured to be
gradable) through the full pipeline. Verified end-to-end: GOOD →
grade + lesions + triage + all four report formats.

## Known limitations

- Full-evidence screening takes ~50 s–4 min/image CPU; the timeline
  shows real backend stages while it runs.
- Enhancement comparison shows before/after status (no pixel slider —
  the pipeline retains status, not the enhanced array).
- Neovascularization detection is not implemented (stated in UI).
- Landing-page engineering metrics are static held-out results with
  non-validation disclaimers.
- `node_modules/`, `.next/` are gitignored build artifacts.
