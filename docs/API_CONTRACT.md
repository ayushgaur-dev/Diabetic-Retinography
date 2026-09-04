# API Contract (SIH26038 — FastAPI backend)

Base URL: `http://127.0.0.1:8001` (configurable). All screening values
originate in `run_screening()`; the API transports them unchanged
(pydantic v2 schemas in `backend/schemas.py`; frontend mirrors in
`frontend/lib/types.ts`).

## GET /api/health → 200 `{status, pipeline, frozen, llm: none}`

## POST /api/screen → 200 `{job_id, image_hash}` | 400 | 404(n/a)

- Request: `multipart/form-data` with `file` (PNG/JPEG, ≤ 15 MB);
  optional query `full_evidence` (default true).
- Errors: 400 unsupported type / oversize / unreadable image.
- Effect: starts background job; returns immediately.

## GET /api/jobs/{job_id} → 200 JobStatus | 404 unknown job

- `state`: queued|running|done|error; `stage`: quality|enhancement|
  grading|evidence|triage|done; on done: `result` (ScreeningSummary) +
  `layers` (PNG data URIs: original/gradcam/vessels/anatomy/lesions/
  combined, whichever exist); on error: `error` (truncated, no traceback).

## POST /api/demo → 200 `{job_id, image_hash, synthetic, note}`

Deterministic synthetic fundus through the real pipeline; always
labelled synthetic.

## GET /api/report/{hash}/{json|md|html|pdf} → file | 400 | 404

- Phase 9 rendering of the stored result; JSON authoritative.
- 400 bad format; 404 unknown hash.

## GET /api/demo-info → 200 DemoInfo

## Frontend parity

`ScreeningSummary` fields map 1:1 to `lib/types.ts`; grade/probabilities/
triage/reasons render verbatim. Polling: 1.2 s interval, 10 min timeout.
CORS: localhost:3000 only. No auth (local/demo deployment scope).
