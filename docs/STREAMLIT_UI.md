# Streamlit Reference UI (SIH26038 — Phase 10A)

## UI workflow

Upload (png/jpg/jpeg, SHA-256 shown, no PHI) → Quality → Enhancement
(BORDERLINE) → Grading → Calibration → Anatomy → Lesions →
Explainability → Triage → Report downloads → Limitations. Single
`app/streamlit_app.py` dashboard; all ML in frozen Phase 2–9 modules,
wired only through `app/screening_pipeline.py`.

## Section → phase mapping

Upload (new) → Quality (Phase 2: status, components, recapture) →
Enhancement (Phase 3: before/after, accepted/rejected) → Screening
result (Phase 5 grade + Phase 7 calibrated probs/confidence) →
Referable (grade≥2, frozen 0.7) → Anatomy (4A vessels, 4B disc, 4C
fovea; "localized/evidence" wording; NV noted unimplemented) → Lesions
(4D statuses/counts/boxes) → Explainability (Phase 6 overlays,
consistency, agreement) → Triage (Phase 8 verbatim) → Report (Phase 9
JSON/Markdown/HTML/PDF) → Limitations.

## State/caching strategy

- Model: `st.cache_resource` singleton.
- Cases: `session_state["case"]` keyed by upload SHA-256;
  `should_reuse()` reuses identical images without recompute; any new
  image pops stale case + report.
- Checkbox toggles only switch display of the cached case (quick vs
  full workup is chosen at run time and recorded in the case).
- No user data leaves the session; no PHI fields exist anywhere.

## Safety gates

UNGRADABLE (and failed enhancement / grading errors) block inference
via `gate_routing()` — labeled `quality_gate_routing`, never presented
as a Phase 8 decision, carrying no fabricated grade. Blocked images get
recapture guidance + the Phase 9 ungradable report path. Stage failures
show messages, never tracebacks; valid prior stages stay visible.

## Report generation

`_report_input()` maps the orchestrator result onto the Phase 9
contract (no recompute); downloads generated on demand. JSON
authoritative. Explainability figure omitted from downloads by design
(available in-session); report notes it.

## Local run

`.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py`
then open the shown localhost URL. Requires `models/` weights (Phase 1)
and `venv` deps; no API keys (RAG/LLM removed from this UI).

## Known performance limitations

Full workup ≈ 40 s–4 min/image CPU (lesions + fovea + Grad-CAM
dominate); quick mode ≈ seconds. Native upload resolution is processed
(behavior preserved); display downscaled to 800px. TF logs to stderr
are normal. Verified: startup health HTTP 200; full real-image fixture
(2136×3216) end-to-end with zero stage errors.
