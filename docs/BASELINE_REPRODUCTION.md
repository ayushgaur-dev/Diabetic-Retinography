# Baseline Reproduction Log (SIH26038 — Phase 1)

Phase 1 only. No new SIH functionality was implemented. One additive test file
(`tests/test_model_smoke.py`, explicitly required by this phase) was created;
no existing source, algorithm, or threshold was modified.

## PROJECT

- Repository: `https://github.com/madatascienceml/project-ds-end-to-end-v2`
- Pinned commit (short): `fefba57`
- Pinned commit (full SHA): `fefba5750d6858265723458e27d927543f6ae7b2`
- Commit subject/date: "Update demo video link in README" / 2026-09-02
- Vendored into this workspace on 2026-09-03 (Phase 0 docs preserved untouched).
- Phase 0 audit: `docs/SIH_ARCHITECTURE_AUDIT.md`; plan: `docs/SIH_IMPLEMENTATION_PLAN.md`.

## ENVIRONMENT

- OS: Microsoft Windows 11 Home Single Language, 10.0.26200 (win32, CPU-only TF)
- Python: 3.13.6 (system and `venv/`)
- Env creation: `python -m venv venv` → `.\venv\Scripts\pip.exe install -r requirements.txt`
- Install result: all 14 pinned packages installed with **zero conflicts, zero version changes**.

## DEPENDENCIES (pip freeze, relevant)

`tensorflow==2.21.0`, `numpy==2.2.6`, `pandas==2.3.3`, `Pillow==12.3.0`,
`opencv-python-headless==5.0.0.93`, `matplotlib==3.10.9`, `scikit-learn==1.7.2`,
`sentence-transformers==5.7.0`, `faiss-cpu==1.15.0`, `kagglehub==1.0.2`,
`streamlit==1.60.0`, `openai==3.0.0`, `python-dotenv==1.2.2`, `pytest==9.1.1`,
plus `keras==3.15.1` (unpinned transitive dep via TF 2.21 — newer than the 3.12.x
referenced in `docs/experiments.md`; model still loads, see below).

## MODEL

- Architecture: `Sequential([EfficientNetB0 (ImageNet, avg-pool, last 20 layers unfrozen), ` +
  `Dropout(0.3), Dense(5, softmax)])` — 5 ICDRSS grades 0–4.
- Weights: `models/efficientnetb0_finetuned_patched.keras` (23,425,638 bytes, SHA256 in
  `docs/MODEL_ARTIFACTS.md`), from Hugging Face `manudaza/retinal-triage-efficientnetb0`.
- Loading: `keras_load_model(...)` succeeds under Keras 3.15.1 — the documented
  `quantization_config` fragility did NOT reproduce with the patched file here.
- Smoke test: `tests/test_model_smoke.py` (4 tests; skips if weights absent).

## PREPROCESSING (existing, verified consistent)

- Input: RGB fundus photo → `PIL.convert("RGB").resize((224, 224))` → `float32` →
  `efficientnet.preprocess_input` → batch dim. (`app/streamlit_app.py:101-111`,
  identical to training notebook `03[9]`.)
- NOT `/255.0` (that path belongs to the discarded baseline CNN, notebook `02[11]`).
- Training augmentation (notebooks only, not inference): horizontal flip, brightness ±0.1,
  `rot90` — the 90°/270° rotations are anatomically questionable; recorded for revisit,
  NOT changed in Phase 1.
- No CLAHE / illumination normalisation / denoising / masking exists (SIH Phases 2–3).

## INFERENCE

- Command: covered by `tests/test_model_smoke.py::test_inference_returns_valid_distribution`.
- Example (synthetic fundus-like fixture, seed 7 — plumbing only, no clinical meaning):
  probs `[0.9657, 0.0201, 0.0072, 0.0036, 0.0035]`, sum 1.0 → class `0 — No DR`,
  raw softmax max `0.9657`.
- **This number is RAW MODEL PROBABILITY, not calibrated confidence.** No temperature
  scaling / ECE / reliability evaluation exists. Known safety gap → SIH Phase 7.

## GRAD-CAM

- Command: `tests/test_model_smoke.py::test_gradcam_heatmap_and_overlay_valid`.
- Output (same fixture): heatmap `(7, 7)` from `top_activation`, finite, max 1.0,
  mean 0.18, nonzero everywhere; predicted-class index 0; overlay `(224, 224, 3)` uint8.
- Existing limitation preserved: attention concentrates on the optic disc across grades
  (notebook `03[45]`, `docs/experiments.md`) — documented, not redesigned.

## TRIAGE

- Command: `.\venv\Scripts\python.exe -m pytest tests/test_triage.py`
  and `test_triage_deterministic_on_model_output` (model grade → rules → identical dict twice).
- Example: grade 0 @ 0.9657 + neutral intake → `virtual_followup`, 12 months, no forced review.
- Rules are deterministic; the LLM never sets action/interval/grade (verified in
  `src/rag/generator.py:152-160` — authoritative fields copied verbatim).

## RAG

- Index: rebuilt via `.\venv\Scripts\python.exe -m src.rag.indexer` → 7 chunks,
  384-dim (`all-MiniLM-L6-v2`), `data/guidelines/index/` (gitignored). Retrieval verified:
  severe-NPDR interval query → `recommended_follow_up_intervals` at rank 1.
- API requirement: `OPENAI_API_KEY` env var only (never in code). No key is configured here.
- Degraded mode (verified, NO code change needed): without a key, `generate_report`
  returns `_fallback_report` — findings/guideline None + error string, rule-engine
  action/interval preserved, `requires_human_review` forced True. The app surfaces the
  rule decision with an error notice instead of crashing.

## STREAMLIT

- Launch: `.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.headless true`
- Status: boots; `/_stcore/health` → HTTP 200. Full click-through (upload→infer→report)
  not exercised here beyond the covered pipeline pieces (model/Grad-CAM/triage/RAG-fallback
  all verified individually); interactive run needs weights + index (present) + API key
  (absent → degraded report path, by design).

## TESTS

- Command: `.\venv\Scripts\python.exe -m pytest tests/`
- Result: **35 passed, 0 failed, 0 skipped, 0 errors** (31 pre-existing triage + 4 new smoke).
- No existing test or source file was modified.

## KNOWN LIMITATIONS (carried forward)

- QWK ≈ 0.7987 and all per-class recalls are VALIDATION-set results (n≈549) — must never
  be labelled test performance. Held-out test split exists but was never evaluated (Phase 5).
- Raw softmax max is displayed as "Confidence" — NOT calibrated (Phase 7).
- APTOS `train.csv` has no patient IDs → no group-aware split possible (inherited, documented).
- APTOS raw images (~10 GB), IDRiD, DRIVE, Messidor-2 not present (later phases).
- Model-loading `quantization_config` fragility documented; patched file loads here but any
  re-save with another Keras build must be load-tested.
- `rot90` augmentation includes anatomically implausible 90°/270° rotations (revisit later).
- RAG eval is 18 hand-labelled queries over a 7-chunk corpus; faithfulness review is 2 manual
  cases — demo-grade, not clinical validation.
- `api/` is an empty placeholder (FastAPI deliberately skipped upstream).
