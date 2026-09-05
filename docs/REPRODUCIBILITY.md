# Reproducibility (SIH26038 — Phase 14)

## Python

- Version: 3.13.6 (Windows 11). `python -m venv venv`
- Screening + Streamlit: `pip install -r requirements.txt`
- Backend: `pip install -r backend/requirements.txt`
- Model artifact: `models/efficientnetb0_finetuned_patched.keras`
  (Hugging Face `manudaza/retinal-triage-efficientnetb0`, SHA256 in
  `docs/MODEL_ARTIFACTS.md`; gitignored, download per README).

## Datasets (all gitignored, cached outside repo)

- APTOS mirror (`mariaherrerot/aptos2019`, 8 GB): grading eval.
- DRIVE mirror (28 MB): vessels. Drishti-GS mirror (341 MB): disc.
- IDRiD Localization (203 MB Zenodo): fovea. IDRiD Segmentation
  (557 MB Zenodo): lesions. Env vars: `APTOS_DATA_ROOT`,
  `DRIVE_DATA_ROOT`, `DRISHTI_DATA_ROOT`, `IDRID_DATA_ROOT`,
  `IDRID_SEG_DATA_ROOT`.

## Commands

- Tests: `.\venv\Scripts\python.exe -m pytest tests/ -q` (343 green).
- Streamlit: `... -m streamlit run app/streamlit_app.py`.
- Backend: `... -m uvicorn backend.api:app --port 8001`.
- Frontend: `cd frontend; npm.cmd install`;
  dev `node node_modules/next/dist/bin/next dev`;
  prod `.../next build` + `.../next start -p 3001`
  (use `npm.cmd`/`node` directly — `.ps1` blocked by policy).
- MATLAB (`matlab/`, needs MATLAB + Image Processing Toolbox):
  `startup; run_all_tests`; scenarios `run_all_scenarios()`;
  Simulink model `build_simulink_model(...)` (needs Simulink).
- Simulink: NOT executable here — structural contracts only.

## Known unavailable runtimes

MATLAB/Simulink (no installation in this environment). Everything else
runs offline with no API keys.
