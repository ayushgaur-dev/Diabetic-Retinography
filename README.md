# SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India

Research/decision-support prototype. **Not a diagnostic device.** Results
require qualified human review. Built on
[madatascienceml/project-ds-end-to-end-v2](https://github.com/madatascienceml/project-ds-end-to-end-v2)
(itself a teaching fork); the grading model is distributed separately
under CC-BY-NC-4.0 (see Model weights below), code here under MIT.

## Pipeline

Fundus image → quality gate (GOOD/BORDERLINE/UNGRADABLE) → adaptive
enhancement → EfficientNetB0 5-class grading → temperature calibration →
vessel/disc/fovea analysis → lesion evidence → Grad-CAM explainability →
deterministic triage (UNGRADABLE/ROUTINE/REFER/URGENT_REVIEW/
TECHNICAL_REVIEW) → automated report (JSON/Markdown/HTML/PDF).

Held-out APTOS results (engineering, not clinical validation): QWK
0.811, referable sensitivity 96.0%, specificity 87.2%, ROC-AUC 0.974.
Full evidence in `reports/`; system map in
`docs/FINAL_SYSTEM_ARCHITECTURE.md`.

## Install (Windows, Python 3.13)

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
.\venv\Scripts\pip.exe install -r requirements.txt        # screening + Streamlit
.\venv\Scripts\pip.exe install -r backend/requirements.txt  # FastAPI backend
```

Model weights (gitignored, ~23 MB, Hugging Face):
`manudaza/retinal-triage-efficientnetb0` → `models/efficientnetb0_finetuned_patched.keras`
(see `docs/MODEL_ARTIFACTS.md`). RAG/LLM features additionally need
`OPENAI_API_KEY`; all screening paths work without it.

## Run

- Streamlit reference UI:
  `.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py`
- FastAPI backend:
  `.\venv\Scripts\python.exe -m uvicorn backend.api:app --port 8001`
- Next.js frontend (`frontend/`, needs `npm.cmd install` once):
  dev `node node_modules/next/dist/bin/next dev`, prod `.../next build` + `.../next start -p 3001`
- MATLAB (`matlab/`, needs MATLAB + Image Processing Toolbox):
  `startup; run_all_tests` — Simulink scenarios: `run_all_scenarios()`
- Tests: `.\venv\Scripts\python.exe -m pytest tests/ -q`

## Docs index

`docs/SIH_REQUIREMENTS_MATRIX.md` (requirement coverage),
`docs/FINAL_VALIDATION_REPORT.md` (system status),
`docs/DEMO_SCRIPT.md` (5–7 min demo), `docs/PERFORMANCE.md`,
`docs/API_CONTRACT.md`, `docs/FRONTEND.md`, `docs/STREAMLIT_UI.md`,
`docs/TRIAGE_ENGINE.md`, `docs/AUTOMATED_SCREENING_REPORT.md`,
`docs/DR_GRADING_EVALUATION.md`, `docs/CALIBRATION.md`,
`docs/EXPLAINABILITY.md`, `docs/VESSEL_SEGMENTATION.md`,
`docs/OPTIC_DISC_LOCALIZATION.md`, `docs/FOVEA_LOCALIZATION.md`,
`docs/LESION_EVIDENCE.md`, `docs/IMAGE_QUALITY_ASSESSMENT.md`,
`docs/IMAGE_ENHANCEMENT.md`, `docs/MATLAB_WORKFLOW.md`,
`docs/SIMULINK_WORKFLOW.md`, `docs/FINAL_SYSTEM_ARCHITECTURE.md`.
