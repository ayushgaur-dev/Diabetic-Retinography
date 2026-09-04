# SIH Requirements Matrix (SIH26038 — Phase 13)

Status scale: IMPLEMENTED / PARTIALLY IMPLEMENTED / DEMONSTRATION ONLY /
NOT VALIDATED. Nothing is marked IMPLEMENTED on documentation alone.

| # | Requirement | Status | Implementation | Evidence / test | Limitation |
|---|---|---|---|---|---|
| 1 | Image quality (focus/illum/contrast/exposure/FOV, G/B/U) | IMPLEMENTED | `src/quality/`, `configs/quality_thresholds.json` | `tests/test_quality.py` (22); APTOS stratification | Focus bar too strict on real data (measured, not retuned) |
| 2 | Adaptive enhancement | IMPLEMENTED | `src/preprocessing/`, `configs/enhancement_config.json` | `tests/test_enhancement.py` (22); ablation: no grading gain | Does not rescue most APTOS BORDERLINE; never claimed to |
| 3 | Retinal anatomy (vessels/disc/fovea) | IMPLEMENTED | `src/retina/*`, configs | vessel F1 0.71 (DRIVE); disc 98% @0.25DD (Drishti); fovea med err 0.033 (IDRiD) | Research datasets; classical baselines |
| 4 | Lesion evidence (MA/HE/EX/SE) | PARTIALLY IMPLEMENTED | `src/retina/lesions/` | EX F1 0.23, HE 0.11, MA 0.08, SE 0.00 (IDRiD test) | SE non-functional; classical precision limits |
| 5 | 5-class DR grading | IMPLEMENTED | Frozen EfficientNetB0 | Held-out QWK 0.811, acc 0.68 | Minority grades weak (R 0.11–0.66) |
| 6 | Referable DR (≥2) | IMPLEMENTED | Calibrated P2+P3+P4 @ frozen 0.7 | Sens 96.0%, spec 87.2%, AUROC 0.974 + CIs | In-dataset; 5.6% twin contamination (bounded) |
| 7 | Sensitivity/specificity evidence | IMPLEMENTED | `reports/grading/` | Above + bootstrap CIs + dedup rerun | Targets met as engineering results, not clinical claims |
| 8 | Calibration | IMPLEMENTED | `src/calibration/`, T=0.9542 | ECE 0.047→0.044, NLL/Brier, reliability figs | Small effect; MCE(20) worsened (reported) |
| 9 | Grad-CAM | IMPLEMENTED | `src/models/gradcam.py` + adapter | Real heatmaps; optic-disc attention finding | Attribution, not causation |
| 10 | Explainability | IMPLEMENTED | `src/explainability/` | 10-image subset: 9 SUPPORTIVE; faithfulness drops | Descriptive stats, ~40 s/image |
| 11 | Triage | IMPLEMENTED | `src/triage/` | 550-image eval: 0 referable-FNs to ROUTINE; 34 unit tests | Low automation (64% technical review); heuristic thresholds |
| 12 | Automated report | IMPLEMENTED | `src/reporting/` (JSON/MD/HTML/PDF) | 30 tests; byte-identical determinism | Template renderer; LLM explicitly excluded |
| 13 | Rural/telemedicine workflow | DEMONSTRATION ONLY | MATLAB DES + Simulink builder, 10k–150k scenarios | `matlab/tests/` + structural checks | No MATLAB runtime here; unexecuted; assumptions, not measurements |
| 14 | MATLAB | IMPLEMENTED | `matlab/` preprocessing/quality/viz/integration | matlab.unittest suite (runs where MATLAB exists) | Demo-grade ports; parity notes documented |
| 15 | Simulink | DEMONSTRATION ONLY | Structural model builder + DES engine | Contracts tested; execution requires Simulink | No execution outputs included |
| 16 | Frontend | IMPLEMENTED | Next.js + FastAPI + Streamlit reference | Build+tsc clean; 10 API tests; E2E demo verified | Local/demo scope; no auth; ~50 s–4 min screenings |
| 17 | Privacy/safety | IMPLEMENTED | No PHI fields; hash-keyed sessions; safety gates A–H tested | `tests/test_safety_gates.py`, privacy grep clean | Demo deployment assumptions documented |
