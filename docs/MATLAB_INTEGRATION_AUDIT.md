# MATLAB Integration Audit (SIH26038 — Phase 11)

Date: 2026-09-04. No Python behavior changed in this audit.

## 1. SIH MATLAB requirements (from master context §J–K + Phase 11 brief)

- MATLAB image-processing pipeline: loading, FOV detection, focus /
  illumination / contrast assessment, CLAHE, illumination normalization,
  denoising, vessel extraction, optic-disc localization, visualization.
- Simulink rural-telemedicine workflow simulation (deferred to Phase 12;
  this phase prepares clean struct interfaces for it).
- MATLAB must consume (not recompute) Python screening results.
- No duplication of the EfficientNetB0 model in MATLAB.

## 2. What already exists in Python (sources of truth)

| Capability | Python module | MATLAB role |
|---|---|---|
| Quality GOOD/BORDERLINE/UNGRADABLE + thresholds | `src/quality/`, `configs/quality_thresholds.json` | Demonstration re-implementation reading the SAME JSON thresholds |
| Enhancement (CLAHE-L, illum-norm, bilateral, gray-world) | `src/preprocessing/`, `configs/enhancement_config.json` | Native equivalents (adapthisteq, imflatfield approx, imbilatfilt) |
| Vessels/disc/fovea/lesions | `src/retina/*` | Visualization + lightweight response demos, not replacements |
| Grading/calibration/triage/report | Phases 5/7/8/9 + JSON artifacts | Import-only via Phase 9 JSON; never recomputed |
| Evaluation metrics | `reports/*/metrics.json`, CSVs | Visualization-only (confusion, QWK, ROC/PR, reliability) |

## 3. What MATLAB implements (this phase)

`matlab/`: config loading (JSON), preprocessing, quality demo,
vessel-response + disc/fovea demos, screening-summary from imported
JSON, evaluation-figure scripts, workflow struct pipeline, matlab.unittest
tests, examples. All demonstration-grade and labelled as such.

## 4. What MATLAB only visualizes/integrates

Grades, probabilities, calibration, referable decisions, lesion maps,
Grad-CAM, triage, reports, and all evaluation metrics are READ from
`reports/` artifacts. MATLAB never refits, retrains, or re-decides.

## 5. What is NOT duplicated

EfficientNetB0 (no Deep Learning Toolbox inference of the grader),
temperature fitting, triage rules, lesion detectors, Streamlit/Next.js UI.

## 6. Required toolboxes (actually used — see §9 of workflow doc)

- Image Processing Toolbox: imread/imresize/rgb2gray/adapthisteq/
  imbilatfilt/fspecial/imopen/imclose/bwconncomp/regionprops/matshow.
- Base MATLAB only otherwise: jsondecode, plotting, matlab.unittest.
- NOT required: Deep Learning, Computer Vision, Statistics/ML, Simulink
  (Phase 12), Coder/GPU Coder.

## 7. Integration limitations

- Numerical parity with Python is NOT claimed (OpenCV vs MATLAB kernels,
  uint8 rounding, morphology element differences documented per function).
- MATLAB quality thresholds are READ from the same JSON files (shared,
  not forked); any future Python threshold change propagates automatically.
- No MATLAB installation was available in this environment: .m files are
  written in conservative (R2020b-compatible) syntax and covered by
  matlab.unittest tests for execution where MATLAB exists, plus Python
  structural checks (file presence, fixture validity) in this repo's suite.
- Simulink interfaces are prepared as plain structs (Phase 12 consumes them).
