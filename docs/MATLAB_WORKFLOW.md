# MATLAB Engineering Workflow (SIH26038 — Phase 11)

## Installation requirements

- MATLAB (any recent release; code targets R2020b-compatible syntax).
- Image Processing Toolbox (only additional toolbox).
- No Python, no datasets, no API keys for the core workflows.
- Optional: `APTOS_SAMPLE_IMAGE` env var pointing at a local non-PHI
  fundus image enables one extra test (`TestIntegration/realAptosExample`).

## Toolbox requirements (actually used)

| Toolbox | Used for |
|---|---|
| Image Processing Toolbox | imread/imresize/rgb2gray/rgb2lab/lab2rgb/adapthisteq/imbilatfilt/imgaussfilt/imopen/imclose/strel/bwconncomp/imtophat/regionprops-adjacent ops/bwboundaries/matshow/heatmap/readmatrix |
| Base MATLAB | jsondecode, plotting, matlab.unittest |

NOT used: Deep Learning Toolbox, Computer Vision Toolbox, Statistics
and Machine Learning Toolbox, Simulink (Phase 12), Coder/GPU Coder.

## Project structure

See `matlab/README.md`. Entry points: `startup.m`,
`tests/run_all_tests.m`, `examples/*.m`.

## How to run examples

```matlab
cd matlab
startup
example_quality_demo        % synthetic fundus -> quality + panels
example_screening_summary   % committed Phase 9 JSON -> console summary
```

## How JSON integration works

`load_screening_json()` parses a Phase 9 `report.json` (authoritative
import); `validate_report()` checks required sections; missing optional
evidence degrades to warnings. `run_engineering_workflow(image, json)`
combines MATLAB image processing with the imported result into one
struct and prints the screening summary.

## What MATLAB reproduces vs visualizes

- Reproduces (demonstration-grade): loading, FOV, focus/illumination/
  contrast/exposure assessment, CLAHE-L, illumination normalization,
  bilateral denoise, gray-world color, vessel top-hat response, quality
  aggregation — all reading the SHARED `configs/*.json` thresholds.
- Visualizes only: grades, probabilities, calibration, referable
  decisions, lesion maps, Grad-CAM, triage, all evaluation metrics.
- Known parity notes: MATLAB `var()` uses N−1 vs numpy N; OpenCV vs
  MATLAB morphology kernels differ at boundaries; Gaussian background
  substitutes Python's median background (documented in code). Numerical
  byte-equivalence is NOT claimed.

## Limitations

- Demonstration implementations, not production replacements; the
  Python pipeline remains authoritative for screening.
- Research/decision-support prototype — not a diagnostic device;
  qualified human review required; lesion outputs are evidence/
  candidates; calibrated confidence is uncalibrated and is not clinical
  certainty; ungradable images must not be auto-graded.
- MATLAB was unavailable in the authoring environment: `.m` files use
  conservative syntax and are covered by `matlab.unittest` tests (run
  where MATLAB exists) plus Python structural checks
  (`tests/test_matlab_integration.py`).
- APTOS/IDRiD results do not establish rural deployment validity.

## Relationship to Phase 12 Simulink

`screening_struct()` is the plain-struct contract Phase 12 consumes:
telemedicine blocks will carry the same fields (quality, screening,
anatomy, evidence, triage) without MATLAB/Python duplication.
