# MATLAB Screening Integration (SIH26038 — Phase 11)

Engineering/integration layer over the frozen Python pipeline.
Demonstration-grade MATLAB equivalents + authoritative JSON import.
See `docs/MATLAB_WORKFLOW.md` and `docs/MATLAB_INTEGRATION_AUDIT.md`.

## Layout

- `startup.m` — adds `src/*` to the path.
- `config/` — JSON loader (thresholds are READ from `configs/*.json`, never forked).
- `src/preprocessing/` — loading, FOV, illumination/contrast/color work.
- `src/quality/` — GOOD/BORDERLINE/UNGRADABLE demonstration assessment.
- `src/visualization/` — fundus panels, confusion, reliability, ROC figures.
- `src/evaluation/` — figure scripts over `reports/` artifacts (import-only).
- `src/workflow/` — struct-based engineering pipeline (Phase 12 interface).
- `src/integration/` — Phase 9 JSON parsing/validation + console summary.
- `tests/` — `matlab.unittest` suite (`run_all_tests.m`).
- `examples/` — runnable demos (synthetic + real artifacts, no PHI).
- `outputs/` — generated figures (gitignored).

## Run

```matlab
cd matlab
startup
run_all_tests            % full matlab.unittest suite
example_quality_demo     % synthetic quality walkthrough
example_screening_summary % Phase 9 JSON -> console summary + figure
```

Requires: MATLAB + Image Processing Toolbox (only). No Python needed.
