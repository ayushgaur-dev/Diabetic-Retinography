# Final Validation Report (SIH26038 — Phase 13)

## Architecture

Orchestrated pipeline (`app/screening_pipeline.py`) over frozen Phase
2–9 modules; Streamlit reference + FastAPI/Next.js presentation;
MATLAB import layer; Simulink DES specification. Sources of truth:
Python pipeline (AI), Phase 8 (triage), Phase 9 (reports).

## Model artifact

`efficientnetb0_finetuned_patched.keras` (23,425,638 bytes, SHA256 in
`docs/MODEL_ARTIFACTS.md`), CC-BY-NC-4.0, frozen throughout.

## Evaluation dataset and split

APTOS 2019 mirror, notebook 70/15/15 stratified seed 42 → 2563/549/550.
ID overlap clean; 123 byte-duplicate groups (55 cross-split, 30 mixed
grades); 5.6% test contamination bounded by dedup rerun. No patient IDs.

## Test metrics (held-out, n=550)

Accuracy 0.68, macro F1 0.468, QWK 0.811 (CI 0.774–0.844); per-class
recall 0.98/0.11/0.43/0.66/0.46. TEST RESULT (in-dataset).

## Referable metrics

Sens 96.0% (CI 0.93–0.98), spec 87.2% (CI 0.84–0.91), AUROC 0.974,
threshold frozen 0.7. TEST RESULT — targets met as engineering results.

## Calibration metrics

T=0.9542 (val-fit); test ECE 0.047→0.044, Brier tie, NLL marginal gain;
MCE(20) worsened (reported). TEST RESULT.

## Quality limitations

Gate miscalibrated on real data (541/550 fail focus); enhancement
rescues little and degrades grading net (−41 vs +4). DEMO/TEST RESULT.

## Lesion limitations

Classical baseline: EX F1 0.23, others ≤ 0.11, SE 0.00. DEMO RESULT.

## Runtime

≈ 89 s/image CPU full workup (evidence-dominated); reporting < 0.1 s.
CPU prototype figures, not deployment performance.

## Frontend/API status

Build + tsc clean; 10 API contract tests green; live E2E (demo job +
4 report formats) verified; both pages serve 200 with full content.

## MATLAB/Simulink status

Structural contracts validated (16 Python checks); `.m` suites written
for MATLAB execution. Runtime execution UNAVAILABLE here — no outputs
pretended. DEMONSTRATION ONLY.

## Known limitations / unresolved risks

Heuristic thresholds throughout; thin tuning sets; single-region
research datasets; low triage automation by design trade-off; 40–90 s
screenings; no clinical validation of any kind; staffing-side simulation
values are assumptions.

## Distinctions

TEST RESULT: Phases 5/7 metrics. DEMO RESULT: fixtures, subsets, smoke
runs. ENGINEERING SIMULATION: Phase 12 code, unexecuted. CLINICAL
CLAIM: none made anywhere (claim audit clean).
