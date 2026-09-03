# SIH26038 — Implementation Plan (PHASE 0)

**Date:** 2026-09-03 · **Status:** PLAN ONLY — no code written, no refactoring done.
Companion document: `SIH_ARCHITECTURE_AUDIT.md` (evidence for every claim below).

Working agreement for all phases: the system is a **screening/decision-support prototype,
not a medical device**. Sensitivity >90% / specificity >85% are TARGETS, never claims.
No metric is reported unless produced by a reproducible experiment. Final decisions stay
with qualified clinicians.

---

## 1. Requirement traceability matrix

Each row: **SIH requirement → existing component → missing piece → planned implementation →
dataset → evaluation metric.**

### A. Image quality assessment (SRS: GOOD / BORDERLINE / UNGRADABLE + recapture gate)

| | |
|---|---|
| Existing | EDA luminance stats + guideline text only. No code. |
| Missing | Everything: metrics, thresholds, classifier, gate. |
| Plan | New `src/quality/`: `metrics.py` (Laplacian-variance focus, luminance/exposure, contrast RMS, FOV/retinal-coverage mask fraction, saturation), `assess.py` (metric→GOOD/BORDERLINE/UNGRADABLE via `configs/quality_thresholds.yaml`), gate in pipeline: UNGRADABLE blocks inference + returns recapture instructions; BORDERLINE routes to enhancement then reassessment. Tests on synthetic degradations (blur, darken, crop) + manual review slice of APTOS. |
| Dataset | APTOS 2019 (unsupervised threshold-setting on train split; manual quality labels on a small audit slice only). |
| Metrics | GATE behaviour (ungradable-block rate), agreement of auto-labels vs manual slice (accuracy, Cohen's κ), downstream: % BORDERLINE rescued by enhancement. |

### B. Adaptive enhancement (CLAHE, illumination norm, denoise, colour norm, masking)

| | |
|---|---|
| Existing | Resize + `preprocess_input` only. |
| Missing | All enhancement; versioning. |
| Plan | New `src/preprocessing/`: `enhance.py` (retinal-mask estimation → masked CLAHE on L-channel, background-subtraction illumination normalisation, mild bilateral/non-local denoise, colour normalisation), `adaptive.py` (apply per quality profile: GOOD→minimal, BORDERLINE→full, then reassess; never enhance UNGRADABLE into gradability). `PREPROCESSING_VERSION` constant recorded in every report. Ablation: grading performance with/without enhancement on BORDERLINE slice. |
| Dataset | APTOS 2019 (same splits). |
| Metrics | Quality-metric deltas pre/post, 5-class QWK + per-class recall delta, referable-DR sensitivity/specificity delta on the BORDERLINE slice. |

### C. Retinal structures (optic disc, fovea/macula, vessels)

| | |
|---|---|
| Existing | None in code. (Grad-CAM happens to highlight disc/vessels — model attention, not segmentation.) |
| Missing | All detectors. |
| Plan | New `src/retina/`: `vessels/` (classical Frangi/matched-filter baseline first; supervised U-Net on DRIVE only if baseline insufficient), `optic_disc/` (brightness+shape localisation baseline), `fovea/` (disc-relative geometry prior + dark-region search). Outputs are geometric evidence (masks, centres, confidence) consumed by evidence engine + report overlays — NOT inputs to the grader. |
| Dataset | DRIVE (vessels); IDRiD/APTOS-derived manual points for disc/fovea sanity checks (small, documented). |
| Metrics | Vessels: Dice/AUC on DRIVE test; disc/fovea: Euclidean error (disc-diameters) + detection rate within tolerance. |

### D. Lesion evidence (microaneurysm / exudate / haemorrhage / neovascularisation candidates)

| | |
|---|---|
| Existing | None in code (guideline text describes the lesion types only). |
| Missing | All extractors. |
| Plan | New `src/retina/lesions/`: classical candidate extractors (red-lesion: green-channel morphology; exudate: bright-lesion thresholding outside disc; haemorrhage size-split; NV: vessel-density/tortuosity anomaly flag). Label outputs explicitly `*_candidate` with confidence; NEVER present as confirmed diagnoses. Lesion maps overlay with anatomy for the report. |
| Dataset | IDRiD (lesion annotations for evaluation); APTOS for qualitative grading-correlation review. |
| Metrics | Candidate-level: per-lesion-type precision/recall/FROC-style curves on IDRiD; system-level: correlation of candidate burden with predicted grade (audited, not trained). |

### E. DR grading — 5-class + referable-DR binary (grade ≥ 2)

| | |
|---|---|
| Existing | EfficientNetB0 baseline (QWK 0.7987 val-only), stratified 70/15/15, class weights, tradeoff CSV. |
| Missing | HELD-OUT TEST evaluation, full metric suite, binary referable task, config-driven thresholds, seed/model-version discipline, external validation. |
| Plan | `src/grading/` (promote notebook code to module: `dataset.py`, `model.py`, `train.py`, `evaluate.py`): (1) evaluate frozen baseline on the HELD-OUT test split first, untouched; (2) add thresholds/config; (3) compute full suite — accuracy, precision/recall/F1, per-class, confusion, Cohen's κ, QWK, ROC-AUC/PR-AUC (one-vs-rest) for 5-class; sensitivity, specificity, precision, NPV, F1, ROC-AUC, PR-AUC for referable-DR; (4) Messidor-2 external validation slice; (5) record MODEL_VERSION + seeds. Enhancement/quality ablations reuse this harness. |
| Dataset | APTOS 2019 primary (train/val/test as built); Messidor-2 external validation. |
| Metrics | As above. Targets (not claims): referable sensitivity >90%, specificity >85%. |

### F. Explainability (Grad-CAM + lesion/anatomy evidence + annotated report)

| | |
|---|---|
| Existing | Real top-class Grad-CAM + overlay; honest disc-attention finding. |
| Missing | Class-selectable maps, NaN guard, lesion/anatomy overlays, model-vs-clinical-evidence separation in UI/report. |
| Plan | Extend `src/explainability/`: class-specific Grad-CAM (target-grade argument), epsilon guard, lesion-candidate + vessel + disc/fovea overlay composer, side-by-side "MODEL EVIDENCE vs CLINICAL EVIDENCE" figure for the report/UI with fixed caption semantics (Grad-CAM = contribution to prediction, not causality). |
| Dataset | APTOS test slice (qualitative audit) + IDRiD (lesion-map plausibility). |
| Metrics | Qualitative audit protocol (fixed checklist, N cases/grade); sanity metrics (heatmap mass inside retinal mask; overlap correlation with lesion candidates — reported as observations, not proof). |

### G. Confidence calibration (temperature scaling; never raw softmax)

| | |
|---|---|
| Existing | None — raw softmax max shown as "Confidence". |
| Missing | Calibrator, ECE/reliability/Brier eval, val/calibration vs test discipline. |
| Plan | New `src/calibration/`: temperature scaling fit on VALIDATION (or dedicated cal split — never test), `calibrate.py` + `metrics.py` (ECE, MCE, reliability diagrams, Brier). Calibrated confidence feeds rule engine + report; raw softmax retained alongside for audit. Revisit 0.60/0.75 thresholds post-calibration. |
| Dataset | APTOS val (fit) → test (evaluate). |
| Metrics | ECE (primary), MCE, Brier score, reliability diagrams pre/post; decision-impact: abstention/flag rates pre/post. |

### H. Evidence / triage engine (deterministic; AI predicts, rules interpret)

| | |
|---|---|
| Existing | Pure-function engine + ~20 tests (grade + raw confidence + 3 patient modifiers). |
| Missing | Quality, lesion, vessel, anatomy, calibrated-confidence inputs; referable-DR output; version fields; config-file thresholds. |
| Plan | Extend `src/evidence/` (rules live here going forward): quality gate (UNGRADABLE→recapture, no inference), calibrated-confidence abstention, intermediate-grade safety net (re-tuned post-calibration), lesion/vessel/anatomy evidence notes (flag-only, never auto-escalate grade), referable-DR derivation (grade ≥ 2), `RULE_VERSION` + all input versions in output. Keep model/rules separation; extend `tests/` per new branch; move thresholds to `configs/triage_rules.yaml`. |
| Dataset | APTOS test (decision audit); synthetic edge-case suite (unit level). |
| Metrics | Automation rate, abstention/recapture/referral rates, severe-case review coverage, rule-branch unit coverage (100% of branches). |

### I. Structured report (works WITHOUT any LLM/API)

| | |
|---|---|
| Existing | LLM-drafted findings + guideline citations with guardrails; fail-safe fallback. |
| Missing | Offline template renderer; quality/lesion/anatomy/calibration/uncertainty/version/limitation sections. |
| Plan | New `src/reporting/`: Jinja-/template-based renderer producing the FULL report offline (quality, grade, referable status, calibrated confidence + uncertainty, Grad-CAM + lesion/anatomy figures, recommendation, versions, limitations, clinician-review requirement). LLM kept as OPTIONAL phrasing layer over locked fields only (same guardrails as `generator.py`); pipeline never depends on it. Report export (PDF/HTML) for the demo. |
| Dataset | N/A (deterministic rendering; snapshot tests). |
| Metrics | Schema-completeness checklist (all 13 SIH fields present), snapshot tests, LLM-optional parity check (locked fields identical with/without LLM). |

### J. MATLAB image-processing pipeline

| | |
|---|---|
| Existing | Nothing. |
| Missing | All `.m` files. |
| Plan | New `matlab/`: `quality/` (FOV detection, focus/illumination/contrast metrics), `preprocessing/` (CLAHE, illumination normalisation, denoising), `retina/` (vessel extraction, disc localisation), `visualization/` (side-by-side figure exporter), `evaluation/` (metric parity vs Python on a fixed sample). No DL duplication — MATLAB covers classical image processing only, with a parity table vs `src/quality` + `src/preprocessing`. Requires MATLAB + Image Processing Toolbox (document version). |
| Dataset | Fixed 50-image APTOS sample (IDs committed) for parity. |
| Metrics | Parity: per-metric correlation/MAD vs Python implementation; qualitative figure review. |

### K. Simulink rural-telemedicine simulation (100,000+ patients/year)

| | |
|---|---|
| Existing | Nothing. |
| Missing | All `.slx` + parameter scripts. |
| Plan | New `matlab/simulink/`: discrete-event Patient-Arrival→Acquisition→Upload→Network→Quality→AI→Review-Queue→Ophthalmologist→Referral chain (SimEvents if available, else scripted queue simulation with identical parameters); `params.m` (annual/daily patients, image size, bandwidth, latencies, AI time, rejection rate, review time, ophthalmologist count, working hours, referral rate); outputs (throughput, waits, queues, utilisations, screened/recapture/referral counts). Scenario script sweeps to 100k+/year. |
| Dataset | N/A (parameter-driven; rejection/referral rates fed from measured pipeline stats where available, else labelled assumptions). |
| Metrics | Throughput, mean/max wait, queue lengths, clinician/AI/bandwidth utilisation; sensitivity sweeps over bandwidth, staffing, rejection rate. |

---

## 2. Phase order (sequential; one phase at a time)

| Phase | Goal | Entry gate | Exit gate |
|---|---|---|---|
| 0 | Audit (THIS DOC) | — | Audit + plan committed; next step approved |
| 1 | Base reproducibly runs | Base commit pinned + vendored | `pytest` passes; app boots; weights+index build docs verified; repro log committed |
| 2 | Quality assessment | Phase 1 green | Metrics + thresholds config + gate tests + manual-slice agreement |
| 3 | Adaptive enhancement | Quality metrics frozen | Enhancement module + version + BORDERLINE rescue numbers + ablation |
| 4A–4D | Vessels → disc → fovea → lesions | Prior structure module done | Per-module metrics on DRIVE/IDRiD + overlays |
| 5 | Rigorous grading eval | Preprocessing frozen | FULL suite on held-out test + referable metrics + Messidor-2 slice; NO tuning on test |
| 6 | Integrated explainability | Grading frozen | Class-specific Grad-CAM + evidence overlays + audit checklist |
| 7 | Calibration | Grading frozen | Temperature + ECE/Brier/reliability on test; thresholds revisited |
| 8 | Evidence/triage engine | Calibrated confidence available | Extended rules + versioned outputs + full branch tests |
| 9 | Automated offline report | Evidence schema frozen | Template report with all 13 fields, no API required |
| 10 | Clinician UI | Report schema frozen | Updated Streamlit app (quality, evidence, calibration, report export) |
| 11 | MATLAB pipeline | Python quality/preproc frozen | `.m` modules + parity table on fixed sample |
| 12 | Simulink simulation | Measured rates available | `.slx` + 100k/year scenario results |
| FINAL | Integration, benchmarks, ablations, docs, demo | All phases green | End-to-end demo + ablation table + docs + safety statement |

Phase 5 ordering note: grading evaluation runs AFTER enhancement (3) and structures/lesions
(4A–4D) are available as evidence-only overlays, but the grader itself is evaluated with the
frozen pipeline — evidence modules must never leak into grading features without an explicit
new experiment + version bump.

## 3. Target architecture (adapted to base repo, not enforced blindly)

```
project/
  app/                 # existing Streamlit app (extend in Phase 10)
  src/
    quality/           # NEW (Phase 2)
    preprocessing/     # NEW (Phase 3; promote notebook 03[9] here)
    retina/
      vessels/         # NEW (Phase 4A)
      optic_disc/      # NEW (Phase 4B)
      fovea/           # NEW (Phase 4C)
      lesions/         # NEW (Phase 4D)
    grading/           # NEW module wrapper (Phase 5; model code promoted from notebooks)
    explainability/    # PROMOTE src/models/gradcam.py (Phase 6)
    calibration/       # NEW (Phase 7)
    evidence/          # PROMOTE src/rules/triage.py (Phase 8)
    reporting/         # NEW offline renderer (Phase 9; rag/generator.py becomes optional layer)
    evaluation/        # NEW shared metric harnesses (Phase 5; rag eval stays beside it)
    rag/               # KEEP as-is (indexer, generator, eval)
    data/              # KEEP synthetic_intake.py
  configs/             # NEW: quality_thresholds.yaml, triage_rules.yaml, calibration.yaml
  matlab/              # NEW (Phase 11: quality, preprocessing, retina, visualization, evaluation, simulink/)
  tests/               # EXTEND per phase (keep test_triage.py green)
  reports/ docs/       # EXTEND (repro logs, parity tables, scenario results)
```

Rules: reuse working components (triage, gradcam, RAG, synthetic intake); promote notebook-only
code into `src/` with versions instead of duplicating; never rewrite for cosmetics.

## 4. Dataset strategy (per-strength, no blind pooling)

- **APTOS 2019** → primary DR grading + quality-threshold setting (train) + calibration fit (val) +
  final evaluation (held-out test). Never tune on test.
- **IDRiD** → lesion-candidate evaluation only (Phase 4D). Not pooled into grading training.
- **DRIVE** → vessel-module evaluation only (Phase 4A). Not pooled into grading training.
- **Messidor-2** → external validation of frozen grader (Phase 5). No training, no tuning.
- Splits: keep base 70/15/15 seed-42 for continuity; document no-patient-ID leakage limitation
  (inherited); use group-aware splits wherever IDs exist (DRIVE/IDRiD/Messidor-2).
- Raw images + weights stay out of git (Kaggle/HF provenance + hashes in repro log).

## 5. Cross-cutting engineering rules (binding all phases)

1. Inspect before modifying; one phase at a time; never silently skip requirements.
2. Thresholds/parameters in `configs/` — no magic constants in code.
3. `MODEL_VERSION`, `PREPROCESSING_VERSION`, `RULE_VERSION` recorded in every report/eval row.
4. Fixed seeds where appropriate; train/val/calibration/test separation; no test tuning.
5. Deterministic rules for safety-critical interpretation; AI predicts, rules interpret.
6. No API keys in code (env only); no unnecessary dependencies (justify TF/torch-class additions).
7. Tests for every new module; repro log per experiment; conservative medical claims only;
   candidate/evidence language for lesions; Grad-CAM labelled as model evidence, not causality.

## 6. Recommended next step (Phase 1, awaiting instruction)

1. Pin base commit (`fefba57` or newer HEAD — record hash) and vendor it into this workspace
   via `git init + git subtree` (or fork-clone if network/credentials prefer).
2. Create venv, install `requirements.txt`, run `pytest tests/`, boot Streamlit with
   documented degraded paths (no weights / no index / no API key), download weights from
   Hugging Face, build FAISS index, and record a Phase-1 repro log
   (`docs/PHASE1_REPRO.md`: env, hashes, what runs, what is degraded).
3. Stop and report before Phase 2.

**Do NOT begin Phase 1 until instructed.** Phase 0 deliverables are complete with this file.
