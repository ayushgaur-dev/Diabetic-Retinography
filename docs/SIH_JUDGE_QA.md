# Judge Q&A (SIH26038 — Phase 14, concise honest answers)

## Model

- **Why EfficientNetB0?** Best accuracy/parameter trade-off for a CPU
  prototype; transfer learning from ImageNet beats the from-scratch
  baseline (QWK 0.50→0.81).
- **Why not train from scratch?** 3,662 images cannot train a deep CNN
  reliably (baseline proved it); fine-tuning reuses visual priors.
- **Why five classes?** International clinical DR scale maps directly to
  follow-up intervals; collapsed to binary referable for screening.
- **Why APTOS?** Indian rural screening population, public, DR-graded;
  precisely our target domain demographically.
- **Test split?** Notebook 70/15/15 stratified seed-42 reconstruction;
  test_df built but never evaluated upstream — genuinely held out here.
- **Leakage?** ID overlaps empty; 123 byte-duplicate groups found, 31/550
  test images affected; dedup rerun barely moves metrics (QWK 0.81→0.82).
  No patient IDs exist — image-level only, stated.
- **5.6% contamination?** Test images with byte-twins outside test (mostly
  train). Bounded, disclosed, sensitivity-analyzed — not hidden.
- **Weak Grade 1/2 recall?** Adjacent-grade visual ambiguity (0.11/0.43);
  reported per-class, never averaged away. Referable task absorbs it.

## Quality

- **Ungradable?** Measured focus/illumination/contrast/exposure/FOV vs
  configured thresholds; any BAD → UNGRADABLE, grading never called.
- **Why not enhance everything?** Enhancement amplifies noise and degrades
  grading net (−41 vs +4 on APTOS); conditional application only.
- **False evidence?** Possible — hence acceptance testing + discard path.

## Explainability

- **Grad-CAM meaning?** Regions influencing this prediction (top conv
  layer gradients). Real implementation, predicted-class targeted.
- **Proof of disease?** No — attribution aid; stated in UI, reports, docs.
- **Lesion vs diagnosis?** Candidates from heuristics; never "confirmed".
- **SUPPORTIVE?** Spatial agreement between heatmap and evidence only.

## Calibration

- **Why temperature scaling?** Single-parameter, interpretable, preserves
  argmax; fitted on validation NLL, test untouched.
- **Calibrated confidence?** Better-aligned probability representation,
  not clinical certainty. Effect here is small (ECE 0.047→0.044) — honest.
- **Grade unchanged?** Dividing logits by T>0 preserves ordering —
  verified identical on all 550 test images.

## Triage

- **LLM override?** No LLM exists in the path; rules are pure functions.
- **Evidence override grade?** Never — conflicts route to human review.
- **Ungradable?** Blocked with recapture routing, no grade fabricated.
- **Why human review?** 64% technical-review rate by design trade-off:
  safe (0 referable-FNs) over automated.

## Rural

- **How in rural India?** Camera + lightweight gate locally; grading API
  central or edge; deterministic triage queues specialist review.
- **Poor connectivity?** Job-queue API tolerates latency; Streamlit runs
  fully offline; frontend needs LAN only.
- **Why telemedicine?** Specialists are urban-centralized; screening must
  travel, not patients.
- **Scale?** Simulated scenarios to 150k/year under stated assumptions;
  unexecuted here; no capacity claimed.

## Simulink

- **Why?** Operational questions (staffing, queues, waits) need a
  workflow model, not an AI model.
- **Did 100k run?** No runtime in this environment — code + contracts
  delivered, execution explicitly not claimed.
- **Assumptions vs measured?** Arrival/staffing/network assumed; quality
  mix, triage mix, AI timings measured. Labelled per parameter.

## MATLAB

- **Why?** Engineering deployment target in clinical-engineering contexts;
  demonstration ports + authoritative JSON import.
- **What implemented?** Preprocessing/quality demos, visualizations,
  report import, workflow structs — all labelled demonstration-grade.
- **Why no EfficientNet?** Duplicating the grader adds risk without
  value; Python remains the single AI source of truth.

## Limitations

- **Clinically validated?** No. Nothing here is.
- **Can it diagnose?** No — screening support with mandatory review.
- **Before deployment?** Recalibrated quality gate, prospective
  multi-site validation, regulatory clearance, clinical workflow
  integration, monitoring, and staffing model — years of work, stated.
