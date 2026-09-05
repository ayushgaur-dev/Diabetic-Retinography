# Final Metrics (SIH26038 — Phase 14, single source of truth)

Every number below is copied from a committed report artifact. Context
labels: TEST RESULT (held-out), VALIDATION RESULT, DEMO RESULT (small
samples/smoke runs), ENGINEERING MEASUREMENT (timings/counts),
SIMULATION ASSUMPTION/OUTPUT (Phase 12, unexecuted here).

## DR grading — TEST RESULT (APTOS, notebook 70/15/15 seed-42 split, n=550)

- Dataset: APTOS 2019 mirror; test class distribution
  0:271 / 1:56 / 2:150 / 3:29 / 4:44. Source: `reports/grading/metrics.json`.
- Accuracy 0.68; macro F1 0.4675; weighted F1 0.6728; Cohen's κ 0.5249;
  QWK 0.8107.
- Per-class recall: 0→0.978, 1→0.107, 2→0.427, 3→0.655, 4→0.455.
- Referable (≥2, P2+P3+P4 @ frozen 0.7): sensitivity 0.9596,
  specificity 0.8716, PPV 0.836, NPV 0.969, F1 0.894, ROC-AUC 0.9737, AP 0.96.
- Dedup sensitivity rerun (519 images): QWK 0.822, sens 0.965, spec 0.890.

## Calibration — TEST RESULT (same split; T fit on validation only)

- Temperature 0.9542; val NLL 0.7615→0.7609.
- Test NLL 0.7639→0.7636; ECE(10) 0.0472→0.0440; Brier tie 0.0787;
  MCE(10) 0.141→0.139; MCE(20) worsened 0.157→0.211 (reported).
- Argmax unchanged on all 550; referable decisions changed 2/550.

## Quality — TEST RESULT (same 550)

- GOOD 3 / BORDERLINE 473 / UNGRADABLE 74 / ERROR 0.
- Focus thresholds quantified as too strict on real data (541/550 fail
  focus); NOT retuned (test-tuning forbidden). Source: Phase 5 report.

## Lesions — TEST RESULT (IDRiD Segmentation test, n=27)

- Pixel F1: MA 0.077, HE 0.105, EX 0.232, SE 0.000.
- Object recall: MA 0.180, HE 0.093, EX 0.137, SE 0.259.
- Positives: MA/HE/EX 27/27, SE 14/27.

## Anatomy — TEST/held-out RESULTS (research datasets)

- Vessels (DRIVE training, FOV-restricted): F1 0.7135, acc 0.937,
  sens 0.628, spec 0.981.
- Optic disc (Drishti-GS test, n=51): detection 98.0%, median error
  0.030 disc diameters, det@0.25 = 98.0%, mean bbox IoU 0.79.
- Fovea (IDRiD Localization test, n=103): 103 detected, median
  normalized error 0.033, det@0.05 = 0.573.

## Explainability — DEMO RESULT (10 APTOS images, 2/grade)

- 9 SUPPORTIVE + 1 PARTIALLY_SUPPORTIVE; mean overlap 0.37;
  mean deletion drop 0.032. Descriptive only.

## Triage — TEST RESULT (APTOS test, n=550 + 30-image evidence subset)

- Distribution: TECHNICAL_REVIEW 353 / REFER 50 / UNGRADABLE 74 /
  URGENT_REVIEW 54 / ROUTINE 19.
- Safety: 0 referable false-negatives to ROUTINE; 0 severe undercalls
  to ROUTINE; 39.5% auto-escalated, 60.5% human-reviewed.

## Runtime — ENGINEERING MEASUREMENT (CPU dev laptop)

- Grading eval inference ~80 s/550 images; bootstrap included; total run
  ~300–345 s. Full single-image workup ≈ 40–89 s (evidence-dominated);
  triage 0.04 s; reporting < 0.1 s. Prototype figures, not deployment
  performance.

## Simulation — ASSUMPTIONS + OUTPUT SHAPE (no execution here)

- Scenarios A–D (10k/50k/100k/150k per year), seeded DES code,
  measured mixes + staffing assumptions. No outputs prefilled.
