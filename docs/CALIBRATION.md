# Confidence Calibration (SIH26038 — Phase 7)

## 1. Motivation

Raw softmax maxima (mean 0.69 on test) are not correctness frequencies.
Temperature scaling aligns them with observed frequencies — without
touching the classifier. Accuracy/QWK/F1 invariance is EXPECTED, not a
failure.

## 2. Raw softmax confidence

`raw_confidence` = max softmax, always labelled UNCALIBRATED (in code,
reports, demo, and docs). Nothing in this phase is called calibrated
until the temperature is applied — and even then never clinical.

## 3. Temperature scaling

Calibrated = softmax(log(p)/T), T single scalar. Logits recovered as
log(p): exact for temperature scaling by shift-invariance (documented;
artifact untouched — true pre-softmax logits preferred if ever exposed).

## 4. Validation/test separation

Phase 5 split reconstructed identically (test IDs verified against
`reports/grading/per_image.csv` — abort on drift). T fitted on 549
validation images; 550 test images evaluation-only. `fit_temperature`
takes numpy arrays (structural isolation, tested); test labels never
enter the optimizer.

## 5. Optimization

Deterministic golden-section search on log(T) ∈ [−3, 3] (numpy only),
tolerance 1e-6. T = 1 baseline = identity (tested). Fitted T = 0.9542
(mild sharpening; model slightly under-confident on val). Val NLL
0.7615 → 0.7609. Bounds [0.05, 20] asserted post-fit.

## 6–9. Metrics, ECE, Brier, NLL

ECE (equal-width; 10 primary + 15/20 sensitivity), MCE, multiclass Brier,
NLL, one-vs-rest classwise Brier/ECE (minority classes flagged by
support), binary referable Brier/ECE. Test results (§13).

## 10. Reliability diagrams

`figures/reliability_{raw,calibrated}.png` (multiclass) +
`reliability_referable_{raw,calibrated}.png` (binary) + confidence-shift
histogram. Same bins raw vs calibrated, diagonal reference, counts shown.

## 11. Referable calibration

Score P2+P3+P4 from raw vs calibrated probs; frozen Phase 5 threshold
0.7 preserved (NOT retuned). Binary Brier 0.0743→0.0740, ECE
0.0760→0.0740. Decisions changed: 2/550 (0.36%; both new FPs —
sensitivity 0.9417 unchanged, specificity 0.8991→0.8930). Summed
probabilities can cross thresholds while argmax is fixed — measured,
not assumed.

## 12. Prediction invariance

Multiclass argmax identical on all 550 test images (0 changed —
verified, else the run would have been stopped). Accuracy/macro-F1/QWK
identical by construction.

## 13. Actual results (frozen T = 0.9542, test n = 550)

- NLL 0.7639 → 0.7636; Brier 0.0787 → 0.0787 (tie); ECE(10)
  0.0472 → 0.0440; ECE(15/20) improve; MCE(10) 0.141→0.139 but
  MCE(20) WORSENS 0.157→0.211 (reported, not hidden).
- Classwise: g0/g2/g4 improve marginally; g1 Brier 0.0781→0.0783 and g3
  0.0652→0.0661 worsen marginally (reported).
- Confidence mean 0.688 → 0.698 (less extreme low-end after T<1).
- Bootstrap 95% CIs (1000, seed 42) overlap heavily — effect is small;
  MCE CI [0.10, 0.38] is unstable (stated limitation).
- Honest reading: the model was already well-calibrated (ECE ≈ 0.05);
  temperature scaling gives a small, consistent improvement — NOT a
  transformation. No accuracy/QWK change (expected).

## 14. Limitations

Small effect size; MCE unstable; minority-class classwise estimates
noisy; T fitted on APTOS-val distribution only; calibrated numbers are
not clinical probabilities; APTOS ≠ rural deployment; log(p) recovery
assumes exact softmax outputs.

## 15. Reproducibility

`APTOS_DATA_ROOT=<mirror> python -m src.calibration.evaluate [--out DIR]
[--no-bootstrap]` → `reports/calibration/{metrics.json,per_image.csv,
config.json}` + 5 figures. Test probs reuse verified Phase 5 cache
(spot-check 4.9e-05); val inference fresh (~80 s). Offline, no API.
Model weights hash recorded in metrics; pipeline touches arrays only
(weights immutability structural + hash evidence).
