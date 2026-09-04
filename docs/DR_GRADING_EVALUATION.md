# DR Grading Evaluation (SIH26038 — Phase 5)

## 1. Objective

Answer with held-out evidence: how does the frozen EfficientNetB0 grader
perform on 5-class DR grading and referable-DR screening? No model, UI,
lesion, calibration, or MATLAB changes — evaluation only.

## 2. Model evaluated

`models/efficientnetb0_finetuned_patched.keras` (Phase 1 artifact,
23,425,638 bytes, SHA256 in `docs/MODEL_ARTIFACTS.md`). Loaded read-only
via the Phase 1 path; architecture/weights untouched. Trained/fine-tuned
on APTOS (notebook 70% split) — so this is IN-DATASET evaluation; the
notebook test_df (built, never evaluated there) is the held-out set.

## 3. Dataset

APTOS 2019 via mirror `mariaherrerot/aptos2019` (8 GB, cached outside
repo): `train_1.csv` (2930) + `valid.csv` (366) + `test.csv` (366) with
`id_code,diagnosis`, nested PNG dirs (2930/366/366 files). Combined =
3662 labels; class distribution train/val/test =
1263-259-699-135-207 / 271-55-150-29-44 / 271-56-150-29-44 (matches the
notebook's 1805/370/999/193/295). No missing images, no duplicate IDs, no
malformed grades (loader validates; raises otherwise). The mirror's own
pre-split is NOT trusted — labels are recombined and re-split below.

## 4. Split strategy

Exact notebook reconstruction (`02[6]`): 70/30 then 50/50, stratified,
seed 42 → train 2563 / val 549 / test 550. Frozen before metrics.
Config-overridable seed; same seed reproduces identical splits (tested).

## 5. Leakage audit (headline finding)

- ID overlaps: train/test, val/test, train/val all EMPTY.
- File-hash audit (all 3662 images): **123 duplicate groups; 55 cross-split
  groups; 30 groups with MIXED grades** (byte-identical pixels, different
  DR labels — APTOS label noise, quantified).
- 7 val↔test twin pairs + train↔test twins: 31/550 test images (5.6%) have
  a byte-twin outside test. Patient-level independence is UNKNOWN (no
  patient IDs — documented limitation).
- Impact bounded by dedup sensitivity rerun (strict-clean test set):
  QWK 0.8107→0.8223, sens 0.9596→0.9653, spec 0.8716→0.8896, AUROC
  0.9737→0.9765. Small shift; primary result stands with the caveat above.

## 6. Inference procedure

PIL RGB → 224 → `efficientnet.preprocess_input` → batched predict (32,
CPU); per-image record (id, GT, grade, P0–P4, raw max-softmax).
Runtime: ~80 s per 550 images CPU; total run 298 s.

## 7. Five-class metrics (held-out test, n=550)

Accuracy 0.68; macro P/R/F1 0.587/0.524/0.468; weighted F1 0.673;
Cohen κ 0.525; **QWK 0.8107** (95% CI 0.774–0.844; val-only 0.7987 was
honest and close). Per-class R: 0→0.978, 1→0.107, 2→0.427, 3→0.655,
4→0.455 — minority/intermediate grades weak, NOT hidden.

## 8. Referable DR (grade ≥ 2, score P2+P3+P4, threshold 0.5)

TP 214 / TN 285 / FP 42 / FN 9 (prevalence 0.406):
**sensitivity 0.9596 (CI 0.932–0.983), specificity 0.8716 (CI 0.835–0.908),
PPV 0.836, NPV 0.969, F1 0.894, accuracy 0.907, balanced 0.916.**
Both SIH targets (sens > 90%, spec > 85%) are MET on held-out data —
reported as engineering results with CIs, not clinical claims. Frozen
val-chosen threshold 0.7: sens 0.9417 / spec 0.8991 (also above targets).

## 9. Threshold analysis

Grid 0.3–0.7 on VALIDATION only; frozen = max F1 (tie: sensitivity) =
0.7; applied once to test (§8). Curve in `figures/threshold_curve.png`.
Thresholding ≠ calibration (no temperature scaling anywhere).

## 10. ROC/AUC

AUROC **0.9737** (CI 0.962–0.983); average precision (PR-AUC) **0.96**.
Curves in `figures/roc.png`, `figures/pr.png`.

## 11. Error analysis

Adjacent-pair errors dominate (1↔2 especially); severe mistakes tallied
(`severe_mistakes` in metrics.json). Worst table in `errors` (severe +
referable-FN first, by raw confidence): top entry is a referable-FN
(GT 2 → pred 0) at raw 0.88 — confidently wrong cases exist and are
listed, not buried.

## 12. Quality analysis (second headline finding)

Test quality states: GOOD 3 / BORDERLINE 473 / UNGRADABLE 74 (blocked,
unscored). Component breakdown: focus BAD/BORDERLINE on 541/550;
illumination/contrast mostly GOOD. Conclusion: Phase 2 focus thresholds
(synthetic-tuned) are far too strict for real APTOS texture — the gate as
calibrated would divert ~99.5% of screening images. NOT retuned here
(test-tuning forbidden); recalibration on real data is required future
work. Grading metrics above use original inputs (gate not applied).

## 13. Enhancement ablation (BORDERLINE only)

128 accepted-enhanced graded vs original-input grades: better 4 / worse
45 / same 79; 345 discarded→blocked. Phase 3 enhancement as calibrated
does NOT improve (likely degrades) DR grading on APTOS — descriptive
ablation only, no model or Phase 3 change. Do not route grading through
enhancement without rethink.

## 14. Raw confidence (descriptive, NOT calibration)

Mean max-prob 0.688 (median 0.656); correct 0.787 vs incorrect 0.479 —
separated but uncalibrated. Histogram in `figures/raw_confidence.png`.
Phase 7 will calibrate; nothing here is called calibrated.

## 15. Bootstrap methodology

1000 resamples, seed 42, percentile 95% CIs on accuracy/macro-F1/QWK/
sens/spec/AUROC (see §7–8). Deterministic; rerun reproduces exactly.

## 16. Limitations

In-dataset (not external) validation; 5.6% test contamination bounded
by dedup rerun; no patient IDs; quality gate miscalibrated (see §12);
duplicate/mixed-grade label noise; APTOS ≠ rural deployment; raw
softmax uncalibrated; no lesion use (Phase 8).

## 17. Reproducibility

`APTOS_DATA_ROOT=<mirror root> python -m src.evaluation.evaluate
--split test [--no-quality] [--no-bootstrap] [--seed N]`. Writes
`reports/grading/{metrics.json,per_image.csv,confusion_matrix.csv,
config.json}` + `figures/` (6 PNGs, committed as small evidence files).
No API key needed (fully offline).
