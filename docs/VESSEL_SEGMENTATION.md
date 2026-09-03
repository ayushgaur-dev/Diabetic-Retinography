# Retinal Vessel Segmentation (SIH26038 — Phase 4A)

## 1. Objective

Reproducible classical vessel-segmentation baseline on DRIVE: correct
FOV-restricted evaluation, honest reporting, and a clean
`segment_vessels()` API for the future evidence engine (Phase 8).
No deep learning, no DR-model changes.

## 2. DRIVE dataset

- Source: `https://drive.grand-challenge.org/` (40 images, 565×584, diabetic
  screening population, Netherlands). Research dataset — not a rural-India sample.
- Acquired here via Kaggle mirror `andrewmvd/drive-digital-retinal-images-for-vessel-extraction`
  (28 MB, public). Dataset is NEVER committed; root resolved as
  `--data-root` > `DRIVE_DATA_ROOT` env > `data/drive/` (empty placeholder).
- Manual setup: download from the grand-challenge site or the Kaggle mirror,
  point `DRIVE_DATA_ROOT` at the folder containing `DRIVE/`.

## 3. Dataset structure (verified on disk)

```
DRIVE/training/images/{21..40}_training.tif   20 RGB images
DRIVE/training/1st_manual/{21..40}_manual1.gif 20 vessel GT (binary 0/255)
DRIVE/training/mask/{21..40}_training_mask.gif 20 FOV masks (binary 0/255)
DRIVE/test/images/{01..20}_test.tif           20 RGB images
DRIVE/test/mask/{01..20}_test_mask.gif        20 FOV masks — NO annotations
```

Pairing validated strictly by `dataset.discover()` (missing partner →
`FileNotFoundError` naming the file). **This mirror ships no test
annotations**, so test-split metrics cannot be reported here.

## 4. Preprocessing (`preprocessing.py`)

RGB → green channel (float32 0..1) → mild FOV-restricted percentile stretch
(1–99th). Dedicated path: Phase 3 colour normalisation is NOT reused
(it would compress vessel contrast).

## 5. Green-channel extraction

Haemoglobin absorbs green light, so vessels are darkest against the
background in green; red saturates, blue is noisy. Validated dtype/shape/
finiteness by test.

## 6. Vessel enhancement (`enhancement.py`)

White top-hat on inverted green at disk scales [3, 5, 9, 15] (covers
~2–10 px vessels at 565×584), max-projected, normalised to 0..1.
Chosen for interpretability and zero learned weights. Output is a
**response map, never a probability map**.

## 7. Segmentation (`segmentation.py`)

Percentile threshold inside FOV (frozen at 89.0 ≈ 100 − 11% vessel
fraction observed on image 21). Strict `>` comparison so a degenerate
all-zero response yields an empty mask, not the whole FOV.

## 8. Postprocessing

Drop components < 25 px + optional 3×3 closing, FOV-reclipped.
Conservative by design — thin vessels survive.

## 9. FOV handling

DRIVE FOV masks are used whenever present. Without one,
`segment_vessels()` falls back to the Phase 2 detector with
`fov_source='phase2_fallback'` + warning — never silently.
All metrics count only in-FOV pixels (background is never a true negative).

## 10. Evaluation metrics (`metrics.py`)

Sensitivity (= recall, reported as both with the identity documented),
specificity, precision, F1 (= Dice, likewise), IoU, accuracy —
FOV-restricted, per-image plus MICRO (pooled-pixel, primary) and MACRO
(mean±std) aggregates.

## 11. Experimental protocol

Parameters tuned once on training images 21–23 (percentile 87/89/91 →
F1 0.689/0.696/0.682; froze 89.0), then evaluated ONCE on all 20 training
images vs `1st_manual`. Test split: runtime/mask sanity only (no GT, no
metrics — not faked). Command:
`python -m src.retina.vessels.evaluate [--data-root ...] [--figures]`
writes `reports/vessel_segmentation/{metrics.json,per_image_metrics.csv,config.json}`
(small reproducible records — committed).

## 12. Results (actual, frozen config — from `metrics.json`, not hard-coded here)

- Micro (pooled): F1 0.7135, accuracy 0.9368, sensitivity 0.6280, specificity 0.9810, precision ≈0.80, IoU ≈0.55
- Macro F1 0.7212 ± 0.0509 (n=20); per-image table in `per_image_metrics.csv`
- Honest reading: high specificity, moderate sensitivity — thin vessels are
  under-detected. Classical-baseline territory, below published supervised
  results (reported here only as context: supervised DRIVE methods typically
  exceed F1 ≈ 0.80 — EXTERNAL published values, not ours).

## 13. Runtime

≈27–38 ms/image at 565×584 on CPU (dev laptop); demo single image 35.4 ms.
Lightweight; no real-time claim.

## 14. Limitations

- Classical baseline only; thin-vessel recall is the weak point.
- Measured on DRIVE training split (test annotations unavailable in this
  mirror) — not an independent test result in the strict sense; disclosed.
- DRIVE is a small Dutch research set — performance here does NOT establish
  rural-deployment performance.
- Vessel density is an image statistic, NOT a clinical biomarker; no medical
  claims are made from it.
- Thresholds tuned at DRIVE resolution; resolution sensitivity (seen in
  Phase 3) applies here too.
