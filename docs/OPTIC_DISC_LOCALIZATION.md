# Optic Disc Localization (SIH26038 — Phase 4B)

## 1. Objective

Approximate optic-disc REGION (center + radius + bbox + algorithmic
confidence) for downstream fovea/lesion/evidence use. Localization, NOT
segmentation. No DR-model, Grad-CAM, or vessel-code changes.

## 2. Anatomical motivation

The disc is a bright yellow-white, roughly circular region where major
vessels converge. Failure modes are known upfront: specular reflections,
exudates, and peripapillary atrophy are also bright — so brightness alone
never decides; geometry, local contrast, and vessel convergence vote too.

## 3. Existing infrastructure reused

Phase 2 `detect_retinal_field()` (FOV — NOT duplicated), Phase 2 config
pattern (JSON-over-defaults), Phase 4A `segment_vessels()` (convergence
support only, unmodified), repo dataclass/visualization/demo/test
conventions. Working resolution capped at 800px width with coordinates
mapped back (documented in warnings).

## 4. Candidate generation (`candidates.py`)

Red-channel top-3% in FOV → vessel-aware closing (merges vessel-split disc
fragments; kernel scales with width) → connected components → filters:
area 0.2–5% of FOV, circularity ≥ 0.25, aspect ≤ 2.2. Features per
candidate: brightness, green-channel ring contrast, circularity, aspect,
area. Generation and scoring are separate stages.

## 5. Candidate scoring (`scoring.py`)

Weighted sum (brightness 0.30, contrast 0.25, geometry 0.20, convergence
0.25 — config weights summing to 1.0) with per-candidate contributions
exposed. Brightness is normalized to the IMAGE's own FOV red range
(median..max), so dark-but-valid captures still score (tuning image 002:
red mean 37, still detected at 5px error).

## 6. Vessel convergence (`vessel_convergence.py`)

Annulus [r, 2r] vessel density ÷ global FOV density from the Phase 4A
mask (run at working resolution). Supporting weight only; empty vessel
mask → convergence 0 + warning (tested fallback, no crash).

## 7. Anatomical priors (weak, orientation-free)

Inside-FOV only, plausible normalized size, circular-ish shape. NO
left/right assumption, NO fixed coordinates — all area measures are
FOV-normalized fractions; radius also reported normalized by image width.

## 8. Failure handling

DETECTED (score ≥ 0.45) / LOW_CONFIDENCE (≥ 0.30, detected=true + verify
warning) / NOT_DETECTED (detected=false, center/radius/bbox None, explicit
warnings, downstream-safe). Confidence is an algorithmic candidate score —
NOT calibrated, NOT clinical. Never invents coordinates.

## 9. Evaluation protocol

Tuned on 5 Drishti-GS TRAINING images (frozen), evaluated ONCE on the 51
TEST images (independent). GT: `diskCenter.txt` (matrix order row,col —
verified against predictions + softmap centroids) and OD-softmap majority
diameter. Metrics: center error (px), normalized error (dist ÷ GT
diameter), detection@0.25/0.5/1.0 diameters, circle-bbox IoU.
`python -m src.retina.optic_disc.evaluate` writes
`reports/optic_disc/{metrics.json,per_image.csv,config.json}` (committed).

## 10. Results (actual, from `metrics.json`)

- 50/51 detected (98%); single miss = safe NOT_DETECTED, 0 candidates
- Median normalized error 0.030 disc diameters
- Detection @0.25 = @0.5 = @1.0 = 0.9804 (50/51 within a quarter diameter)
- Mean bbox IoU 0.79
- DRIVE smoke (no GT, qualitative): 3/3 DETECTED, radii ≈42–45px at
  565px width, lateral positions consistent with mixed laterality.

## 11. Runtime

Mean 142 ms/image at ~2045×1752 on CPU (dominated by Phase 4A vessel pass
at working resolution); ~35 ms vessel + morphology overhead at 565px.
Lightweight; no deployment claim.

## 12. Limitations

- Approximate regions, not segmentations; IoU ≈ 0.79 reflects that.
- Confidence is uncalibrated; thresholds are heuristics from 5 images.
- Bright pathology near the disc can still merge/win (closing + geometry
  mitigate, not eliminate).
- Drishti-GS is a small Indian glaucoma-enriched set — NOT rural-screening
  validation and NOT a DR population; no clinical claim is made.
- Resolution behaviour verified at 300–2140px via normalized metrics, but
  thresholds were tuned at 800px working width.
