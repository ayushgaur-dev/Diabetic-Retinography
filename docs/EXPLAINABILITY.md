# Unified Explainability (SIH26038 — Phase 6)

## 1. Objective

One structured, read-only explanation per image: what the model attended
to (Grad-CAM) plus what interpretable detectors found (vessels, disc,
fovea, 4 lesion types), with spatial-agreement statistics. Prediction and
probabilities pass through untouched (tested). No model/UI/lesion/
calibration changes.

## 2. Grad-CAM methodology

Predicted-class path reuses Phase 1 `make_gradcam_heatmap()` UNCHANGED
(verified: heatmap 7×7, finite, min-max, layer `top_activation`).
Explicit-class path adds a targeted variant through the SAME base graph
(the loaded Sequential lacks `.input`, so a generic re-wrap fails —
documented; nested conv search instead). Records explained class, its
probability, layer, shape, normalization. Output is an ACTIVATION
heatmap, never a probability map.

## 3. Explicit evidence sources

FOV (Phase 2), vessels (4A), disc (4B), fovea (4C), 4 lesion detectors
(4D) — all run unmodified via their own pipelines, each individually
accessible in the result, each with graceful fallbacks + warnings.

## 4. Coordinate alignment

Common frame = ORIGINAL pixels, x=column/y=row. Heatmaps bilinear,
binary masks nearest-neighbor, points rasterized directly. Corner-to-
corner mapping (no shifts — shift test included).

## 5. FOV handling

All statistics FOV-restricted; background activation reported as
`outside_fov_fraction` (demo: 2.0%). Background never dominates.

## 6. Lesion/Grad-CAM overlap

Per type: lesion area, overlap area, lesion-inside-hot fraction,
hot-inside-lesion fraction, IoU. Demo case: hard-exudate agreement 0.83,
MA/HE ≈ 0.37. Wording is "spatial agreement", never causation.

## 7. Anatomical context

Mean activation inside disc / fovea / elsewhere + on/off vessels.
Explanatory statistics only; missing landmarks → None + warning.

## 8. Vessel context

Mean on/off-vessel activation. Spatial evidence only; no vascular-
pathology claims.

## 9. Evidence fusion

Descriptive ranking (overlap 0.5 / score 0.3 / area 0.2) for explanation
usefulness, NOT severity. No ML classifier; grade/probabilities immutable.

## 10. Consistency analysis

SUPPORTIVE (≥0.40) / PARTIALLY (≥0.15) / INCONCLUSIVE (no lesions) /
CONFLICTING (lesions present, agreement < 0.15). CONFLICTING means
spatial disagreement, not model error — stated in every reason string.

## 11. Visualization

6 panels (original / Grad-CAM / FOV+vessels / disc+fovea / lesions /
unified) with per-channel overlays and a printed deterministic summary
("predicted" language, no LLM, no invented findings).

## 12. Faithfulness sanity check

Deletion: mask top-20% activation (blurred fill), re-run, report
P(predicted) drop. Subset: mean drop 0.032, median 0.034 — small, and
twice NEGATIVE (masking raised confidence). Sanity signal only; NOT
causal proof. The negative drops are reported, not hidden.

## 13. Actual results (10 APTOS images, 2/grade, frozen pipeline)

9 SUPPORTIVE + 1 PARTIALLY_SUPPORTIVE; mean lesion-overlap 0.37;
40 regions/image (cap); mean 39.7 s/image CPU (lesion + fovea context
dominate). Descriptive subset statistics — not explanation validation.

## 14. Limitations

- Grad-CAM is attribution, not diagnosis; overlap ≠ causation.
- Lesion evidence is heuristic (Phase 4D precision limits apply).
- Confidence uncalibrated (Phase 7); consistency ≠ clinical validation.
- 39.7 s/image too slow for live screening — optimization later.
- APTOS/IDRiD results say nothing about rural deployment.
- Loaded-Sequential `.input` quirk required the base-graph targeted path.

## 15. Reproducibility

`APTOS_DATA_ROOT=<mirror> python -m src.explainability.evaluate
--n-per-grade 2 [--figures]` → `reports/explainability/{metrics.json,
per_image.csv,config.json}` (+ local-only `figures/`, gitignored).
`python -m src.explainability.demo --image IMG [--out FIG]`.
