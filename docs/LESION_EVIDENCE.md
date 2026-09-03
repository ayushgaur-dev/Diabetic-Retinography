# Lesion Evidence Baseline (SIH26038 — Phase 4D)

## 1. Objective

Classical, interpretable lesion-EVIDENCE extraction (microaneurysms,
hemorrhages, hard/soft exudates) for downstream grading/explainability.
Evidence, never diagnosis: candidates carry heuristic confidence, and no
DR-model, Streamlit, calibration, or earlier-module code was changed.

## 2. IDRiD annotation structure

IDRiD `A. Segmentation.zip` (~557 MB, Zenodo mirror, never committed):
54 training + 27 testing JPGs (4288×2848) with pixel-aligned TIF masks
(`IDRiD_NN_{MA,HE,EX,SE,OD}.tif`). Masks normalized by the loader to
binary 0/255 regardless of stored shape/values (one test EX mask is RGBA
{0,255}). Known gaps, all explicit: train IDRiD_43 lacks its HE file; SE
files exist for 26/54 train + 14/27 test only. Missing files are flagged
(`gt_present`) and scored as all-negative. Masks are pixel-aligned with
images (identical shapes verified), so no row/column ambiguity exists.
GT sizes (IDRiD_02, full-res px): MA 41 objs median 198; HE 22 objs median
540; EX 166 objs median 242. Root: `--data-root` > `IDRID_SEG_DATA_ROOT`
(> `IDRID_DATA_ROOT`) > `data/idrid-seg/`.

## 3. Preprocessing

Shared layer: working width 1072 (configurable — NOT 224, which would
erase microaneurysms), Phase 2 FOV, FOV-restricted percentile green.
Scale factor recorded; candidate coordinates mapped to original pixels
(x=column, y=row).

## 4. Microaneurysm method

Inverted-green multi-scale white top-hat [2,3,4,6] → 98.5th percentile →
area 3–220 (scale-aware) → circularity ≥ 0.45 → vessel-overlap veto 0.6.
Features: equivalent diameter, circularity, green intensity, response
contrast, vessel overlap/distance. Score = 0.35 contrast + 0.35 shape +
0.30 isolation.

## 5. Hemorrhage method

Separate detector (scale differs ~10×): regional darkness (green vs
median-51 background) + red darkness, percentile 97, area 8–3000,
circularity ≥ 0.12 (elongated OK), vessel tolerance 0.8. (An earlier
large-scale top-hat variant scored F1 = 0.000 — soft blotches lack the
edges top-hat needs; replaced during tuning, documented here.)

## 6. Hard-exudate method

Luminance ≥ p90 AND yellowness min(R,G)−B ≥ 48 → area/shape filter.
Design correction during tuning: the first version (brightness p98.5 +
saturation + red-blue-gap cap) caught 6% of GT — saturation does NOT
separate yellow exudates from reddish background (both saturated), and
the gap cap was backwards. Measured distributions drove the redesign.

## 7. Soft-exudate method

Luminance ≥ p90 AND low saturation (≤190) AND above-median local texture
('fluffy' vs flat pale sheen). Weakly tuned (2 SE-positive tuning
images); remains the weakest detector — reported honestly, architecture
kept so a trained model can replace it per-type later.

## 8. Optic-disc exclusion

Dilated (×1.6) 4B disc circle zeroed for bright-lesion detectors;
LOW_CONFIDENCE → ×1.2 + warning; NOT_DETECTED → no exclusion + warning.
Never pretends success.

## 9. Fovea context

Phase 4C center (when available) supplies `distance_to_fovea` /
`distance_to_disc` candidate features only — no hard rules; NOT_DETECTED
fovea → −1 features, pipeline continues.

## 10. Vessel context

Phase 4A mask supplies overlap veto + distance-transform proximity (+
sparsity context). Heuristic support only, never a biomarker; missing
vessels → neutral fallback + warning.

## 11. Candidate scoring

Per-type weighted sums (weights sum to 1.0, exposed with features).
Status: best ≥ 0.55 DETECTED; ≥ 0.35 LOW_CONFIDENCE; else NOT_DETECTED
(weak candidates listed, never claimed). Confidence = best candidate
score: heuristic evidence confidence, NOT probability.

## 12. Postprocessing

Hole filling + FOV clipping (+ disc exclusion upstream). No aggressive
smoothing (MA safety). Circularity clamped to [0,1] (pixelated tiny
perimeters otherwise exceed 1 — caught during tuning).

## 13. Evaluation protocol

Tuned on 8 training images (IDRiD_01–08), config frozen, ONE run on 27
test images. Pixel metrics (FOV-restricted micro) + object metrics
(centroid-in-mask recall / any-overlap precision, fixed pre-evaluation)
+ positive-image detection rate. Reports in `reports/lesions/`.

## 14. Actual results (frozen config, IDRiD test, 27 images)

| Lesion | pos imgs | pix P / R / F1 / IoU | obj R / P |
|---|---|---|---|
| Microaneurysm | 27/27 | 0.101 / 0.062 / 0.077 / 0.040 | 0.180 / 0.100 |
| Hemorrhage | 27/27 | 0.434 / 0.060 / 0.105 / 0.056 | 0.093 / 0.205 |
| Hard exudate | 27/27 | 0.404 / 0.163 / 0.232 / 0.131 | 0.137 / 0.373 |
| Soft exudate | 14/27 | 0.000 / 0.000 / 0.000 / 0.000 | 0.259 / 0.259 |

Reading: HE/EX reach ~0.4 precision (usable evidence with review);
MA recall is low (tiny-lesion regime); SE pixel metrics are zero
(trace overlap only) — the classical pale-gate fails on IDRiD and is
reported as such. Ugly numbers kept per protocol; no test-set tuning.

## 15. Runtime

Mean ≈9.2 s/image at 4288×2848 CPU (median in metrics.json): shared
context (vessels/disc/fovea at high res) dominates; four detectors share
it. Demo single image 3.9 s. No deployment claim; bottleneck documented
for later optimization (never at the cost of MA sensitivity).

## 16. Limitations

- Classical CV confuses vessels/pigment/illumination with lesions
  (MA precision ≈ 0.10).
- SE detector non-functional on IDRiD (F1 0.000) — needs a learned model.
- 8-image tuning is thin; thresholds are heuristics.
- IDRiD is single-region Indian research data, quality-controlled —
  not rural-field data; no clinical claim.
- Missing-mask-as-negative is a standard but imperfect reading.

## 17. Failure modes

Empty FOV / blank input → graceful NOT_DETECTED (tested); missing
disc/fovea/vessels → neutral fallbacks (tested); RGBA/255-valued masks
normalized (tested); weak candidates listed but unclaimed.

## 18. Reproducibility

`IDRID_SEG_DATA_ROOT=<extracted dir> python -m src.retina.lesions.evaluate
--split test`. Rerun history: HE top-hat→regional-darkness redesign and
EX saturation→yellowness redesign happened during TUNING (train only);
post-freeze changes were bug fixes only (circularity clamp, empty-FOV
red guard), followed by the single recorded test run.
