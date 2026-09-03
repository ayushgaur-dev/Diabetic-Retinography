# Fovea Localization (SIH26038 — Phase 4C)

## 1. Objective

Interpretable foveal-center estimation (center + region + heuristic
confidence) from disc geometry, appearance, and vessel sparsity. No neural
network, no DR-model/Streamlit/calibration changes, no 4A/4B modifications.

## 2. Anatomical intuition

The fovea lies ~2.5 disc diameters temporal to the disc, slightly below
the disc–fovea axis, in a relatively avascular, dark, smooth zone. Each
property is a separate scored feature so failures are diagnosable.

## 3. Dataset

IDRiD Localization (Nanded, Maharashtra — Indian screening population,
50° FOV, 4288×2848 JPG): 413 training + 103 testing images with fovea AND
disc center CSVs. Acquired as `C. Localization.zip` (~203 MB) via a
Zenodo mirror; never committed. Root: `--data-root` > `IDRID_DATA_ROOT` >
`data/idrid/`. CSV quirk documented: 515 rows/file, only own-split rows
carry coordinates (x=column, y=row — verified against anatomy); blanks
skipped, 413+103 usable, none missing in-split. OD+ fovea CSVs also give
GT laterality (disc nasal).

## 4. Preprocessing

Working width 800px (coordinates mapped back; documented in warnings).
Green channel (reused 4A extractor), Phase 2 FOV (reused detector).

## 5. Optic-disc dependency

Supplied disc dict (input-frame contract, rescaled internally) or Phase 4B
run (recorded in warnings). DETECTED/LOW_CONFIDENCE → full geometry (LOW
adds verify warning); NOT_DETECTED → whole-FOV fallback with pseudo-DD,
geometry weight redistributed (sum still 1.0). Never crashes without a disc.

## 6. Candidate generation

Grid (0.25 DD step) in a 1.5 DD search disk around the expected point per
temporal hypothesis (both sides when laterality unknown); FOV hard gate;
distance-transform boundary feature.

## 7. Candidate features

disc-relative distance/angle, temporal-geometry (Gaussian σ=1 DD),
darkness (patch vs FOV median), contrast (patch vs ring), texture
(smooth-dark vs surround), vessel-sparsity (inner vs outer density ratio —
a heuristic, NOT a biomarker), FOV validity, boundary distance. Weak
features were kept only if used; geometry/appearance/vessel/fov weights
0.35/0.35/0.20/0.10 (sum 1.0, exposed per candidate).

## 8. Scoring

Weighted sum with inspectable contributions (answers "why A beat B").
Laterality: disc-x vs FOV-center-x (margin guard 0.03); unknown →
dual-hypothesis search, ambiguity recorded.

## 9. Failure states

DETECTED ≥ 0.45 / LOW_CONFIDENCE ≥ 0.30 / NOT_DETECTED below; ambiguity
penalty (gap < 0.10 → ×0.7 + warning). Nothing forced; missing disc and
empty search are explicit NOT_DETECTED with downstream-safe dicts.

## 10. Confidence semantics

Heuristic candidate confidence (algorithmic score). NEVER called
calibrated/diagnostic/clinical probability anywhere in code, docs, or UI.

## 11. Evaluation protocol

Tuned on 8 training images (IDRiD_001–008), config frozen; ONE held-out
run on 103 test images end-to-end (4B runs internally — true system test).
Normalization: error ÷ image (FOV) width (GT disc diameters unavailable —
documented alternative). Thresholds 0.025/0.05/0.10 reported, never tuned.
`evaluate` writes `reports/fovea/{metrics.json,per_image.csv,config.json}`.

## 12. Results (actual, frozen config)

- 103/103 DETECTED (0 LOW, 0 NOT), median normalized error 0.0334
- Detection @0.025/0.05/0.10: 0.4369 / 0.5728 / 0.7961
- Laterality: left 47/47 (median 0.043), right 56/56 (median 0.025) —
  no orientation bug (GT-derived breakdown)
- Engineering results only; NOT clinical validation.

## 13. Runtime

Mean ≈1.6 s/image at 4288×2848 on CPU (≈1.1 s single demo) — dominated by
two Phase 4A vessel passes + grid scoring. A coordinate-grid reuse cut
~3.1 s → ~1.6 s with byte-identical outputs. No deployment claim.

## 14. Limitations

- Classical baseline; dark hemorrhages can mimic foveal darkness (tuning
  image 005); wrong-disc geometry degrades gracefully but still misleads.
- IDRiD is Indian but single-clinic, quality-controlled — not rural-field data.
- Scale sensitivity: proportional scenes agree, but absolute noise/vessel
  widths shift appearance scores across resolutions (synthetic test bounds it).
- 0.35/0.35/0.20/0.10 weights from 8 tuning images — modest data, honestly small.

## 15. Reproducibility

`IDRID_DATA_ROOT=<extracted dir> python -m src.retina.fovea.evaluate --split test`.
Rerun history (§13 protocol): (1) appearance 0–255 scale bug found via
resolution test → fixed, same config, test median 0.067→0.033;
(2) supplied-disc frame bug (working vs input pixels) → fixed + candidate
frame unified to original pixels; end-to-end numbers unchanged (0.0334).
No test-set tuning at any point.

## 16. Known failure modes

Bright-pathology disc hijack (IDRiD_001's 4B miss — fovea still landed
0.044 via appearance/vessel); hemorrhage-mimic darkness; near-FOV-edge
foveae lose boundary score; overlapping-grid ties trigger ambiguity
penalty by design.
