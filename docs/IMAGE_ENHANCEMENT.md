# Adaptive Image Enhancement (SIH26038 — Phase 3)

## 1. Purpose

Rescue BORDERLINE fundus images: identify the weak quality dimensions,
apply only the matching operations, reassess, and accept or discard.
GOOD images bypass untouched; UNGRADABLE images stay rejected.
The EfficientNetB0 architecture and weights are unchanged — only the
choice of input image (original vs accepted-enhanced) is new.

## 2. Why enhancement is conditional

Blind enhancement wastes compute on GOOD images and risks degrading them
(CLAHE amplifies noise; gain shifts colour). The Phase 2 `QualityResult`
tells the pipeline WHY an image is marginal, so each operation has a
named reason. No DR-grade or label input exists anywhere in this subsystem
(no leakage by construction — verified by `test_no_labels_used_in_enhancement`).

## 3. CLAHE (`operations.apply_clahe`)

LAB colour space, CLAHE on the **L channel only**, then merge back —
never independent RGB equalisation (which distorts hue). Config:
`clip_limit` 2.0, `tile_grid` 8×8. Verified to raise masked std-contrast
on the low-contrast fixture while keeping red the dominant channel.

## 4. Illumination normalization (`operations.normalize_illumination`)

Per-channel gain = masked channel mean / large-median-blur background
(kernel 41, odd-enforced), gains clipped to [0.5, 2.0]. The wide kernel
tracks the slow illumination field while vessels/lesions pass through;
the cap prevents amplifying pathology. Verified to reduce 4×4-block
non-uniformity on a gradient-lit fixture.

## 5. Denoising (`operations.denoise`)

Conservative bilateral filter (d=5, σcolor=25, σspace=5) — edge-preserving
and deliberately mild, because Phase 4D will analyse small lesions.
Runs automatically after CLAHE (which otherwise amplifies grain).

## 6. Color normalization (`operations.normalize_color`)

Masked gray-world with gain cap 1.25: corrects global camera cast while
preserving relative colour differences. Triggered only when the masked
inter-channel mean spread exceeds 25.0. Chosen over percentile-stretch
(which would flatten clinically meaningful colour) — and documented as
NOT a domain-shift fix.

## 7. Retinal masking

Reuses the Phase 2 `detect_retinal_field()` — no second detector.
Every op applies inside the mask and restores the original background
outside it (asserted by `test_operations_preserve_background`).

## 8. Adaptive operation selection (`enhancement.select_operations`)

| Borderline dimension | Operations | Notes |
|---|---|---|
| contrast | clahe + denoise | denoise counters CLAHE grain |
| illumination | illumination_normalization + color_normalization | — |
| exposure (dark-dominated) | illumination_normalization | — |
| exposure (saturation-dominated) | none (+ warning) | clipped pixels hold no signal |
| focus | none (+ warning) | blur destroys information; no sharpening (`allow_unsharp: false`) |

Plus the image-based color-cast trigger. Order deduplicated, all names
recorded in `operations_applied`.

## 9. Before/after quality assessment

Every enhanced image is reassessed with the Phase 2 pipeline.
Comparison uses status ranks AND per-dimension moves — never the raw
`overall_score` alone. `IMPROVED` = higher rank, or same rank with a
strictly improved dimension and none worsened. Result carries
`before_quality`, `after_quality`, `changed_quality_dimensions`.

## 10. Failure handling

Safety checks (`_safety_checks`, thresholds in `sanity_checks`): extra
clipping ≤ 5 pts, mean shift ≤ 40, per-channel shift ≤ 45, mask-fraction
ratio ≥ 0.8, plus a non-finite-pixel veto. Any failure → enhanced image
DISCARDED (`enhanced_image=None`), original retained, `comparison=WORSE`.
UNGRADABLE inputs are never enhanced. A failed rescue returns recapture
guidance and the model is NOT called.

## 11. Integration with inference

`screen_enhance_infer(rgb, model_fn)`: GOOD → model(original);
BORDERLINE → enhance → accepted → model(enhanced), else recapture with no
call; UNGRADABLE → recapture with no call. Streamlit Screening flow uses
this: success banner names the transition and ops; failure stops with
recapture notes. Exact order: raw → quality → optional enhancement →
reassess → existing model preprocessing (resize + `preprocess_input`,
exactly once) → model. `ENHANCEMENT_VERSION="1.0.0"` is recorded in results.

## 12. Performance (dev laptop, CPU)

- Quality assessment: ≈4 ms (224px)
- Enhancement total (assess + ops + reassess): ≈15 ms (224px, clahe+denoise)
- Demo CLI end-to-end: ≈134 ms incl. figure rendering
- 512px smoke: slower (larger morphology/median kernels) but same ms-scale.
Lightweight enough to be plausible for rural/edge CPU deployment; no
real-time claim made.

## 13. Limitations (explicit)

- Enhancement does NOT restore information lost to severe blur or clipping.
- Thresholds/parameters are engineering heuristics, not clinically validated.
- Quality-metric improvement (e.g. 0.688 → 0.825) does NOT imply improved DR
  accuracy — formal grading experiments come later.
- Thresholds were tuned at 224px reference scale; other resolutions shift
  verdicts (observed: 512px low-contrast variant rates UNGRADABLE while the
  224px version is BORDERLINE-rescuable). Resolution handling is future work.
- No real APTOS images were available in-repo; smoke tests use synthetic
  fixtures (224px + 512px). Real-image validation is deferred to later phases.
- Focus recovery is intentionally unimplemented.
