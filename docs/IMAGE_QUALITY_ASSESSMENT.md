# Image Quality Assessment (SIH26038 — Phase 2)

## 1. Purpose

Decide, BEFORE DR inference, whether a fundus photograph is **GOOD**,
**BORDERLINE**, or **UNGRADABLE**. An ungradable image must never silently
produce a DR grade — the gate blocks the classifier and returns recapture
instructions instead. No DR model code was modified in this phase.

## 2. Pipeline

```
RGB uint8 image
  -> validate (shape/dtype/size; ValueError on invalid input)
  -> field-of-view detection (red-channel threshold + morphology + largest component)
  -> focus | illumination | contrast | exposure (measured INSIDE the mask)
  -> retinal coverage (fitted-circle completeness)
  -> aggregation rules -> QualityResult + recapture feedback
  -> gate: UNGRADABLE blocks model_fn | BORDERLINE flags enhancement_required | GOOD passes
```

Entry points: `assess_image()` in `src/quality/quality_pipeline.py`;
`screen_image()` in `src/quality/gate.py` (injects any `model_fn`, so the
existing EfficientNet inference code is untouched — backward compatible).

Streamlit integration (`app/streamlit_app.py`, Screening flow only):
quality runs right after upload — UNGRADABLE stops with recapture messages,
BORDERLINE warns (enhancement is a Phase 3 hook) and continues, GOOD shows a
one-line caption. Direct image→prediction code paths are preserved.

## 3. Metrics (all measured inside the retinal mask)

| Component | File | Raw measurement | Normalisation |
|---|---|---|---|
| Focus | `focus.py` | Laplacian variance (uint8 gray) | log10 mapped between `log_lo`/`log_hi` |
| Illumination | `illumination.py` | median gray + non-uniformity (std of 4×4 block means) | ramps around good/borderline bands |
| Contrast | `contrast.py` | std of gray (+ p95–p5 guard; upper guard: excessive ≠ better) | linear between `std_borderline`/`std_good` |
| Exposure | `exposure.py` | fraction of pixels ≤ black / ≥ white level | 1 − clipped/tolerance |
| Field of view | `field_of_view.py` + `coverage.py` | mask pixels / frame pixels | linear between fraction bounds |
| Retinal coverage | `coverage.py` | mask area / min-enclosing-circle area | linear between completeness bounds |

## 4. Thresholds

All thresholds live in `configs/quality_thresholds.json` (JSON chosen: stdlib
only, zero new dependencies; repo convention is no-config — this is the first
config file). Key values: focus good/borderline variance 300/60; illumination
median good 50–150, borderline 30–180, non-uniformity 18/30; contrast std
15/8 with excessive cap 70; exposure black 8%/20%, white 2%/8%; field fraction
40%/20% (plausible floor 5%); completeness 75%/55%.

**Every threshold is an ENGINEERING HEURISTIC**, set from synthetic-fixture
measurements (sharp 1556 vs blurred 1.8 lap-var; dark median 19; mask fraction
0.64 for a full circular field vs 0.05 for a small one). NOT clinically
validated. No threshold is hard-coded in Python (enforced by
`test_no_thresholds_hardcoded_in_component_modules`).

## 5. Quality states

`aggregate()` in `src/quality/quality_score.py`: any component BAD →
UNGRADABLE; else any BORDERLINE → BORDERLINE (`enhancement_required=True`,
`enhancement_applied=False` — Phase 3 hook); else GOOD. `overall_score` is a
config-weighted mean and is ADVISORY ONLY — the gate acts on status.

## 6. Recapture logic

Deterministic per-component messages in `quality_pipeline.RECAPTURE_MESSAGES`
(plain language, no jargon — asserted by test), plus a closing line:
UNGRADABLE → "cannot be graded, recapture before screening";
BORDERLINE → "marginal, enhancement will be required (Phase 3)".
Multiple failures yield multiple instructions.

## 7. Integration

`screen_image(rgb, model_fn, config)` returns quality dict + `gradable` +
`model_called` + `prediction`/`blocked_reason`. Proven by test:
UNGRADABLE → model_fn never invoked; GOOD → invoked exactly once.

## 8. Tests

`tests/test_quality.py` — 22 tests: 7 fixture behaviours (sharp/blurred/
dark/bright/low-contrast/small-field/solid), invalid inputs (4 cases),
schema, config defaults + override, no-hardcoded-thresholds scan,
aggregation rules (3), feedback determinism + plain language, gate
block/call/borderline tests. Fixtures are deterministic synthetic arrays.

## 9. Performance

Measured on a Windows 11 dev laptop (CPU), 224×224 images: **≈4–17 ms per
image, mean ≈8.5 ms across 8 fixtures** (demo CLI: 13.6 ms incl. overhead).
No real-time claim — but comfortably lightweight for field deployment.
Figure rendering (matplotlib) is the slowest step and runs only on demand.

## 10. Limitations

- Thresholds are heuristics from synthetic fixtures, NOT tuned on real APTOS
  images and NOT clinically validated — re-tune with measured data before any
  clinical use (later phase).
- No real-fundus validation yet (APTOS raw images absent); solid-gray and
  black images are rejected, but adversarial/near-duplicate edge cases are
  untested.
- BORDERLINE images are graded unenhanced in the app until Phase 3.
- Focus metric is confounded by severe darkness (dark images score low
  sharpness) — harmless here since they fail illumination anyway.
- `rot90`-style orientation issues from training augmentation are out of scope.
- Overall score must not be mistaken for calibrated confidence (Phase 7).
