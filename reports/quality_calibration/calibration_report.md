# Quality-Gate Focus Recalibration Report (SIH26038)

## 1. Scope and integrity

- Recalibrated **only** the Laplacian focus thresholds at the existing
  **fixed 224×224 quality scale**. No Tenengrad switch, no pipeline redesign,
  no aggregation change (`any BAD -> UNGRADABLE, else any BORDERLINE ->
  BORDERLINE, else GOOD` preserved in `src/quality/quality_score.py`).
- **TRAIN n=2563** (APTOS mirror `mariaherrerot/aptos2019`, notebook
  70/15/15 stratified seed 42) defined candidates; **VAL n=549** selected
  thresholds. **Frozen TEST n=550 never inspected, never used for tuning.**
- Grading model (`efficientnetb0_finetuned_patched.keras`), temperature
  (0.9542), referable threshold (0.7) and Phase 5 config untouched.
- All thresholds remain **engineering/research heuristics, NOT clinically
  validated**. External clinical validation still required.

## 2. Problem

Old synthetic-derived focus thresholds (GOOD ≥300 / BAD <60,
`configs/quality_thresholds.json`, `src/quality/focus.py`) rejected ~99% of
real APTOS images as non-GOOD. VAL baseline (old): GOOD 3 (0.5%) /
BORDERLINE 449 (81.8%) / UNGRADABLE 97 (17.7%). TRAIN baseline: GOOD 17
(0.7%) / BORDERLINE 2091 (81.6%) / UNGRADABLE 455 (17.7%).

## 3. Real-fundus distributions (224×224)

TRAIN focus Laplacian variance: p1 17.2 / p5 33.7 / p10 46.8 / p25 74.5 /
**p50 121.0** / p75 159.2 / p90 200.2 / p95 237.4 / p99 319.0.
Below 60: 16.4%; 60–300: 82.2%; ≥300: 1.4%.
VAL focus: p50 123.3 (same shape; below 60: 16.2%; ≥300: 1.5%).
Other components at 224 are overwhelmingly GOOD (illumination median 89,
contrast std 17.4, exposure clipped ~0.001, FOV 0.797) — focus alone drives
rejection. Grade 0 median (~148–153) exceeds grades 1–4 (~84–118);
grade 4 lowest (VAL p50 83.9).

## 4. Complete VAL sweep (18 pairs; BAD × GOOD, BAD < GOOD)

Aggregation simulated per image: focus status from candidate pair +
worst non-focus component status (measured once at 224). Downstream
referable sens/spec computed on graded subset (gate-passed) using frozen
model VAL inference (VAL only).

| BAD | GOOD | GOOD% | BORDERLINE% | UNGRADABLE% | Pass% | G0 | G1 | G2 | G3 | G4 | Ref pass | Nonref pass | n graded | Sens | Spec |
|-----|------|-------|-------------|-------------|-------|----|----|----|----|----|----------|-------------|----------|------|------|
| 25 | 80 | 36.8 | 59.0 | 4.2 | 95.8 | 97.4 | 94.5 | 93.3 | 100.0 | 93.2 | 94.2 | 96.9 | 526 | 0.9524 | 0.8544 |
| 25 | 100 | 35.2 | 60.7 | 4.2 | 95.8 | 97.4 | 94.5 | 93.3 | 100.0 | 93.2 | 94.2 | 96.9 | 526 | 0.9524 | 0.8544 |
| 25 | 120 | 32.1 | 63.8 | 4.2 | 95.8 | 97.4 | 94.5 | 93.3 | 100.0 | 93.2 | 94.2 | 96.9 | 526 | 0.9524 | 0.8544 |
| 25 | 150 | 22.2 | 73.6 | 4.2 | 95.8 | 97.4 | 94.5 | 93.3 | 100.0 | 93.2 | 94.2 | 96.9 | 526 | 0.9524 | 0.8544 |
| 25 | 180 | 10.0 | 85.8 | 4.2 | 95.8 | 97.4 | 94.5 | 93.3 | 100.0 | 93.2 | 94.2 | 96.9 | 526 | 0.9524 | 0.8544 |
| 25 | 200 | 5.6 | 90.2 | 4.2 | 95.8 | 97.4 | 94.5 | 93.3 | 100.0 | 93.2 | 94.2 | 96.9 | 526 | 0.9524 | 0.8544 |
| 40 | 80 | 36.8 | 54.5 | 8.7 | 91.3 | 93.4 | 92.7 | 90.0 | 93.1 | 79.5 | 88.3 | 93.3 | 501 | 0.9492 | 0.8520 |
| 40 | 100 | 35.2 | 56.1 | 8.7 | 91.3 | 93.4 | 92.7 | 90.0 | 93.1 | 79.5 | 88.3 | 93.3 | 501 | 0.9492 | 0.8520 |
| 40 | 120 | 32.1 | 59.2 | 8.7 | 91.3 | 93.4 | 92.7 | 90.0 | 93.1 | 79.5 | 88.3 | 93.3 | 501 | 0.9492 | 0.8520 |
| 40 | 150 | 22.2 | 69.0 | 8.7 | 91.3 | 93.4 | 92.7 | 90.0 | 93.1 | 79.5 | 88.3 | 93.3 | 501 | 0.9492 | 0.8520 |
| 40 | 180 | 10.0 | 81.2 | 8.7 | 91.3 | 93.4 | 92.7 | 90.0 | 93.1 | 79.5 | 88.3 | 93.3 | 501 | 0.9492 | 0.8520 |
| 40 | 200 | 5.6 | 85.6 | 8.7 | 91.3 | 93.4 | 92.7 | 90.0 | 93.1 | 79.5 | 88.3 | 93.3 | 501 | 0.9492 | 0.8520 |
| 60 | 80 | 36.8 | 45.5 | 17.7 | 82.3 | 86.0 | 81.8 | 81.3 | 79.3 | 65.9 | 78.0 | 85.3 | 452 | 0.9540 | 0.8561 |
| 60 | 100 | 35.2 | 47.2 | 17.7 | 82.3 | 86.0 | 81.8 | 81.3 | 79.3 | 65.9 | 78.0 | 85.3 | 452 | 0.9540 | 0.8561 |
| 60 | 120 | 32.1 | 50.3 | 17.7 | 82.3 | 86.0 | 81.8 | 81.3 | 79.3 | 65.9 | 78.0 | 85.3 | 452 | 0.9540 | 0.8561 |
| 60 | 150 | 22.2 | 60.1 | 17.7 | 82.3 | 86.0 | 81.8 | 81.3 | 79.3 | 65.9 | 78.0 | 85.3 | 452 | 0.9540 | 0.8561 |
| 60 | 180 | 10.0 | 72.3 | 17.7 | 82.3 | 86.0 | 81.8 | 81.3 | 79.3 | 65.9 | 78.0 | 85.3 | 452 | 0.9540 | 0.8561 |
| 60 | 200 | 5.6 | 76.7 | 17.7 | 82.3 | 86.0 | 81.8 | 81.3 | 79.3 | 65.9 | 78.0 | 85.3 | 452 | 0.9540 | 0.8561 |

Baseline (old 300/60) VAL: GOOD 3 / BORDERLINE 449 / UNGRADABLE 97.
Key structural finding: **BAD drives pass/reject; GOOD only splits
GOOD vs BORDERLINE** (both pass the gate). Downstream sens/spec stable
across all pairs (0.949–0.954 / 0.852–0.856) — gate changes coverage, not
model accuracy. Enhancement-discard (Phase 5: ~73% of BORDERLINE discarded
on test) is NOT included here; gate pass is an upper bound on graded
coverage. Preferring more GOOD (lower GOOD threshold) reduces enhancement
load — one reason to anchor GOOD at the median rather than higher.

## 5. Selection rationale (NOT highest pass rate)

**Selected: GOOD ≥120 / BAD <40**
(`configs/quality_thresholds.json`, `src/quality/config.py` DEFAULTS).

- GOOD 120 ~= TRAIN median (121) / VAL median (123): principled,
  distribution-anchored; yields VAL GOOD 32.1% (vs 0.5% before) —
  substantial false-rejection fix without declaring most images GOOD.
- BAD 40 ~= TRAIN p6–7: minimal effective safety relaxation (60→40, one
  candidate step). Halves VAL rejection 17.7%→8.7% (97→48 blocked) and
  raises pass 82.3%→91.3%. Keeps a real safety tail (worst ~9% blocked
  including other-component BADs); synthetic blurred fixture (1.8) still
  BAD by wide margin.
- Bias acceptable but noted: grade pass 79.5 (G4) –93.4 (G0); referable
  88.3 vs non-referable 93.3 (5pt gap). Grade 4 has genuinely lower focus
  (VAL p50 83.9); the 25–40 band (25 VAL images, 13 referable, 80%
  correctly graded) is the tradeoff zone — blocking it preserves safety
  margin at the cost of grade-4 coverage.
- Alternative BAD 25 (pass 95.8%, ref 94.2 vs nonref 96.9, minimal bias)
  documented but rejected as primary: it quarters rejection to 4.2% and
  Evans safety margin to ~p2.5; defensible only if field data later shows
  higher coverage is safe. Kept as the higher-coverage alternative.
- GOOD 100/150 neighbours differ only in GOOD/BORDERLINE split (same
  pass); 120 chosen over 100 (less lax GOOD) and 150 (less enhancement
  load: BORDERLINE 59% vs 69%).

If field evidence later contradicts the safety/bias balance, revisit BAD
25 vs 40 — do not re-tune on the frozen test.

## 6. What changed

- `configs/quality_thresholds.json`: focus `good_var` 300→120,
  `borderline_var` 60→40 (+ provenance comment). No other threshold
  touched. `log_lo/hi` unchanged (scores advisory only).
- `src/quality/config.py`: DEFAULTS fallback mirrored (120/40).
- `docs/IMAGE_QUALITY_ASSESSMENT.md`: §4a recalibration record + §10
  limitation update. No pipeline, aggregation, model, calibration or
  evaluation-config change.
- Synthetic fixtures: **unchanged** — all 22 `tests/test_quality.py`
  pass as-is (sharp 1556 still GOOD, blurred 1.8 still BAD). No test
  weakened.

## 7. Old vs new (VAL n=549; TRAIN n=2563 for baseline)

- VAL old (300/60): GOOD 0.5% / BORDERLINE 81.8% / UNGRADABLE 17.7%.
- VAL new (120/40): GOOD 32.1% / BORDERLINE 59.2% / UNGRADABLE 8.7%.
- TRAIN old: GOOD 0.7% / BORDERLINE 81.6% / UNGRADABLE 17.7%.
- TRAIN new (exact re-run, n=2563, 120/40): GOOD 801 (31.3%) / BORDERLINE
  1550 (60.5%) / UNGRADABLE 212 (8.3%) — matches VAL within 1pt on all
  states, confirming split stability.
- Frozen TEST results: **unchanged, untouched**.

## 8. Risks / limitations

- APTOS ≠ deployment domain; thresholds may not transfer.
- Grade-4 lower focus creates residual bias (79.5% pass at selected pair).
- No human blur labels; 25–40 band gradability (80% correct) vs quality
  tradeoff unresolved without manual review.
- VAL n=549 small for tail estimation; single-dataset calibration.
- Gate pass ≠ final graded coverage (BORDERLINE→enhancement discard
  ~73% on test per Phase 5); effective coverage lower than 91.3%.
- Scale coupling: thresholds valid ONLY at 224×224; any resize change
  requires re-validation.
