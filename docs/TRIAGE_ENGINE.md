# Evidence-Based Triage Engine (SIH26038 — Phase 8)

## 1. Objective

Deterministic screening-workflow recommendations (UNGRADABLE / ROUTINE /
REFER / URGENT_REVIEW / TECHNICAL_REVIEW) from frozen phase outputs.
Workflow routing only — never diagnosis. Model grade immutable; no LLM
can override decisions (no LLM exists in the path).

## 2. Input contract

`TriageInput`: quality_status/score, enhancement_status (none/success/
failed), predicted_grade, raw + calibrated probabilities, calibrated
confidence, calibrated referable score, lesion evidence map, vessel
flag, disc/fovea statuses, consistency, processing errors. Assembled from
artifacts (no recompute) by `pipeline.build_input`.

## 3. Output contract

`TriageResult`: decision, priority (LOW/NORMAL/HIGH/CRITICAL, configured),
referable bool, reason codes, evidence summary, 7 safety flags,
calibrated confidence, immutable grade copy, referable score, warnings,
method, deterministic explanation.

## 4. Quality priority

UNGRADABLE → UNGRADABLE + recapture (nothing overrides it).
BORDERLINE+success → proceed + warning; BORDERLINE+failed →
TECHNICAL_REVIEW (grading withheld); BORDERLINE+none → proceed + warning.

## 5. Severity rules

Grade ≥ 3 → URGENT_REVIEW (HIGH_DR_SEVERITY, CRITICAL); grade→state map
mirrors legacy `BASE_RULES` actions. Screening priorities, not treatment
priorities.

## 6. Referable rule

Grade ≥ 2 OR calibrated P2+P3+P4 ≥ frozen 0.7 → REFER. Threshold
preserved from Phase 5, documented, configurable but not retuned.

## 7. Confidence rule

Phase 7 calibrated confidence only (0.60 bar adapted from legacy 0.60
raw-softmax rule — provenance documented, heuristic status kept).
ROUTINE-bound + low confidence → TECHNICAL_REVIEW; REFER/URGENT keep
state and take the flag.

## 8. Evidence rule

Presence flags per lesion type (+MULTIPLE_LESION_TYPES); strong floors
(MA 5 / HE 3 / EX 3 / SE 2) for conflict. Evidence never overwrites grade.

## 9. Conflict handling

Grade ≤ 1 + sub-threshold score + strong evidence → TECHNICAL_REVIEW +
EVIDENCE_MODEL_CONFLICT by default (grade preserved); configured
patterns may escalate to REFER (default list empty — safest default).
Evaluated AFTER referable so score-referrals are never downgraded.

## 10. Reason codes

IMAGE_UNGRADABLE, RECAPTURE_REQUIRED, HIGH_DR_SEVERITY,
REFERABLE_DR_PREDICTION, lesion flags, MULTIPLE_LESION_TYPES,
LOW_MODEL_CONFIDENCE, EVIDENCE_MODEL_CONFLICT (+ESCALATED variant),
ENHANCEMENT_FAILED, INVALID_INPUT, PROCESSING_FAILURE,
NON_REFERABLE_PREDICTION. Deterministic, machine-readable.

## 11. Rule priority

INVALID_INPUT > UNGRADABLE > ENHANCEMENT_FAILED > PROCESSING_FAILURE >
HIGH_SEVERITY > REFERABLE > EVIDENCE_CONFLICT > LOW_CONFIDENCE > ROUTINE
(`src/triage/rules.py` canonical registry; config cross-checked in tests).

## 12. Safety behavior

7 flags; optional evidence (vessels/disc/fovea/lesions/Grad-CAM) fails to
warnings, never crashes; invalid input and processing errors fall back to
TECHNICAL_REVIEW; UNGRADABLE never graded.

## 13. Evaluation (APTOS test, n=550 + 30-image full-evidence subset)

Distribution: TECHNICAL_REVIEW 353 / REFER 50 / UNGRADABLE 74 /
URGENT_REVIEW 54 / ROUTINE 19. Top reasons: ENHANCEMENT_FAILED 345,
RECAPTURE_REQUIRED 419, REFERABLE 104, LOW_CONFIDENCE 97.
Subset (full lesions): 21 TECHNICAL / 6 UNGRADABLE / 2 URGENT / 1 REFER.
Safety: **0 referable false-negatives to ROUTINE, 0 severe undercalls
to ROUTINE**; 39.5% of GT-referable auto-escalated (REFER/URGENT),
60.5% human-reviewed (TECHNICAL), 0% left routine. Honest trade-off:
SAFE (nothing referable silently passed) but LOW-AUTOMATION — driven by
the strict quality gate (Phase 5 finding), usually-failing enhancement,
and the un-retuned 0.60 bar. All three are future calibration work, and
the engine exposes exactly where to intervene (reason codes).

## 14. Limitations

Thresholds heuristic; conflict floors thinly tuned; subset-only lesion
coverage at scale; no latency optimization; APTOS-only; no clinical
claims; legacy `src/rules/` system left intact (different schema) with
its 31 tests green.

## 15. Reproducibility

`APTOS_DATA_ROOT=<mirror> python -m src.triage.evaluate --subset 30`
→ `reports/triage/{metrics.json,per_image.csv,config.json}` (+ local
cards). Demo: `python -m src.triage.demo --image IMG [--fast] [--out]`.
Legacy tests preserved; new engine tests in `tests/test_triage_engine.py`
(the `test_triage.py` name belongs to the legacy suite — not overwritten).
