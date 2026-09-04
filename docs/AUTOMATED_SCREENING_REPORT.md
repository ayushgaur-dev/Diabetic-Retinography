# Automated Screening Report (SIH26038 — Phase 9)

## 1. Objective

Deterministic, read-only screening reports from frozen phase outputs:
JSON-authoritative with Markdown/HTML/PDF renderings of the same dict.
No inference reruns, no LLM, no network. Screening support — never diagnosis.

## 2. Input contract

`ScreeningReportInput`: image_id, quality, enhancement, grading (grade +
raw/calibrated probs + confidences), referable (score/threshold),
lesions, vessels, disc, fovea, explainability, triage, warnings.
Built by `input_adapter.adapt()` (or straight from triage dicts). No
patient metadata unless supplied — none is invented.

## 3. Report structure

Summary, quality (+UNGRADABLE notice, no trusted grade), enhancement
(applied/failed/not-required + borderline transitions), grading (both
distributions + calibrated-confidence sentence), referable (score vs
frozen 0.7 + definition), lesions (per type + candidate boxes +
limitation note), anatomy, explainability (consistency + agreement +
figure slot), triage (verbatim decision/reasons/flags/explanation),
limitations, provenance, warnings.

## 4–8. Sections

Grading shows all 5 raw + calibrated probabilities without misleading
rounding; referable uses grade≥2 / P2+P3+P4 @0.7 (threshold not claimed
validated); lesions use evidence/candidate wording with the configured
non-confirmation notice; anatomy reports statuses + confidences as
localization (not diagnostic) values; explainability reports spatial
agreement with the causation disclaimer and embeds (or notes the
absence of) the Phase 6 figure artifact.

## 9. Triage section

Verbatim Phase 8 decision/priority/reason codes/safety flags/
explanation. No reinterpretation, no second rule engine. Reason codes
translated via config map (originals preserved).

## 10. Safety language

Workflow verbs only (recapture/routine/specialist/human review); no
treatment, medication, or urgency beyond configured priority. Safety
flags surfaced (none hidden). Banned diagnostic phrases enforced by
test (`diagnosed`, `confirmed <lesion>`, positive certainty claims);
required negations ("not clinical certainty", "not clinically
confirmed") asserted present.

## 11. Provenance

Model, temperature (0.9542), triage engine version, explainability
method, deterministic generator note, report/generator versions — no secrets.

## 12. Output formats

JSON (sorted keys, byte-equivalent), Markdown, standalone offline HTML
(inline CSS, no CDN; figure base64-embedded when supplied), matplotlib
PDF (text + optional figure). One logic path; JSON authoritative.

## 13. Deterministic generation + LLM future

Same input → byte-identical JSON (tested). LLM explicitly out of scope;
documented extension point: validated fact dicts in, stylistic wording
out, with grade/probs/statuses/decisions/codes/flags locked.

## 14. Limitations

Support-only; heuristic lesions; uncalibrated-interpretation guardrails;
dataset≠deployment; review required for escalations. UNGRADABLE reports
carry no grade. INCOMPLETE (not fabricated) when required fields miss.

## 15. Reproducibility

`python -m src.reporting.demo [--input JSON | --example refer|ungradable]
[--out DIR]`. Fixture: `tests/fixtures/screening_input_example.json`
(synthetic, no PHI). Examples committed under
`reports/reporting/examples/`. Perf (fixture): generation 0.2 ms,
json/md/html ~0.1 ms, PDF ~65 ms — no model/CV reruns anywhere.
