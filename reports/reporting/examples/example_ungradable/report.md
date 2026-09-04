# AI-Assisted Retinal Screening Report

Image: `SYNTHETIC-EXAMPLE-UNGRADABLE`
Report status: COMPLETE

## Screening Summary

Image quality: **UNGRADABLE**
Model prediction: **not assessed**
Calibrated confidence: **None**
Referable probability: **None**
Workflow recommendation: **UNGRADABLE**
Priority: **NORMAL**

## Image Quality

Status: **UNGRADABLE**
- focus: BAD
- illumination: UNKNOWN
- contrast: UNKNOWN
- exposure: UNKNOWN
- field_of_view: UNKNOWN
- retinal_coverage: UNKNOWN
- Recapture: Image appears out of focus. Keep the camera steady and recapture.
- Recapture: This image cannot be graded. Please recapture before screening.

Image quality is UNGRADABLE: no DR grade can be trusted for this image; recapture is recommended.

## Enhancement

Status: **not attempted (ungradable input)**

## DR Grading

DR grade was not assessed (grading blocked or unavailable).

## Referable Screening

Referable probability: **None**
Decision threshold: **0.7**
Screening classification: **NON_REFERABLE**
Referable DR = predicted Grade >= 2; score = P2+P3+P4.
Threshold preserved from Phase 5; not claimed clinically validated.

## Lesion Evidence

### Microaneurysm
Status: **UNKNOWN** — candidates: 0
### Hemorrhage
Status: **UNKNOWN** — candidates: 0
### Hard Exudate
Status: **UNKNOWN** — candidates: 0
### Soft Exudate
Status: **UNKNOWN** — candidates: 0

*Lesion findings are algorithmic evidence candidates and are not clinically confirmed findings.*

## Anatomical Evidence

- optic_disc: UNKNOWN
- fovea: UNKNOWN
- vessels: unavailable
Optic disc localization unavailable.
Fovea localization unavailable.
Vessel map: unavailable.

## Explainability

Consistency: **UNKNOWN**

Detected lesion evidence overlapping Grad-CAM regions is spatial agreement, not proof the lesion caused the prediction.

## Workflow Recommendation

Recommendation: **UNGRADABLE**
Priority: **NORMAL**
Image recapture recommended.
Reasons:
- IMAGE_UNGRADABLE: Image quality is insufficient for reliable automated screening.
- RECAPTURE_REQUIRED: Image recapture is recommended.
Safety flags raised: image_quality_issue

Decision: UNGRADABLE

## Limitations

- This is an AI-assisted screening tool.
- It is not a standalone diagnostic system.
- Algorithmic lesion evidence is not clinically confirmed.
- Confidence values do not represent clinical certainty.
- Dataset performance may not generalize to rural clinical deployment.
- Human review remains necessary for escalated/uncertain cases.

## Provenance

- model: EfficientNetB0 (5-class DR grading)
- calibration: Temperature scaling
- temperature: 0.9542
- triage: deterministic rule engine (evidence_triage_v1)
- explainability: Grad-CAM + explicit evidence
- generator: deterministic (no LLM, no network)
