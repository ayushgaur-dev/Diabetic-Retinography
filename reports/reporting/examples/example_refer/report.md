# AI-Assisted Retinal Screening Report

Image: `SYNTHETIC-EXAMPLE-001`
Report status: COMPLETE

## Screening Summary

Image quality: **BORDERLINE**
Model prediction: **Moderate (Grade 2)**
Calibrated confidence: **0.81**
Referable probability: **0.83**
Workflow recommendation: **REFER**
Priority: **HIGH**

## Image Quality

Status: **BORDERLINE**
- focus: BORDERLINE
- illumination: UNKNOWN
- contrast: UNKNOWN
- exposure: UNKNOWN
- field_of_view: GOOD
- retinal_coverage: UNKNOWN

## Enhancement

Status: **applied**
Operations: clahe, denoise
Quality: BORDERLINE → GOOD
Enhancement adjusts image quality; it does not change the screening assessment.

## DR Grading

- Grade 0 (No DR): raw 0.0600 / calibrated 0.0500
- Grade 1 (Mild): raw 0.1300 / calibrated 0.1200
- Grade 2 (Moderate): raw 0.7200 / calibrated 0.7400
- Grade 3 (Severe): raw 0.0700 / calibrated 0.0700
- Grade 4 (Proliferative): raw 0.0200 / calibrated 0.0200
Predicted grade: **2 — Moderate**
Calibrated model confidence: 0.8100. Calibrated confidence is not clinical certainty.

## Referable Screening

Referable probability: **0.83**
Decision threshold: **0.7**
Screening classification: **REFERABLE**
Referable DR = predicted Grade >= 2; score = P2+P3+P4.
Threshold preserved from Phase 5; not claimed clinically validated.

## Lesion Evidence

### Microaneurysm
Status: **NOT_DETECTED** — candidates: 0
### Hemorrhage
Status: **DETECTED** — candidates: 4
- bbox [200, 200, 260, 240], score 0.8
### Hard Exudate
Status: **DETECTED** — candidates: 3
- bbox [100, 100, 140, 130], score 0.7
### Soft Exudate
Status: **NOT_DETECTED** — candidates: 0

*Lesion findings are algorithmic evidence candidates and are not clinically confirmed findings.*

## Anatomical Evidence

- optic_disc: DETECTED
- fovea: DETECTED
- vessels: available
Optic disc: detected (localization confidence 0.74).
Fovea: detected (localization confidence 0.66).
Vessel map: available.

## Explainability

Consistency: **SUPPORTIVE**
high spatial agreement (overlap 0.62)
- hemorrhage: spatial agreement 0.62
Detected lesion evidence overlapping Grad-CAM regions is spatial agreement, not proof the lesion caused the prediction.

## Workflow Recommendation

Recommendation: **REFER**
Priority: **HIGH**
Specialist review recommended.
Reasons:
- REFERABLE_DR_PREDICTION: Predicted DR severity meets the referable screening criterion.
- HEMORRHAGE_EVIDENCE: Hemorrhage candidate regions were detected by the lesion-evidence module.
- HARD_EXUDATE_EVIDENCE: Hard-exudate candidate regions were detected by the lesion-evidence module.
Safety flags raised: image_quality_issue

Decision: REFER

## Limitations

- This is an AI-assisted screening tool.
- It is not a standalone diagnostic system.
- Algorithmic lesion evidence is not clinically confirmed.
- Confidence values do not represent clinical certainty.
- Dataset performance may not generalize to rural clinical deployment.
- Human review remains necessary for escalated/uncertain cases.

## Warnings

- BORDERLINE_IMAGE_ENHANCED: graded enhanced input.

## Provenance

- model: EfficientNetB0 (5-class DR grading)
- calibration: Temperature scaling
- temperature: 0.9542
- triage: deterministic rule engine (evidence_triage_v1)
- explainability: Grad-CAM + explicit evidence
- generator: deterministic (no LLM, no network)
