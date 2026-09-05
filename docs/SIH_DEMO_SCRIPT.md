# SIH Demo Script (5–7 minutes, Phase 14)

Open with: "AI-assisted research prototype, not a diagnostic device.
Every output needs qualified human review."

| Time | Click | Should appear | Say | Concept | Do NOT claim |
|---|---|---|---|---|---|
| 00:00 | — (title) | Landing page | Rural backlog; screening, not diagnosis | Problem | Clinical need as fact |
| 00:30 | Start screening → Choose image | Preview + hash | "No patient data; content-hashed caching" | Privacy/integrity | Anonymization proof |
| 00:45 | Analyze | Quality panel | Dimensions + verdict; show UNGRADABLE example second | Safety gate | "AI sees quality like doctors" |
| 01:05 | — | Before/after, accepted/rejected | "Adjusts quality; cannot fix unusable images" | Enhancement limits | Guaranteed rescue |
| 01:30 | — | Grade + raw vs calibrated table | "Model predicted, not diagnosed" | Grading | Certainty |
| 02:00 | — | 0.74 confidence + disclaimer | "Probability representation, not clinical certainty" | Calibration | Trustworthiness |
| 02:20 | — | Vessel/disc/fovea overlays | "Localized", landmark wording | Anatomy | Precision |
| 02:45 | — | Counts + boxes | "Candidates, never confirmed" | Lesion evidence | Detections = disease |
| 03:15 | — | Grad-CAM + SUPPORTIVE | "Agreement is not proof" | Attribution | Causation |
| 03:45 | — | Decision + reasons verbatim | "Workflow recommendation from frozen rules" | Triage | Diagnosis/action orders |
| 04:10 | Generate → Download JSON | Report + provenance | "JSON authoritative; deterministic" | Reporting | LLM-quality prose |
| 04:30 | — (architecture) | Backend/frontend split | "Frontend never computes clinical values" | Web architecture | Production readiness |
| 05:00 | — (MATLAB/Simulink) | Workflow code + scenarios | "Engineering assumptions; unexecuted here" | Simulation honesty | 100k capacity |
| 05:30 | — (limitations) | Safety strip | "Dataset results ≠ deployment; review required" | Honesty | Deployment validity |

Fallbacks: synthetic demo button (no datasets/keys); UNGRADABLE fixture
for the recapture path; committed example reports if live run stalls.
