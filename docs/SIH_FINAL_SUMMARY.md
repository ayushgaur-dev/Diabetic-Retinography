# SIH Final Summary (SIH26038 — Phase 14)

Honest statuses only: IMPLEMENTED / PARTIALLY IMPLEMENTED /
DEMONSTRATION ONLY / NOT CLINICALLY VALIDATED (applies to everything).

| Requirement | Implementation | Evidence | Status | Limitation |
|---|---|---|---|---|
| Quality gate (G/B/U + recapture) | `src/quality/` | 22 tests; APTOS stratification | IMPLEMENTED | Focus bar too strict on real data |
| Adaptive enhancement | `src/preprocessing/` | 22 tests; ablation data | IMPLEMENTED | Rarely rescues; net-negative grading delta |
| Vessels | `src/retina/vessels/` | DRIVE F1 0.71 | IMPLEMENTED | Classical; thin-vessel recall |
| Optic disc | `src/retina/optic_disc/` | Drishti 98% @0.25DD | IMPLEMENTED | Bright-pathology hijack possible |
| Fovea | `src/retina/fovea/` | IDRiD med err 0.033 | IMPLEMENTED | Hemorrhage-mimic darkness |
| Lesions MA/HE/EX | `src/retina/lesions/` | IDRiD F1 0.08–0.23 | PARTIALLY IMPLEMENTED | Low precision; SE non-functional |
| Lesions SE | same | F1 0.000 | DEMONSTRATION ONLY | Needs learned model |
| 5-class grading | Frozen EfficientNetB0 | QWK 0.811 held-out | IMPLEMENTED | Weak minority grades |
| Referable DR | Calibrated score @0.7 | Sens 96%, spec 87% + CIs | IMPLEMENTED | In-dataset; 5.6% contamination bounded |
| Calibration | Temperature scaling | ECE 0.047→0.044 | IMPLEMENTED | Small effect; MCE(20) worsened |
| Grad-CAM | Real heatmaps + adapter | Optic-disc attention finding | IMPLEMENTED | Attribution ≠ causation |
| Explainability | Fusion + consistency + faithfulness | 10-image subset stats | IMPLEMENTED | Descriptive; ~40 s/image |
| Triage | Deterministic engine | 0 referable-FNs; 34 tests | IMPLEMENTED | 64% technical-review rate |
| Reports | JSON/MD/HTML/PDF generator | 30 tests; byte-identical | IMPLEMENTED | Template renderer; no LLM |
| Rural workflow | MATLAB DES + Simulink builder | Code + contracts; unexecuted | DEMONSTRATION ONLY | Assumptions; no runtime here |
| MATLAB integration | `matlab/` ports + JSON import | unittest suite (MATLAB-side) | IMPLEMENTED | Demo-grade; parity notes |
| Streamlit UI | Reference dashboard | Smoke-tested | IMPLEMENTED | Local/demo scope |
| Web frontend + API | Next.js + FastAPI | Build/tsc/API/E2E green | IMPLEMENTED | Local/demo scope; no auth |
| Privacy/safety | Gates A–H, no-PHI design | Safety tests; greps clean | IMPLEMENTED | Demo-deployment assumptions |

NOT CLINICALLY VALIDATED: the entire system, without exception.
