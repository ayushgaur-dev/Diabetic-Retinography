# Project Status (SIH26038 — Phase 14, final)

## WHAT WORKS

End-to-end screening on real images: quality gate, enhancement,
5-class grading, temperature calibration, vessels/disc/fovea, lesion
evidence, Grad-CAM fusion, deterministic triage, JSON/MD/HTML/PDF
reports — via Streamlit, FastAPI+Next.js, and MATLAB import paths.

## WHAT IS TESTED

343 Python tests green (unit/integration/contract/safety); tsc clean;
`next build` clean; 10 API contract tests; live API + frontend E2E
verified; MATLAB suites written for MATLAB execution + 16 structural
checks here.

## WHAT IS DEMO-ONLY

10-image explainability subset; synthetic fixtures; landing-page metric
cards (real held-out numbers, labelled); MATLAB ports; demo scripts.

## WHAT IS SIMULATION-ONLY

Phase 12 DES + Simulink builder: code and contracts, unexecuted in this
environment. No outputs prefilled, none claimed.

## WHAT IS NOT CLINICALLY VALIDATED

Everything. No diagnosis, no deployment validity, no capacity claims.

## WHAT REAL DEPLOYMENT WOULD REQUIRE

Recalibrated quality gate; prospective multi-site validation;
regulatory clearance; clinical workflow integration and staffing;
monitoring/drift detection; security hardening beyond demo scope;
usability studies with target users. Years of work — stated, not hidden.

## UNRESOLVED RISKS

Heuristic thresholds; thin tuning sets; single-region datasets; 5.6%
test contamination (bounded); low triage automation by design; 40–90 s
CPU screenings; staffing-side simulation assumptions; Streamlit/Next.js
are demo-grade hosts.
