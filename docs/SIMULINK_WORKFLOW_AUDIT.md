# Simulink Workflow Audit (SIH26038 — Phase 12)

Date: 2026-09-04. No Python/MATLAB behavior changed in this audit.

## 1. SIH telemedicine requirements

Demonstrate a rural/telemedicine screening workflow at 100,000+
patients/year scale: arrivals, quality routing, automated screening,
human review, tele-ophthalmology escalation, queues, capacity, latency,
throughput, failures/retries, scalability. Engineering simulation only.

## 2. Simulation objectives

Answer operationally: given arrival volumes and service capacities, what
are throughput, latency, utilization, queue lengths, workload, and the
dominant bottleneck? All outputs are conditional on stated assumptions.

## 3. System boundaries

IN SCOPE: patient/image arrival, capture, quality gate, enhancement, AI
grading service, triage routing, local review, tele-ophthalmology,
network delay, retry/failure accounting.
OUT OF SCOPE: the AI internals (EfficientNet, calibration, triage rules,
lesion detectors run unchanged elsewhere), clinical outcomes, diagnostic
performance, staffing costs, hardware procurement.

## 4. Inputs

Scenario struct: annual volume, operating calendar, arrival model +
seed, quality mix, enhancement accept rate, AI service-time distribution,
triage mix, reviewer/tele pools (count, hours, service times, capacity),
network delay, retry policy, queue limits.

## 5. Outputs

Per-scenario report struct: throughput (day/year), latency breakdown,
per-stage utilization, queue max/mean, failures/retries/escalations,
reviewer workload hours, bottleneck verdict. Figures + JSON-able structs.

## 6. Assumptions (ENGINEERING SCENARIO ASSUMPTIONS, not measurements)

Operating calendar 300 days x 8 h; Poisson arrivals unless stated;
review/tele service times, capture time, network delay, reviewer counts
are planning assumptions. Every assumed value is labelled ASSUMPTION.

## 7. Parameters (measured vs assumed)

MEASURED prototype values (this repo): quality ~0.01 s/image, enhancement
~0.015 s, grading inference ~0.15 s/image (Phase 5 runtimes), quality mix
GOOD 0.5% / BORDERLINE 86% / UNGRADABLE 13.5% (APTOS test), enhancement
accept ~27% of BORDERLINE (Phase 5 ablation), triage mix from Phase 8
evaluation. Everything else is an ASSUMPTION (reviewer times, staffing,
network, capture). Mixed provenance is recorded per parameter.

## 8. Limitations

- This is an engineering simulation. It is not a clinical trial, not
  evidence of clinical efficacy, and never a claim of real-world performance.
- No MATLAB/Simulink runtime exists in this environment: the DES engine,
  Simulink builder, and tests are written for execution where MATLAB
  exists; structural checks run here. No execution is pretended.
- SimEvents availability is NOT assumed: the engine is base-MATLAB code;
  the builder creates a documented Stateflow-free structural model and
  notes the SimEvents alternative.
- Duplicated/approximate rural behavior is NOT claimed; arrivals are a
  neutral Poisson/default model.
- Results never transfer to clinical capacity claims.

## 9. Relationship to MATLAB/Python

Python/MATLAB screening results are external processing results consumed
as routing probabilities (measured mixes above). `screening_struct()`
(Phase 11) remains the per-image contract; simulation uses its own
scenario/report structs and never modifies screening code.
