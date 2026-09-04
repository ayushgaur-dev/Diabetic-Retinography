# Simulink Telemedicine Workflow (SIH26038 — Phase 12)

## System architecture

Event-calendar discrete-event simulation (base MATLAB, no SimEvents
required) plus a programmatic Simulink structural model:

```
PATIENT ARRIVAL (Poisson/deterministic, seeded)
  -> IMAGE CAPTURE (server pool)
  -> QUALITY GATE (GOOD/BORDERLINE/UNGRADABLE scenario mix)
  -> ENHANCEMENT delay (BORDERLINE accepted fraction) | RECAPTURE loop
  -> AI SCREENING service (measured grading runtime + margin)
  -> TRIAGE routing (measured mix; UNGRADABLE never reaches here)
  -> ROUTINE completion | REVIEW queue (urgent priority)
  -> TELE-OPHTHALMOLOGY escalation | completion
Failures (AI errors, overflow, double quality failure) counted, never clinical.
```

This is an ENGINEERING SIMULATION. It is NOT a clinical trial, NOT
evidence of clinical efficacy, and NEVER a claim of real-world rural
performance or guaranteed capacity. Every number below is conditional
on the stated assumptions.

## Simulink model structure

`build_simulink_model.m` generates `telemedicine_workflow.slx` with 10
subsystems (01 Patient Arrival … 10 Metrics), wired in workflow order.
Subsystems document their engine counterparts; MATLAB Function routing
is added by hand after generation. SimEvents is NOT assumed — the file
notes the Entity Generator/Server/Terminator alternative for licensed
environments. The executable simulation of record is `simulate_workflow.m`.

## Assumptions (ENGINEERING SCENARIO ASSUMPTIONS)

Calendar 300 days x 8 h; Poisson arrivals (deterministic option);
capture 120 s; review 240 s; tele 600 s + 30 s network; tele escalation
0.4/0.8/0.2 by class; AI failure 0.001 with one retry; one recapture
then fail; finite queues (overflow counted, not silently dropped);
servers always staffed within the horizon (overtime allowed past H).

## Scenario parameters

Presets A/B/C/D = 10k/50k/100k/150k patients/year (`create_scenario`).
Arrival rate = annual / operating seconds (e.g. 100k -> 0.01157/s, mean
interarrival 86.4 s). Seed recorded per scenario (default 26038).

## Arrival model

Poisson (exponential interarrivals via `-log(rand)`, base MATLAB — NOT
`exprnd`, which needs the Statistics Toolbox) or deterministic grid.
Neutral default; never presented as rural patient behavior.

## Service-time model

Exponential (default) or deterministic; AI grading mean 0.5 s (measured
~0.15 s inference + margin, labelled measured); quality/enhancement are
unqueued delays (10/20 ms measured scale); review/tele/capture/network
are assumptions.

## Capacity model

Server pools (default capture 2, AI 4, review 2, tele 1) with finite
FIFO queues (urgent front-insertion at review/tele). Utilization =
busy_time / (servers x horizon). Reviewer/tele hours reported.

## Triage routing

Measured Phase 8 mix renormalized among graded images (ROUTINE 19,
REFER 50, URGENT 54, TECHNICAL 353 of 476); REFER/TECHNICAL queue for
review, URGENT with priority, ROUTINE completes locally. The Python
triage algorithm is NOT recreated — proportions are scenario inputs.

## Human review model

Operational workload only (no clinician-performance claims): reviewer
pool, mean review time, availability via server count, queue capacity.

## Tele-ophthalmology model

Rural site -> remote queue -> ophthalmology review -> outcome; capacity,
response time, hours, queue capacity configurable; network delay added
to tele service. Assumptions throughout.

## Failure/retry model

Quality failure -> recapture once -> fail; enhancement reject -> same;
AI error -> one retry -> fail; queue overflow -> counted drop -> fail.
Retries, drops, failures, escalations all reported.

## Metrics

Throughput (/day, /year), latency (mean/p95 end-to-end + per-stage queue
waits), utilization per pool, queue max/mean (+ stride-capped time
series), failures/retries/overflow, reviewed/tele counts, reviewer
hours, bottleneck (max-utilization pool). Exact queue max/mean come from
running counters, not sampled estimates.

## Scenarios

Run `run_all_scenarios` (A-D). Report language (enforced in code
comments and docs): "Under the specified engineering assumptions, the
simulated workflow processed X cases/year with Y utilization and Z
queueing behavior." Per-scenario results are written by execution in
MATLAB (unavailable in the authoring environment — see below), never
prefilled here.

## Bottleneck analysis

`collect_metrics` names the max-utilization pool among capture/AI/
review/tele-ophthalmology with its utilization; plots compare scenarios.

## Limitations

- MATLAB/Simulink was unavailable where this was authored: engine,
  builder, and `matlab.unittest` suite are written for execution in
  MATLAB; Python structural checks verify contracts only. Execution
  outputs are NOT included or pretended.
- Mixed measured/assumed provenance (see audit); staffing-side values
  are the dominant uncertainty.
- Overloaded scenarios grow queues without bound within the horizon —
  the model reports this honestly (waits, overflow) instead of hiding it.
- No clinical content: patients are tokens with timestamps, and the
  safety model of Phases 2-9 is preserved (nothing here diagnoses).

## Reproducibility

Seed in every scenario config; `run_scenario` saves the config snapshot
with results; deterministic reruns reproduce bit-identical reports
(tested). Record MATLAB/toolbox versions when running.

## Exact run instructions

```matlab
cd matlab
startup
rep = run_scenario(create_scenario('C'));   % 100k/year
reps = run_all_scenarios();                  % A-D + figures
run_all_tests                                % matlab.unittest suite
build_simulink_model('telemedicine_workflow') % .slx (needs Simulink)
```
