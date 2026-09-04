# SIH Demo Script (5–7 minutes, Phase 13)

Safety line (say first): "This is an AI-assisted research prototype,
not a diagnostic device. Every output needs qualified human review."

1. **Problem** (30 s): rural DR backlog; screening, not diagnosis.
2. **Upload** (30 s): fundus image in the web UI; SHA shown, no patient data.
3. **Quality** (30 s): GOOD/BORDERLINE/UNGRADABLE + dimensions; show a
   blocked UNGRADABLE example with recapture guidance.
4. **Enhancement** (30 s): BORDERLINE before/after + accepted/rejected;
   "adjusts quality, does not guarantee validity."
5. **Grading** (30 s): Grade 0–4 + raw vs calibrated probabilities.
6. **Calibration** (20 s): T=0.9542; confidence is not clinical certainty.
7. **Anatomy** (30 s): vessel map, disc + fovea overlays ("localized").
8. **Lesions** (30 s): candidate counts/boxes ("evidence, never confirmed").
9. **Grad-CAM** (30 s): attribution + consistency; "agreement is not proof."
10. **Triage** (30 s): decision, priority, reason codes verbatim.
11. **Report** (30 s): download JSON (authoritative) + PDF.
12. **Web frontend** (20 s): same pipeline, editorial UI.
13. **MATLAB/Simulink** (30 s): integration workflow + scenario simulator
    (engineering assumptions, unexecuted here).
14. **Scalability concept** (20 s): scenario design A–D, bottleneck
    thinking — no capacity claims.
15. **Safety/limitations** (30 s): uncalibrated-interpretation guards,
    dataset≠deployment, human review required.

Backup: synthetic demo endpoint (no datasets/keys needed).
