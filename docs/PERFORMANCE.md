# Performance Audit (SIH26038 — Phase 13)

Measured on a real APTOS image (2136×3216, CPU-only Windows 11 dev
laptop, full evidence workup): **total ≈ 89 s**.

| Stage | Time | Note |
|---|---|---|
| Quality | 0.2 s | 224px reference |
| Enhancement | 0.45 s | BORDERLINE path |
| Grading | 20.8 s first call (≈ 2–4 s steady) | TF graph compile on first predict |
| Calibration | < 1 ms | array math |
| Evidence (vessels/disc/fovea/lesions/Grad-CAM) | ≈ 67 s | dominant cost; lesion pipeline ~9 s + fovea/disc/vessels + explainability |
| Triage | 0.04 s | pure rules |
| Reporting | < 0.1 s (PDF ~65 ms) | no reruns |

Slowest stages: lesion evidence, fovea/disc localization, Grad-CAM
context — all CPU classical/TF passes at high resolution. No algorithm
changed for speed; the documented lever is working-resolution policy
(never at the cost of MA sensitivity without re-validation).

These are CPU prototype runtimes. They are not clinical or deployment
performance figures.
