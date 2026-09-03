# ICDRSS Clinical Guidelines — Retinal Screening Triage System

This document provides the clinical reference corpus for the Gen AI 
report-generation component of this system. Grade definitions follow 
the International Clinical Diabetic Retinopathy (ICDR) Severity Scale 
(Wilkinson et al., 2003). All other content — follow-up intervals, 
referral criteria, image quality standards, and clinical notes — is 
authored from 15 years of clinical experience in ophthalmic nursing.

---

## Screening Context

**When screening begins**:
- Type 2 diabetes: from diagnosis, as retinopathy may already be 
  present subclinically for years before diagnosis.
- Type 1 diabetes: from 5 years post-diagnosis, or at puberty if 
  puberty occurs earlier.

**Pregnancy**: diabetic patients who become pregnant require more 
frequent screening (approximately each trimester), as pregnancy can 
accelerate retinopathy progression. This system's synthetic intake 
parameters do not currently capture pregnancy status — noted as a 
limitation.

---

## Grade Definitions (ICDRSS)

This system classifies fundus images into five grades of diabetic 
retinopathy severity, following the International Clinical Diabetic 
Retinopathy (ICDR) Severity Scale (Wilkinson et al., 2003), the most 
widely used classification system in clinical practice.

### Grade 0 — No Diabetic Retinopathy
Normal retina, no visible lesions. Clinically the most straightforward 
grade to distinguish from all others.

### Grade 1 — Mild Non-Proliferative DR (NPDR)
Microaneurysms only, with no other lesions present.

### Grade 2 — Moderate Non-Proliferative DR (NPDR)
Microaneurysms plus retinal haemorrhages and/or hard exudates, without 
meeting the criteria for severe NPDR.

**Clinical note**: distinguishing Grade 1 from Grade 2 is one of the more 
challenging boundaries in practice, requiring careful assessment of 
lesion count and distribution rather than their mere presence.

### Grade 3 — Severe Non-Proliferative DR (NPDR)
Any of: more than 20 intraretinal haemorrhages in all four quadrants, 
definite venous beading in two or more quadrants, or prominent 
intraretinal microvascular abnormalities (IRMA) in one or more 
quadrants — with no signs of proliferative disease.

### Grade 4 — Proliferative DR (PDR)
Neovascularisation and/or vitreous/pre-retinal haemorrhage. Represents 
the highest-risk grade for imminent vision loss.

**Clinical note**: distinguishing Severe NPDR (Grade 3) from early 
Proliferative DR (Grade 4) can also be subtle in practice, as both 
represent advanced disease along a continuum.

---

**Note on scope**: this scale addresses diabetic retinopathy severity 
only. Diabetic macular oedema (DME) is graded independently (M0/M1/M2 in 
the full ICDR system) and requires three-dimensional assessment (OCT or 
stereo fundus photography) that a single 2D fundus image cannot reliably 
provide — see the Macular Oedema section below.

---

## Characteristic Findings by Grade

### Grade 0 — No DR
No microaneurysms, haemorrhages, exudates, or vascular abnormalities. 
Normal vessel calibre and course.

### Grade 1 — Mild NPDR
Microaneurysms: small, round, red dots caused by capillary wall 
outpouching. The earliest visible sign of diabetic retinopathy.

### Grade 2 — Moderate NPDR
Microaneurysms plus one or more of:
- Retinal (dot/blot) haemorrhages
- Hard exudates (yellow-white lipid deposits, sharply demarcated)
- Cotton wool spots (soft, ill-defined white patches indicating 
  microinfarction)

None of these findings yet meet the quantitative criteria for severe NPDR.

**Clinical note on hard exudates**: significance depends heavily on size 
and location, not just presence. Large exudates, or exudates located 
close to the macula, raise concern for diabetic macular oedema (DME) — a 
sight-threatening complication that is graded and managed independently 
of the DR severity grade itself. In practice, exudates near the macula 
were a common trigger for referral regardless of the overall DR grade. 
See "Macular Oedema and Fundus Photography Limitations" below for how 
this system handles that limitation.

### Grade 3 — Severe NPDR
Any one of the "4-2-1 rule" criteria:
- More than 20 intraretinal haemorrhages in each of all 4 quadrants
- Definite venous beading in 2 or more quadrants
- Prominent intraretinal microvascular abnormalities (IRMA) in 1 or 
  more quadrants

No neovascularisation or vitreous/pre-retinal haemorrhage present.

### Grade 4 — Proliferative DR
One or both of:
- Neovascularisation (new, abnormal vessel growth on the retina or 
  optic disc)
- Vitreous or pre-retinal haemorrhage

Represents active, sight-threatening disease requiring urgent intervention.

---

## Recommended Follow-up Intervals

| Grade | Description | Follow-up interval (this system) |
|---|---|---|
| 0 | No DR | 12 months |
| 1 | Mild NPDR | 6 months |
| 2 | Moderate NPDR | 6 months |
| 3 | Severe NPDR | 1 month (urgent referral) |
| 4 | Proliferative DR | 1 month (urgent referral) |

These intervals reflect standard screening practice for Grades 0-2 
(annual review for no retinopathy, six-monthly for mild-to-moderate 
disease).

**Note on Grade 3 interval**: standard screening protocols typically 
recommend 3-4 month follow-up (or referral) for severe NPDR. This system 
applies a stricter 1-month interval by design: as a virtual clinic whose 
core purpose is to minimise time from detection to specialist 
intervention, and given the real risk of progression from severe NPDR 
to proliferative disease without timely treatment, a shorter interval 
was chosen as an intentional safety margin over the standard screening 
protocol.

**Note**: these intervals apply to the baseline DR grade only. Individual 
patient risk factors (glycaemic control, diabetes duration) may shorten 
the recommended interval further — see patient risk modifiers in the 
triage rule engine (`src/rules/triage.py`). Additional risk factors 
recognised in clinical practice — hypertension, renal function — are 
not currently modelled and are noted as future work.

---

## Urgent Referral Criteria

The following findings warrant urgent ophthalmology referral regardless 
of the calculated DR grade, as they indicate active or imminent 
sight-threatening pathology:

- **Neovascularisation or vitreous/pre-retinal haemorrhage** (defining 
  features of Grade 4 — always urgent)
- **Extensive intraretinal haemorrhage** (Grade 3 criteria) — high risk 
  of progression to proliferative disease within a short timeframe
- **Large hard exudates, or exudates located near the macula, in 
  isolation** — clinically significant enough to indicate at least 
  Grade 3-level concern for diabetic macular oedema (DME), even without 
  meeting the full haemorrhage/IRMA/venous beading criteria
- **Large or macular exudates combined with other severe-grade 
  findings** (extensive haemorrhage, venous beading, IRMA) — indicates 
  a more advanced and urgent presentation than either sign alone
- **Sudden visual acuity loss reported by the patient**, even with a 
  low or borderline DR grade on imaging — may indicate an acute event 
  (e.g. vitreous haemorrhage, macular involvement) not fully captured 
  by a single static image
- **Poor image quality preventing confident grading** — see Image 
  Quality Criteria below; an ungradable image is not equivalent to a 
  normal one and should not default to routine follow-up

**Design implication**: this system's abstention logic (confidence 
threshold, see rule engine) is intended to route uncertain or 
borderline cases toward clinician review rather than defaulting to a 
conservative low-risk classification, consistent with this referral 
philosophy.

---

## Image Quality Criteria

Reliable grading requires a fundus image of sufficient technical quality.

### Acquisition protocol context
- **Pupillary dilation (mydriasis)**: pharmacological dilation is 
  recommended to maximise image quality and field of view, especially 
  in population screening. Non-mydriatic capture exists but carries a 
  higher rate of ungradable images (10-20%), particularly with small 
  pupils, cataracts, or poor patient cooperation.
- **Field of view**: clinical screening typically uses 1-2 field 
  protocols (disc- and macula-centred, 45-50°), rather than the 
  7-field ETDRS research standard. This system assumes single-field, 
  macula/disc-centred capture, consistent with the training dataset.

### Exposure
- **Under-exposed images**: retinal structures are difficult to 
  distinguish against a dark background; microaneurysms and small 
  haemorrhages are especially likely to be missed.
- **Over-exposed images**: washed-out appearance may obscure subtle 
  lesions, particularly hard exudates.

### Framing and field of view
- **Incomplete field of view**: the optic disc and macula should both 
  be visible; images cropped to exclude these landmarks cannot be 
  reliably graded.
- **Non-standard aspect ratio or shape**: substantially cropped or 
  irregularly-shaped captures may indicate non-standard acquisition and 
  warrant recapture.

### Artefacts
- **Obstructions**: eyelashes, eyelid shadows, or reflections covering a 
  substantial portion of the retina.
- **Motion blur or poor focus**: fine lesions are the first features 
  lost to blur, which can artificially lower the apparent severity grade.

### System behaviour on poor-quality images
This system's quality gate (Stage 0) flags images meeting the above 
criteria before they reach the grading model. An ungradable image should 
always route to recapture or in-person evaluation — never to a default 
"no DR" classification, consistent with standard screening practice 
(poor quality images are referred for direct/indirect ophthalmoscopy, 
not assumed normal).

**Empirical note**: manual inspection of the training dataset (see EDA, 
`notebooks/01_data_download_eda.ipynb`) confirmed meaningful real-world 
variability in exposure (luminance range 19.6–106.0 on a 0-255 scale), 
framing, and image geometry — reinforcing that this is not a theoretical 
concern but an observed characteristic of routine fundus photography.

---

## Macular Oedema and Fundus Photography Limitations

### What diabetic macular oedema (DME) is
Diabetic macular oedema is retinal thickening at or near the macula, 
caused by fluid leakage from damaged vessels. It is graded 
independently of DR severity (M0: no oedema; M1/M2: oedema present, 
graded by proximity to the fovea) and can occur at any DR grade, 
including Grade 0-1, where overall retinopathy severity appears mild.

### Why this matters for grading
DME is one of the leading causes of vision loss in diabetic patients, 
and its presence does not correlate simply with DR severity — a patient 
with mild background retinopathy can have sight-threatening macular 
oedema, while a patient with more extensive peripheral retinopathy may 
have a healthy macula. This is why hard exudates near the macula 
warrant referral regardless of the calculated DR grade (see Urgent 
Referral Criteria).

### The fundamental limitation of 2D fundus photography for DME
Reliable DME assessment requires evaluating retinal **thickness**, which 
is a three-dimensional property. Standard 2D colour fundus photography 
can show secondary signs suggestive of DME (hard exudates near the 
fovea) but cannot directly measure retinal thickening. Confident DME 
diagnosis requires:
- **Optical Coherence Tomography (OCT)**, providing cross-sectional 
  retinal imaging, or
- **Stereo fundus photography / slit-lamp biomicroscopy**, providing 
  3D assessment through dilated examination

In routine clinical practice, diabetic and high-myopia patients are 
typically assessed with fundus photography **and** OCT together, 
alongside visual acuity, refraction history (for myopic patients), and 
metabolic markers (HbA1c, medication) — a multimodal evaluation, not a 
single image in isolation.

### How this system handles the limitation
This system's vision model is trained exclusively on 2D fundus 
photography (APTOS 2019 dataset) and grades DR severity only. It does 
**not** assess macular status. Every generated report includes an 
explicit field, `macular_status_assessed: false`, to prevent the 
absence of a DME finding from being misread as a negative finding. 
Suspicion of DME based on secondary signs (large or macular hard 
exudates) is flagged for referral regardless of DR grade, but confirmed 
DME diagnosis remains outside this system's scope and requires OCT or 
equivalent assessment by the reviewing clinician.

**This is a deliberate scope boundary, not an oversight**: OCT 
integration is noted as future work in the project brief, given dataset 
and time constraints for the current implementation.
