# Final Project Brief — Retinal Screening Triage System

**Student:** Manu | **Cohort:** June 26 | **Duration:** 7 days
**Status:** Awaiting approval

---

## 1. Business Problem

Diabetic patients are increasingly managed through **virtual clinics**: the patient attends a short "express" visit where visual acuity, fundus photography and OCT scans are captured, and then goes home. An ophthalmologist reviews the images afterwards and decides whether the patient needs an in-person appointment or can continue on virtual follow-up at 3, 6 or 12 months.

The bottleneck is **not image acquisition — it is the review queue**. A large share of reviewed images are normal or mild cases that consume specialist time without changing patient management.

**Objective:** build a decision-support system that triages the review queue, flags urgent cases, and drafts a structured pre-report for the ophthalmologist to validate.

> **Scope statement:** This is a *triage assistant*, not a diagnostic device. A production version would qualify as a medical device under EU MDR and as a high-risk AI system under the EU AI Act. Every output requires clinician validation. This constraint is treated as a design requirement, not a disclaimer.

**Domain background:** I have 15 years of clinical experience in ophthalmic nursing, including direct work in retinal imaging workflows. This informs the clinical logic, the safety thresholds and the reference guideline corpus used in this project.

---

## 2. Dataset

**Primary — APTOS 2019 Blindness Detection (Kaggle)**
`https://www.kaggle.com/c/aptos2019-blindness-detection`

| Attribute | Value |
|---|---|
| Modality | Fundus photography |
| Volume | 3,662 labelled images |
| Labels | 5 ordinal grades (ICDRSS): No DR, Mild, Moderate, Severe, Proliferative |
| Class distribution | 1,805 / 370 / 999 / 193 / 295 |
| Source | Aravind Eye Hospital, graded by trained clinicians |
| Licence | Non-commercial / research use. Images will **not** be redistributed in the repository — only a download script |

**Why this dataset:** the labels form an *ordinal severity scale*, which maps directly onto follow-up interval assignment. Diagnostic-category datasets (e.g. Kermany OCT2017) do not support this mapping.

**Known limitations — acknowledged explicitly:**

- `train.csv` contains only two columns: `id_code` and `diagnosis`. There is **no patient metadata and no patient identifier**.
- Because images cannot be grouped by patient, patient-level splitting is impossible. Splits will be stratified by grade, and the potential for same-patient leakage between train and validation is reported as a limitation.
- Severe class imbalance (193 images in grade 3 vs 1,805 in grade 0), affecting precisely the grades that carry the highest clinical cost.

**Demonstration metadata (synthetic).** A small synthetic table (age, years since diabetes diagnosis, HbA1c, visual acuity, spherical refraction) will be generated **solely to demonstrate the triage rules and report format end-to-end**. It is explicitly labelled as synthetic in code, README and presentation, and is **never used to train, tune or evaluate any model**. Its only role is to stand in for data that, in a real deployment, comes from the electronic health record and from measurements taken during the visit. See Appendix A.

**Reference corpus for the Gen AI component** — a curated clinical guideline document (ICDRSS grade definitions, characteristic findings, follow-up intervals, urgent referral criteria, image quality criteria), authored by me from domain knowledge and public clinical guidelines.

---

## 3. Training Data vs Inference Data

A deliberate architectural distinction, stated up front because it determines what the dataset must and must not provide:

| | Training (offline, once) | Inference (per patient) |
|---|---|---|
| **Source** | APTOS 2019 | Retinal camera + EHR + nurse-recorded measurements |
| **Inputs** | Labelled fundus images only | Fundus image **+** patient parameters entered at intake |
| **Consumed by** | Vision model | Vision model (image) + rule engine (parameters) |

The vision model learns a single mapping — **pixels to ICDRSS grade** — and requires nothing but labelled images. Patient parameters are never learned from the dataset; they are supplied at inference time through the intake form, exactly as they are recorded during a real express visit. This mirrors clinical reality: the model is trained on a reference image set and then applied to patients who were never part of it.

---

## 4. Proposed Architecture

```
Fundus image  +  patient parameters (intake form)
        |
[0] Quality gate — is the image usable?
        |
[1] Vision model — EfficientNetB0 (transfer learning)
    -> ICDRSS grade 0-4 + confidence + Grad-CAM heatmap
        |
[2] Rule engine — DETERMINISTIC, AUDITABLE
    combines predicted grade + confidence + patient parameters
    -> action + follow-up interval, or ABSTAIN -> human review
        |
[3] RAG — retrieve guideline passages relevant to the decision
    (sentence-transformers embeddings + FAISS vector store)
        |
[4] LLM — draft the structured report from the decision + retrieved
    guidelines. The LLM DRAFTS; it never decides the interval.
        |
FastAPI endpoint + Streamlit demo UI
```

**Design rationale for the rule engine.** The follow-up interval is produced by explicit, guideline-derived rules rather than a learned model. Two reasons: (a) no public fundus dataset carries the required outcome label — the ophthalmologist's actual scheduling decision, or subsequent progression; (b) in a regulated clinical setting the scheduling logic must be reviewable line by line by a clinician. The learned component is confined to what the image can support; the clinical decision remains explicit and traceable.

**Patient parameters as risk modifiers.** Age, diabetes duration, HbA1c, visual acuity and spherical refraction act as modifiers within the rule engine — for example, poor metabolic control shortening the interval for a mild grade, or high myopia triggering an additional surveillance note. They are not classification targets.

**Modularity.** Each stage is an independent module. If a later stage fails, earlier stages still deliver a working system.

**Output schema** includes: predicted grade, confidence, findings, guideline applied, recommended action, follow-up interval in months, human-review flag, and the IDs of the retrieved source chunks (audit trail).

---

## 5. Coverage of Project Requirements

| Requirement | How it is addressed |
|---|---|
| **Data collection** | APTOS 2019 via Kaggle API; synthetic intake generator; hand-authored guideline corpus |
| **Data preparation** | Image resizing/normalisation, quality filtering, class-imbalance handling (class weights + targeted augmentation), stratified splits |
| **EDA** | Class distribution, image resolution and luminance analysis, quality assessment, per-class visual inspection. **Tableau dashboard** covering class balance, image quality and model performance |
| **ML / Deep Learning** | Baseline CNN from scratch -> EfficientNetB0 transfer learning. Grad-CAM explainability. Justified model selection |
| **Gen AI** | RAG pipeline (embeddings + vector store + retrieval) feeding an LLM that drafts structured clinical pre-reports grounded in retrieved guidelines |
| **GitHub repo** | Modular source package, exploration notebooks, tests, Dockerfile, requirements.txt, documented README |
| **Project planning** | Trello board, daily commits |
| **Presentation** | 15-20 min deck plus recorded demo video |

---

## 6. Evaluation Metrics

| Component | Metric | Rationale |
|---|---|---|
| Vision | **Quadratic Weighted Kappa** | Official competition metric; penalises distant errors more heavily, matching the clinical cost structure |
| Vision — safety | **Recall on grades 3-4** | A missed severe case is the highest-cost failure mode |
| Triage | **% of queue safely automated** | The operational KPI |
| Retrieval | **Hit rate @ k** on a 15-20 query test set | Verifies the RAG retrieves the correct guideline passage |
| Generation | **Faithfulness** | Checks the report asserts nothing absent from the retrieved context |

The central analytical result will be the **trade-off curve between queue automation rate and severe-case recall**, with an explicit, clinically justified operating threshold.

---

## 7. Deliverables

- GitHub repository (modular code, README, requirements.txt, Dockerfile)
- Tableau dashboard
- FastAPI endpoint + Streamlit demo interface
- Recorded demo video
- Final presentation slides

---

## 8. Timeline

| Day | Focus | Freeze point |
|---|---|---|
| 1 | Repo scaffold, data download, initial EDA | — |
| 2 | Preprocessing, baseline CNN, EDA complete | EDA frozen |
| 3 | Transfer learning, Grad-CAM, rule engine | Vision model frozen |
| 4 | RAG: corpus indexing, retrieval, report drafting | — |
| 5 | RAG evaluation, FastAPI, Streamlit app, Docker | **All code frozen** |
| 6 | Tableau dashboard, README, documentation, demo recording | — |
| 7 | Slides, rehearsal | — |

The rule engine is scheduled before the RAG stage because stages [3] and [4] consume its output. The Tableau dashboard is scheduled after model training because it reports model performance alongside dataset characteristics.

---

## 9. Risks and Contingencies

| Risk | Mitigation |
|---|---|
| RAG underperforms | Replace retrieval with deterministic grade-to-guideline lookup; LLM drafting retained |
| Low QWK due to class imbalance | Collapse to binary task (referable vs non-referable DR) — clinically valid and better supported by the data |
| FastAPI/Docker complications | Streamlit-only demo; deployment documented rather than containerised |

---

## 10. Out of Scope

Deliberately excluded, with reasons:

- **Myopic branch.** The target virtual clinic also manages high-myopia patients, but APTOS carries no myopic maculopathy labels. Refraction is therefore used only as a risk modifier in the rule engine, never as a classification target. Grading myopic maculopathy is out of scope and flagged as future work.
- **OCT modality.** A genuinely multimodal system (fundus + OCT) would require a second model and a second dataset. Flagged as future work.
- **Real patient metadata.** BRSET (16,266 fundus images from 8,524 patients, with demographics, insulin use and diabetes duration alongside ICDR grading) would support a true image-plus-tabular fusion model. It is hosted on PhysioNet under credentialed access requiring CITI human-subjects research training and a signed data use agreement; approval timelines are incompatible with a seven-day project. Credentialing will be requested in parallel for future work.
- **Longitudinal progression modelling** and **prospective clinical validation.**

---

## Appendix A — Synthetic Intake Data

### How it will be generated

A dedicated generator module, kept separate from all model code and based on explicit rules rather than naive random sampling:

- **Clinically plausible distributions** rather than uniform ranges — e.g. age ~ N(60, 12) bounded to 18–90; HbA1c ~ N(7.8, 1.4) bounded to 5.0–14.0; spherical refraction drawn from a bimodal distribution reflecting emmetropic and myopic subpopulations.
- **Internal consistency constraints** — e.g. diabetes duration cannot exceed a plausible age at onset; high myopia (≤ −6.00 D) flags the myopic surveillance branch.
- **Fixed random seed** for reproducibility.

### Safeguards

1. The generator lives in its own module, isolated from the training pipeline, and is headed by an explicit notice: `SYNTHETIC DATA — DEMONSTRATION ONLY. NOT USED FOR TRAINING OR EVALUATION.`
2. The output file is clearly named to identify it as synthetic demonstration data.
3. Its synthetic nature is stated in the README, in the demo interface and in the final presentation.
4. It is **never** used to train, tune or evaluate any model.

### Why it must not be used for training

Two independent reasons. If parameters are generated independently of the image label, they carry no signal and any fusion model trained on them would learn noise, making its reported metrics meaningless. If they were instead conditioned on the true label to make demo cases clinically coherent, this would constitute **label leakage** — the target encoded in the features, producing artificially inflated performance.

Either way, the data is valid only as a functional stand-in for inputs that a real deployment would receive at inference time. The learned component of the system remains trained exclusively on APTOS fundus images.

### Question for approval

> Is the use of clearly-labelled synthetic intake data — for demonstration purposes only, never for training or evaluation — acceptable for this project?
