# Retinal Screening Triage System

## Problem

Virtual diabetic-patient clinics generate a growing backlog of fundus
photographs awaiting ophthalmologist review. This system assists that
review queue: it classifies each image by diabetic retinopathy severity
(ICDRSS grade 0-4), applies a deterministic clinical rule engine to
propose a triage action and follow-up interval, and uses a RAG-grounded
LLM to draft a structured pre-report — reducing the manual first-pass
review burden while keeping a clinician in the loop for every decision.

## Architecture

```
Fundus image
│
▼
EfficientNetB0 (fine-tuned) → predicted grade (0-4) + confidence
│
▼
Deterministic rule engine → action + follow-up interval
│ (grade + confidence + HbA1c + diabetes_years + spherical_refraction)
▼
RAG retrieval over clinical guideline corpus → relevant passages
│
▼
LLM (gpt-4o-mini) → structured clinical pre-report (drafts only —
│ never re-reads or overrides the rule engine's
│ action/interval/grade fields)
▼
Streamlit app → image + Grad-CAM heatmap + report for clinician review
```

## Results

| Model | QWK | Grade 3 Recall |
|---|---|---|
| Baseline CNN | 0.4972 | 0.103 |
| EfficientNetB0 (feature extraction) | 0.7910 | 0.586 |
| **EfficientNetB0 (fine-tuned, final)** | **0.7987** | **0.690** |

The fine-tuned model was selected as final based on Grade 3 recall —
catching severe cases was prioritized over overall accuracy or QWK
alone, since a missed severe case carries far higher clinical cost
than a false positive.

**Per-class recall (fine-tuned model)**:

| Grade | Recall |
|---|---|
| 0 | 0.952 |
| 1 | 0.309 |
| 2 | 0.353 |
| 3 | 0.690 |
| 4 | 0.455 |

Note the pattern here matches the Grad-CAM finding below: Grades 0 and
4 (the clear extremes) are recalled well, while the intermediate
grades (1, 2, 3) — which are harder to distinguish visually, as
independently confirmed by clinical experience — show lower recall.

**RAG retrieval performance**: hit rate@3 = 100%, hit rate@1 ≈ 78%
(7-chunk clinical guideline corpus — see Limitations).

## Key Findings

- **Model attention**: Grad-CAM analysis shows the model's attention
concentrates on the optic disc across all severity grades, rather
than shifting toward grade-specific lesions. This is independently
corroborated by clinical experience: Grade 0 and Grade 4 are easy to
distinguish visually, but the boundaries between Grade 1↔2 and
Grade 3↔4 are genuinely harder to separate — for the model and for a
human reviewer alike. This is documented as a real finding about the
task's difficulty, not treated as a model flaw to hide.

- **Follow-up intervals**: The rule engine's follow-up intervals
(12/6/6/1/1 months for grades 0-4) are clinically defined, not
learned or invented. Grade 3's 1-month interval is intentionally
stricter than the typical 3-4 month screening standard — full
rationale documented in `data/guidelines/icdrss_guidelines.md`.

## Limitations

- **Synthetic data**: Patient intake data used in the demo (HbA1c,
diabetes duration, spherical refraction, age, visual acuity) is
synthetically generated for demonstration purposes — it does not
come from real patients.
- **Partial rule engine scope**: Only 3 of 5 captured intake fields
(HbA1c, diabetes_years, spherical_refraction) currently feed the
rule engine's modifiers. Age and visual_acuity are captured but not
yet incorporated — a known, accepted scope limit for this phase.
- **RAG corpus size**: The clinical guideline corpus is intentionally
small (7 sections/chunks). Retrieval hit rate@3 is 100%, top-1 is
~78% — an accepted limitation of a small corpus, not an area under
active optimization (a chunk-prefix experiment was tried and
reverted after it made retrieval worse).
- **Optional deployment layer skipped**: A FastAPI/Docker deployment
layer (Block 19 of the project workflow) was deliberately not
implemented, per the original scope plan — documented in
`docs/workflow.md`.

## ⚠️ Not a Diagnostic Device

This system is a decision-support prototype developed for educational
purposes. It is **not** a certified medical device and has **not** been
validated, approved, or cleared by any regulatory body (e.g. FDA, CE
marking, AEMPS). All outputs — grade predictions, triage
recommendations, and generated reports — are intended solely to assist
a licensed ophthalmologist's review and must never be used as the sole
basis for a clinical decision. Final diagnosis and treatment decisions
remain the sole responsibility of the treating clinician.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/madatascienceml/project-ds-end-to-end-v2.git
cd project-ds-end-to-end-v2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables — create a `.env` file in the project
root with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

5. Trained model weights are hosted on Hugging Face Hub (not included
in this repository — see `.gitignore`, `models/*`). Download it
with:
```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='manudaza/retinal-triage-efficientnetb0', filename='efficientnetb0_finetuned_patched.keras', local_dir='models/')"
```
Or download manually from the
[model repo](https://huggingface.co/manudaza/retinal-triage-efficientnetb0)
and place it at `models/efficientnetb0_finetuned_patched.keras`.

## Usage

Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

This opens a local web interface with two tabs:
- **Screening**: upload a fundus image (or use a pre-filled synthetic
intake form), view the predicted grade, Grad-CAM heatmap, triage
decision, and the generated clinical pre-report.
- **Insights**: model performance summary (QWK, per-grade recall, RAG
retrieval metrics), a summary of the current session's processed
cases (grade distribution and automation rate), and a link to the
full Tableau dashboard.

To explore the underlying notebooks (EDA, preprocessing, model
training, evaluation):
```bash
jupyter notebook notebooks/
```

## Dashboard

Interactive Tableau dashboard covering class distribution, image
quality by class, confusion matrix, and the automation-vs-recall
trade-off curve:

**[View the full dashboard on Tableau Public →](https://public.tableau.com/app/profile/manu.daza/viz/RetinalScreeningTriageSystemFigures/RetinalScreeningTriageSystemModelEvaluationDashboard)**

## Demo Video

A full walkthrough of the app — image upload, prediction, Grad-CAM
heatmap, triage decision, and generated report:

**[Watch the demo video →](https://youtu.be/5TFIDQ0_5xA)**

## Presentation

**[View the presentation slides (PDF) →](https://drive.google.com/file/d/18QmRTkhVfjLDOHwO3nZVcP7nfTyT9SDT/view?usp=sharing)**

## Tech Stack

- **Language**: Python
- **Deep Learning**: TensorFlow/Keras, EfficientNetB0
- **RAG / LLM**: Custom RAG pipeline (FAISS + sentence-transformers + OpenAI, no LangChain dependency), OpenAI (gpt-4o-mini)
- **App**: Streamlit
- **Data**: pandas, NumPy, OpenCV, Pillow
- **Visualization**: Matplotlib, Tableau Public
- **Model hosting**: Hugging Face Hub
- **Dev tools**: Git/GitHub, Google Colab (training), Claude Code Desktop

## License

Code in this repository is licensed under MIT (see `LICENSE`). The
trained model is licensed separately under CC-BY-NC-4.0 (non-commercial)
and distributed via its
[Hugging Face repo](https://huggingface.co/manudaza/retinal-triage-efficientnetb0).
