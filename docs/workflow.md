# Workflow — Retinal Screening Triage System

Step-by-step execution checklist for the 7-day sprint.
Mirrors the GitHub Projects board one-to-one: 23 blocks, 23 cards.

---

## Sprint Rules

1. **Commit at the end of every block.** Don't accumulate — if something breaks, you can roll back.
2. **Every day must end with something that runs end to end**, however crude.
3. **Never perfect one stage before closing the full loop.** Dummy model → hardcoded report → then swap in real components.
4. **Day 5, 20:00 → code freeze.** Whatever isn't done, isn't in.
5. If a step blocks you for more than 45 minutes, move the card to `Blocked`, apply the fallback from section 9 of the brief, and continue.

---

## DAY 0 — Prerequisites

### Block 1 — Prerequisites (30 min)

- [ ] Accept competition Rules on Kaggle (APTOS 2019)
  > Without this the download returns 403. The only blocker that cannot be resolved on Day 1.
- [ ] Create Kaggle API token (`kaggle.json`)
- [ ] Test download in Colab to confirm access works
- [ ] Request PhysioNet credentialing (future work, non-blocking)

---

## DAY 1 — Structure, Data and First Contact

### Block 2 — Repository setup (1h)

- [ ] Enable Issues in repository settings
  > Forked repos have Issues disabled by default. Required for linking commits to the project board.
- [ ] Move the Ironhack README to `docs/ironhack_requirements.md`
- [ ] Move the brief to `docs/project_brief.md`
- [ ] Add this workflow to `docs/workflow.md`
- [ ] Write `.gitignore` before the first commit
  > If 10 GB of images end up in the Git history, cleaning it afterwards is painful.
- [ ] Create the full folder tree with `.gitkeep` placeholders
  > Building the whole tree up front prevents improvising file locations later.
- [ ] Initial `requirements.txt`
- [ ] Commit: `chore: initial project structure`

### Block 3 — Data acquisition (1.5h)

- [ ] Mount Google Drive in Colab
- [ ] Download APTOS via `kagglehub`
- [ ] Extract to `MyDrive/June/final_project/data/raw/`
  > To Drive, not to Colab's local disk — local disk is wiped on every restart.
- [ ] Verify integrity: file count vs `train.csv` row count
  > Catch mismatches now, not on Day 3.
- [ ] Load `train.csv` — `head()`, `info()`, `shape`

### Block 4 — Initial EDA (2h) — `notebooks/`

- [ ] Class distribution + bar chart
- [ ] Compute imbalance ratio (grade 0 vs grade 3)
- [ ] Plot 5x3 image grid, one row per grade
- [ ] Clinical inspection: note artifacts, focus and framing issues
  > Look at them properly, with a clinical eye. This is the input for Stage 0 (quality gate) — information only you can extract.
- [ ] Extract per-image dimensions → resolution distribution
- [ ] Compute mean luminance → flag over/under-exposed captures
- [ ] Save `data/processed/eda_summary.csv`
  > This CSV feeds the Tableau dashboard on Day 6. Prepare it now.
- [ ] Commit: `feat: initial EDA and data verification`

---

## DAY 2 — Preprocessing and Baseline

### Block 5 — Data pipeline (2h) — `src/data/`

- [ ] Stratified train/val/test split 70/15/15
  > Mandatory: with 193 grade-3 images, a random split can leave almost none in validation.
- [ ] Document leakage limitation (no patient ID → no patient-level split)
- [ ] Preprocessing function: resize 224x224 + normalisation
- [ ] Augmentation: rotation, horizontal flip only, brightness, zoom
  > Horizontal flip only. A vertically flipped retina does not exist clinically.
- [ ] Build `tf.data.Dataset` with batching and prefetch
- [ ] Visually inspect one augmented batch
  > Excessive augmentation destroys microaneurysms.

### Block 6 — Metrics (1h) — `src/models/`

- [ ] Implement Quadratic Weighted Kappa
  > Do this before training anything, or you will optimise the wrong metric.
- [ ] Evaluation function: QWK + confusion matrix + per-class recall
- [ ] Compute `class_weight` for imbalance

### Block 7 — Baseline model (2h) — `src/models/`

- [ ] Simple CNN from scratch (3 Conv+Pool blocks + Dense)
- [ ] Train ~15 epochs with EarlyStopping
- [ ] Evaluate: QWK, confusion matrix, recall grades 3-4
- [ ] Create `experiments.md` and log the run
  > One line per run. On Day 7 you must justify your model choice.
- [ ] Commit: `feat: preprocessing pipeline and baseline CNN`

---

## DAY 3 — Transfer Learning, Explainability and Rules

### Block 8 — Transfer learning (3h) — `src/models/`

- [ ] Load EfficientNetB0, `include_top=False`, ImageNet weights
- [ ] Phase 1 — feature extraction (frozen base)
- [ ] Evaluate and log
- [ ] Phase 2 — fine-tuning (unfreeze top layers, lr=1e-5)
- [ ] Evaluate and log — track train/val gap, not just peak
  > More fine-tuning does not always generalise better.
- [ ] Comparison table: baseline vs feature extraction vs fine-tuning
- [ ] Use distinct variable names per experiment
  > `model_baseline`, `model_fe`, `model_ft` — never reuse `model`.
- [ ] Save best model to `models/`

### Block 9 — Decision curve (1h) — `src/models/`, `reports/figures/`

- [ ] Extract softmax probabilities on validation set
- [ ] For thresholds 0.5-0.95: compute % automated and recall grades 3-4
- [ ] Plot both curves together
  > This is the central chart of the presentation.
- [ ] Select operating threshold
- [ ] Write the clinical justification for the threshold

### Block 10 — Grad-CAM (1.5h) — `src/models/`, `reports/figures/`

- [ ] Implement Grad-CAM on final conv layer
- [ ] Generate heatmaps for 2-3 examples per grade
- [ ] Qualitative clinical validation: is attention where it should be?
  > If a grade-4 heatmap highlights the image border, that is a real finding worth presenting.
- [ ] Grad-CAM output: implemented as a live overlay in the Streamlit
  app's Screening tab (Block 18, `src/models/gradcam.py`), shown
  alongside the original image for every uploaded case — not as
  static exported images.
  > `reports/figures/` remains reserved for static exports (e.g.
  > presentation slides) if needed later — it is not a pending
  > deliverable from this block.

### Block 11 — Rule engine (1.5h) — `src/rules/`, `tests/`

- [ ] Decision table: grade → action + base interval
- [ ] Add modifiers: HbA1c, diabetes duration, visual acuity, high myopia
- [ ] Add abstention logic (confidence below threshold)
- [ ] Implement as pure function: dict in, dict out
  > No model dependency — testable without loading the network.
- [ ] Write 5-6 unit tests
- [ ] Commit: `feat: transfer learning, grad-cam and triage rule engine`

---

## DAY 4 — RAG

### Block 12 — Guideline corpus (2h) — `data/guidelines/`

- [ ] ICDRSS grade definitions
- [ ] Characteristic findings per grade
- [ ] Recommended follow-up intervals
- [ ] Urgent referral criteria
- [ ] Image quality criteria
- [ ] Note on DME and fundus photography limitation
- [ ] Cite public sources
  > This comes from 15 years of clinical practice — the part nobody else can write.

### Block 13 — Vector index (1.5h) — `src/rag/`, `models/`

- [ ] Install `sentence-transformers` and `faiss-cpu`
- [ ] Chunking: ~500 tokens, ~50 overlap
- [ ] Load `all-MiniLM-L6-v2` in one reusable function
  > Using a different embedding model at index and query time returns noise.
- [ ] Generate embeddings for all chunks
- [ ] Build FAISS index and persist to disk
- [ ] Save `chunk_id → text` mapping

### Block 14 — Retrieval verification (30 min)

- [ ] Run 10 test queries
- [ ] Print retrieved chunks
- [ ] Manually verify correctness
- [ ] GATE: do not proceed to the LLM until this passes
  > If retrieval is broken you will blame the LLM and debug the wrong stage for hours.

### Block 15 — Report generation (2h) — `src/rag/`

- [ ] Define JSON output schema
- [ ] Write prompt: rule decision + retrieved chunks → report
- [ ] Instruction: use only the provided passages
- [ ] Instruction: the interval is given, do not modify it
  > The LLM drafts. It never decides.
- [ ] Include `source_chunks` in output (audit trail)
- [ ] Include `macular_status_assessed: false`
- [ ] Robust JSON parsing with try/except
- [ ] Test with 3 cases across different grades
- [ ] Commit: `feat: RAG pipeline with grounded report generation`

---

## DAY 5 — Evaluation, App and Freeze

### Block 16 — RAG evaluation (1.5h) — `src/rag/`

- [ ] Build eval set: 15-20 queries with correct chunk annotated
- [ ] Compute hit rate @ 3
- [ ] Faithfulness check on 10 generated reports
- [ ] Record both metrics
  > Almost no bootcamp project evaluates its RAG. This is a strong maturity signal.

### Block 17 — Synthetic intake generator (45 min) — `data/synthetic/`

- [ ] Plausible distributions, not uniform ranges
- [ ] Internal consistency constraints
- [ ] Header notice: `SYNTHETIC DATA — DEMONSTRATION ONLY`
- [ ] Fixed random seed
- [ ] Generate ~100 demo rows

### Block 18 — Streamlit app (2.5h) — `app/`

- [ ] App skeleton with two tabs: Screening / Insights
- [ ] Screening tab: image uploader + pre-filled editable intake form
  > Pre-filled so the demo runs alone; editable so you can answer live questions.
- [ ] Wire pipeline: image → model → rules → RAG → report
- [ ] Display Grad-CAM heatmap alongside original
- [ ] Display structured report + clinical review notice
- [ ] Run full flow 3 times with different cases

### Block 19 — API and container, optional (1.5h) — `api/`

- [ ] FastAPI `/predict` endpoint
- [ ] Test with curl or Postman
- [ ] Dockerfile
- [ ] Local build
- [ ] Drop this block if not working by 20:00
  > Document deployment in the README instead. Not a blocker.
- [ ] CODE FREEZE
- [ ] Commit: `feat: evaluation, streamlit app and deployment`

---

## DAY 6 — Tableau, Documentation and Demo

### Block 20 — Tableau dashboard (3h) — `data/processed/`

- [ ] Export CSVs: EDA summary, model results, prediction log
- [ ] Install Tableau Public
- [ ] Sheet 1: class distribution
- [ ] Sheet 2: image quality by class
- [ ] Sheet 3: confusion matrix
- [ ] Sheet 4: automation vs recall trade-off
- [ ] Assemble dashboard
- [ ] Publish to Tableau Public and save link
  > Unpublished, nobody can see it. The link goes in the README.

### Block 21 — Documentation (2h)

- [ ] README: problem, architecture diagram, results, limitations, install, usage
- [ ] Declare synthetic data explicitly
- [ ] State "not a diagnostic device" notice
- [ ] `pip freeze > requirements.txt`, then prune
- [ ] Verify `.gitignore` works (`git status` clean)
- [ ] Add dashboard link to README

### Block 22 — Demo recording (1h)

- [ ] Record 60-90s of the full flow
  > Safety net for Day 7, and reusable on LinkedIn.
- [ ] Upload video
- [ ] Link in README
- [ ] Commit: `docs: README, dashboard and demo`

---

## DAY 7 — Presentation

### Block 23 — Presentation (full day)

- [ ] Build 10-slide deck:
  1. The problem: the review queue in the virtual clinic
  2. Why me: 15 years in ophthalmology
  3. Dataset and its limitations
  4. System architecture
  5. Model results (QWK, severe-grade recall)
  6. The trade-off curve
  7. Grad-CAM: what the model attends to
  8. The Gen AI component and its evaluation
  9. Live demo
  10. Limitations and future work
- [ ] Prepare answers to the predictable questions:
  - Why not an end-to-end model for the interval?
  - Why no OCT?
  - Why QWK and not accuracy?
  - How do you know the RAG doesn't hallucinate?
- [ ] Timed rehearsal #1
- [ ] Cut whatever overruns
- [ ] Timed rehearsal #2
- [ ] Verify demo on presentation machine
- [ ] Screenshot the completed board for the slides
- [ ] Final commit + tag `v1.0`

---

## Final Delivery Checklist

- [ ] Repository public and accessible
- [ ] Complete README with architecture diagram and results
- [ ] Working `requirements.txt`
- [ ] Notebooks runnable end to end
- [ ] Tableau dashboard published and linked
- [ ] Demo video linked
- [ ] Slides submitted
- [ ] Heavy data excluded from the repo
- [ ] Synthetic data declared in all three places (code, README, slides)
