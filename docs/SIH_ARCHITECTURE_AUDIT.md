# SIH26038 — Architecture Audit (PHASE 0)

**Date:** 2026-09-03
**Scope:** Complete repository audit. No code implemented, no refactoring, no new features (per Phase 0 instructions).
**Auditor method:** Full read of every source file + notebook cell inventory + data-file inspection.

## 0. Critical workspace finding

**The local workspace (`C:\Users\Lenovo\Documents\Default Project`) was EMPTY and is NOT a git repository.**

There was no local code to audit. The audit below was performed against a fresh
`git clone --depth 1` of the declared base repository:

- Remote: `https://github.com/madatascienceml/project-ds-end-to-end-v2`
- Local clone used for audit: temp directory only (NOT copied into the workspace)
- HEAD at clone time: `fefba57 "Update demo video link in README"` (single squashed view; full history not fetched)
- The base repo is itself a fork of `ironhack-labs/project-ds-end-to-end-v2` (a bootcamp scaffold).

**Consequence for Phase 1:** "Make existing repository reproducibly run" cannot mean
"run what's in the workspace" — there is nothing there. Phase 1 must begin by
deciding how the base code enters this workspace (clone/fork/subtree + pin to a
commit hash), then making *that* runnable. See `SIH_IMPLEMENTATION_PLAN.md`.

All file paths below (e.g. `src/rules/triage.py:22`) refer to paths **inside the base
repository**, not to files currently in this workspace.

---

## 1. Directory structure (base repo, 30 tracked files)

```
api/                  # EMPTY — only .gitkeep (FastAPI layer deliberately skipped)
app/streamlit_app.py  # Streamlit UI, 2 tabs (Screening / Insights), ~430 lines
data/
  guidelines/icdrss_guidelines.md   # 7-section clinical corpus (RAG source of truth)
  processed/confusion_matrix.csv    # 5x5 validation-set confusion matrix (committed)
  processed/eda_summary.csv         # 200-row EDA sample: id_code, diagnosis, w, h, luminance
  processed/tradeoff_curve.csv      # threshold/automation/severe-recall table (committed)
  raw/.gitkeep                      # EMPTY by design — ~10 GB APTOS images, gitignored
  synthetic/synthetic_intake_demo.csv  # 100 synthetic demo rows (seed 42)
docs/                 # experiments.md, workflow.md, project_brief.md, ironhack_requirements.md
models/.gitkeep       # EMPTY by design — weights on Hugging Face, gitignored
notebooks/            # 01 EDA, 02 preprocessing+baseline, 03 transfer+GradCAM+rules, 04 tableau exports
reports/              # deck PDF + demo MP4 (figures/ has only .gitkeep)
src/
  data/synthetic_intake.py   # synthetic demo-data generator (isolated, seeded)
  models/gradcam.py          # Grad-CAM module extracted from notebook 03
  rag/indexer.py             # chunk + embed + FAISS index build/load/retrieve
  rag/test_retrieval.py      # manual 10-query retrieval check (script, NOT pytest)
  rag/eval_rag.py            # 18-query hit-rate eval + manual faithfulness review
  rag/generator.py           # gpt-4o-mini report drafter with guardrails + fallback
  rules/triage.py            # deterministic triage rule engine (pure function)
tests/test_triage.py  # ~20 unit tests for the rule engine ONLY
requirements.txt      # 14 pinned deps (see §13)
```

File inventory was enumerated exhaustively (all 30 files listed by size; every `.py`
file read in full; all 4 notebooks inventoried cell-by-cell with key cells read verbatim).

---

## 2. Source-code inventory — what each component does

| Component | File | Verdict |
|---|---|---|
| Preprocessing (train/infer) | notebooks `02[11]`, `03[9]` (inline functions, NOT in `src/`) | Works, but minimal: resize 224×224 + normalisation only. No module, no version, not importable. |
| Baseline CNN | notebook `02[31]` (3×Conv+Pool, Dense128, Dropout0.3) | Works as baseline. QWK 0.4972. Not persisted as a module. |
| EfficientNetB0 grading model | notebook `03[15,21,22,24]`; arch `Sequential([EfficientNetB0(pooling=avg, frozen→last-20-unfrozen), Dropout(0.3), Dense(5, softmax)])` | Works. The reusable asset. Weights NOT in repo (Hugging Face). |
| Grad-CAM | `src/models/gradcam.py:50-117` | Real Grad-CAM from the actual model. Top-predicted-class only; has fragility issues (see §11). |
| Rule engine | `src/rules/triage.py:22-83` | Deterministic, pure, well-tested. Reuse as-is; extend with new inputs. |
| RAG indexer/retriever | `src/rag/indexer.py:111-225` | Works. Section-chunking + `all-MiniLM-L6-v2` + flat-L2 FAISS. Index NOT committed (gitignored, must be rebuilt). |
| RAG report generator | `src/rag/generator.py:107-162` | Works WITH API key. Authoritative fields copied, citations filtered, fail-safe fallback. No offline mode. |
| RAG eval | `src/rag/eval_rag.py:76-216` | Works. Hit@3 = 100% (18/18), top-1 = 77.8% (14/18). Faithfulness = 2-case manual review (honest, small). |
| Synthetic intake | `src/data/synthetic_intake.py` | Works. Seeded (42), clearly labelled, isolated from training. Reuse pattern for SIH demos. |
| Streamlit UI | `app/streamlit_app.py` | Works (given model + index + API key). Graceful degraders everywhere. Hardcoded metrics. No quality/enhancement/calibration/lesion UI. |
| Tests | `tests/test_triage.py` | Only the rule engine is tested. Nothing else has tests. |
| API/deploy | `api/.gitkeep` | Does not exist. Deliberately skipped (documented in `docs/workflow.md` Block 19). |

---

## 3. Model loading

- Weights file `models/efficientnetb0_finetuned_patched.keras` is **absent from the repo**
  (gitignored: `models/*`). Download documented in README from Hugging Face
  (`manudaza/retinal-triage-efficientnetb0`).
- Known load bug documented in `docs/experiments.md:104-130`: Colab-saved `.keras`
  contains `quantization_config` in the Dense config; local Keras 3.12 rejects it.
  Fix = hand-patched zip copy. Reproducible in principle, fragile in practice —
  any re-training/re-saving with a different Keras build can re-break loading.
- `src/models/gradcam.py:40-47` (`get_base_model`) **assumes** `Sequential([base, Dropout, Dense])`
  i.e. base model is `layers[0]`. Any architecture change silently breaks Grad-CAM.
- App loads via `st.cache_resource` with graceful error (`app/streamlit_app.py:81-86`) — good.

## 4. Preprocessing (current)

- Baseline notebook `02[11]`: resize → `/255.0`.
- Transfer notebook `03[9]`: resize → `efficientnet.preprocess_input` (NOT /255 — a
  documented near-random-performance bug was fixed this way; `docs/experiments.md:15-17`).
- App inference `app/streamlit_app.py:101-111` matches `03[9]` — consistent. Good.
- Augmentation: horizontal flip + brightness(±0.1) + `rot90` (notebook `02[15]`, `03[9]`).
  Note: `rot90` includes 90°/270° rotations (not just 180°), which is anatomically
  questionable for fundus images (optic-disc/macula geometry rotated sideways).
  Minor, but record it before reusing augmentation for SIH.
- **There is NO**: CLAHE, illumination normalisation, denoising, colour normalisation,
  retinal masking, quality gating, or adaptive enhancement anywhere in the repo.

## 5. Training procedure (as executed in notebooks)

- Data: APTOS 2019 only (3662 images; 1805/370/999/193/295 for grades 0–4).
- Split: stratified 70/15/15, `random_state=42` (notebook `02[6]`, repeated in `03[5]`, `04[5]`).
  Reproducible given the same CSV. No patient IDs exist in APTOS `train.csv`, so no
  group-aware split is possible — **documented as a limitation** in `docs/project_brief.md:39-42`.
- Imbalance: `compute_class_weight('balanced')` + EarlyStopping (`val_loss`, patience 3,
  `restore_best_weights=True`).
- Phase 1 (feature extraction, frozen base, Adam default): 11 epochs, best epoch 8.
- Phase 2 (fine-tune: unfreeze last 20 layers, Adam lr=1e-5, `model_ft = model_fe` same object):
  4 epochs, best epoch 1. Trainable params 1,136,181.
- Seeds: `random_state=42` (splits), `SEED=42` + `tf.random.set_seed` (notebook 03).
  Augmentation randomness (`random_flip`, `random_brightness`, `rot90`) is NOT independently
  seeded — exact-epoch reproducibility is approximate, not bitwise.

## 6. Inference

- `app/streamlit_app.py:222-229`: `model.predict` → `argmax` grade + `max` softmax as "Confidence".
- Raw softmax maximum is **presented directly as calibrated confidence** (`st.metric("Confidence", ...)`).
  No temperature scaling, no ECE/Brier, no reliability diagram. This is the single most
  safety-relevant gap versus SIH §G.
- Grad-CAM is computed for the top predicted class only (`gradcam.py:78-79`), overlaid on the
  original resized image (`streamlit_app.py:236-242`). Real heatmap, correctly plumbed.

## 7. Grad-CAM

- Genuine implementation (gradient of predicted-class channel w.r.t. last 4D layer
  `top_activation`), extracted from notebook `03[38-39]` into `src/models/gradcam.py`.
- **Honest negative finding** (notebook `03[45]`, `docs/experiments.md:55-62`): attention
  concentrates on the **optic disc across all grades**, not on grade-specific lesions.
  Corroborated by the per-class recall profile. Used to justify the 0.75 intermediate-grade
  safety net — exemplary honest engineering; preserve this pattern for SIH.
- Limitations: (a) top-class only, no class-selectable maps; (b) `heatmap / max(heatmap)`
  with no epsilon — a zero-activation edge case yields NaN/inf; (c) JET colormap overlay
  only, no lesion/anatomy overlays; (d) `get_base_model` architecture assumption (see §3).

## 8. Rule engine

- `src/rules/triage.py:11-17` BASE_RULES: 0→(virtual_followup,12), 1→(virtual_followup,6),
  2→(ophthalmologist_review,6), 3→(urgent_referral,1), 4→(urgent_referral,1).
- Gates: abstain if confidence < 0.60 (`:31-39`); flag human review for grades 1–3 if
  confidence < 0.75 (`:43-49`); patient modifiers for HbA1c≥9, diabetes≥15y (grade>0),
  myopia≤−6D note-only (`:55-83`). Modifiers only shorten/flag, never loosen — good safety shape.
- Separated from the model (pure function), ~20 unit tests, all deterministic. **This is the
  component closest to SIH §H and should be extended, not rewritten.**
- Gaps vs SIH §H: no quality input, no lesion/vessel/anatomy input, no calibrated-confidence
  input, no binary referable-DR output, no model/preprocessing/rule version fields,
  thresholds hardcoded as module constants (no config file), age/visual_acuity accepted but unused.

## 9. Reporting (RAG)

- Corpus `data/guidelines/icdrss_guidelines.md`: 7 sections, clinically authored, includes
  honest scope boundaries (DME cannot be assessed from 2D fundus; `macular_status_assessed: false`
  plumbed end-to-end into `generator.py:101,160` and the UI `:324-328`). Reuse as the RAG seed.
- `generator.py`: system prompt forbids overriding the triage decision; authoritative fields
  copied verbatim (`:152-160`); cited chunk IDs filtered against actually-retrieved IDs
  (`:145-148`); failures return a fail-safe dict with `requires_human_review=True` (`:85-104`).
- **Gaps vs SIH §I:** report REQUIRES `OPENAI_API_KEY` + built FAISS index + embedding-model
  download — there is **no offline/template fallback** (on API failure, findings=None).
  Schema lacks quality, lesions, anatomy, calibrated confidence, uncertainty, and all version
  fields. RAG eval is 18 hand-labelled queries on a 7-chunk corpus — adequate for the demo,
  not a clinical validation.

## 10. Streamlit UI

- Two tabs (Screening / Insights), synthetic-prefilled editable intake, image+Grad-CAM+metrics
  display, session-only case log, static metrics + Tableau link + minimal EDA view.
- Every Screening render carries the not-a-device notice (`DEVICE_NOTICE`,
...[truncated 8188 chars]