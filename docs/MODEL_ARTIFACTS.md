# Model Artifacts (SIH26038 — Phase 1)

## Grading model (EfficientNetB0, fine-tuned)

- **Source:** Hugging Face Hub — `manudaza/retinal-triage-efficientnetb0`
  (as documented in the base README; weights are NOT committed to git — `models/*` is gitignored).
- **File:** `efficientnetb0_finetuned_patched.keras`
- **Local path:** `models/efficientnetb0_finetuned_patched.keras`
- **Size:** 23,425,638 bytes
- **SHA256:** `2a993b3affc96051aedc79d7b4c20f1538c6c9199d91032b22cfcc4642a24dd4`
- **Architecture:** `Sequential([EfficientNetB0(include_top=False, pooling='avg', ` +
  `last 20 layers unfrozen), Dropout(0.3), Dense(5, softmax)])` — verified on load:
  3 layers, base is `Functional`, output shape `(None, 5)`, last conv layer `top_activation`.
- **License:** CC-BY-NC-4.0 (non-commercial), per base README — separate from the code (MIT).
- **Compatibility note:** this is the *patched* copy (Dense-layer `quantization_config`
  key stripped from the archived config so local Keras can deserialise it).
  The unpatched original (`efficientnetb0_finetuned.keras`) was NOT downloaded.
  Loads cleanly under the Phase 1 environment (Keras 3.15.1 / TF 2.21.0, CPU) —
  the documented fragility did not bite here, but any re-save with a different
  Keras build may re-break loading. Do not re-save without testing load.
- **Download procedure (Windows, from repo root with venv active):**
  `.\venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; ` +
  `hf_hub_download(repo_id='manudaza/retinal-triage-efficientnetb0', ` +
  `filename='efficientnetb0_finetuned_patched.keras', local_dir='models/')"`
  No authentication required (public repo; anonymous rate limits apply).

## RAG embedding model

- **Model:** `all-MiniLM-L6-v2` (sentence-transformers, 384-dim), auto-downloaded from
  Hugging Face on first use into the Hugging Face cache. No manual step.

## FAISS guideline index (rebuilt in Phase 1, gitignored)

- **Source document:** `data/guidelines/icdrss_guidelines.md` (7 `##` sections).
- **Build:** `.\venv\Scripts\python.exe -m src.rag.indexer`
- **Artifacts:** `data/guidelines/index/guidelines.faiss` (10,797 bytes),
  `data/guidelines/index/chunks.json` (13,185 bytes) — 7 vectors, 384-dim.
- Rebuild is deterministic in content (chunking is rule-based); embedding
  numerics depend on the sentence-transformers/torch build.

## Data present / absent

- Present: `data/processed/` (confusion_matrix, eda_summary 200-row sample, tradeoff_curve),
  `data/synthetic/synthetic_intake_demo.csv` (100 rows, seed 42),
  `data/guidelines/icdrss_guidelines.md`, rebuilt FAISS index, downloaded weights.
- Absent (by design, gitignored): `data/raw/` APTOS images (~10 GB, Kaggle download —
  NOT fetched in Phase 1), original unpatched `.keras`, `api/` implementation.
