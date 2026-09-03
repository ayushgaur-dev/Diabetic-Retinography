Experiment 1 — Baseline CNN (from scratch)
- Architecture: 3x Conv2D+MaxPool blocks, Dense(128), Dropout(0.3)
- Trained: 5 epochs (EarlyStopping, patience=3), stopped at epoch 2 best weights
- Train accuracy: 0.61, Val accuracy: 0.60
- QWK: 0.4972
- Per-class recall: Grade 0: 0.749, Grade 1: 0.691, Grade 2: 0.400, Grade 3: 0.103, Grade 4: 0.250
- Key finding: severe train/val recall gap on minority classes (Grade 3
  especially) despite class_weight balancing — motivates transfer learning
  in next phase.

  
  ## Experiment 2 — EfficientNetB0, Phase 1 (Feature Extraction)
- Architecture: EfficientNetB0 (frozen, ImageNet weights) + Dropout(0.3) + Dense(5)
- Trainable params: 6,405 (vs 11,169,605 in baseline)
- Preprocessing: EfficientNet-specific (tf.keras.applications.efficientnet.preprocess_input), 
  NOT /255.0 — using generic normalisation caused near-random performance (~20% accuracy) 
  before this fix
- Trained: 11 epochs (EarlyStopping, patience=3), best weights restored from epoch 8
- Val accuracy: 0.72, Val loss: 0.73
- QWK: 0.7910
- Per-class recall: Grade 0: 0.923, Grade 1: 0.564, Grade 2: 0.553, Grade 3: 0.586, Grade 4: 0.341
- Key finding: massive improvement over baseline, especially Grade 3 recall 
  (0.103 → 0.586). Slight recall drop on Grade 1 vs baseline, with errors 
  concentrated toward Grade 2 (adjacent class) — consistent with QWK's 
  tolerance for near-miss errors.


  ## Experiment 3 — EfficientNetB0, Phase 2 (Fine-tuning)
- Unfroze last 20 layers of EfficientNetB0 base (1,136,181 trainable params)
- Learning rate: 1e-5 (Adam)
- Trained: 4 epochs (EarlyStopping, patience=3), best weights restored from epoch 1
- QWK: 0.7987 (vs 0.7910 in Feature Extraction — marginal global improvement)
- Per-class recall: Grade 0: 0.952, Grade 1: 0.309, Grade 2: 0.353, Grade 3: 0.690, Grade 4: 0.455
- Key finding: fine-tuning improved Grade 3 recall substantially (0.586 → 0.690) 
  — the clinically critical class — at the cost of reduced recall on Grades 1-2, 
  where predictions shifted toward more severe grades. Given the project's 
  design priority on catching severe cases (false negatives are the highest-
  cost error), this model is selected as final over Feature Extraction despite 
  its more mixed per-class profile.


## Final model selection

**Selected model**: fine-tuned EfficientNetB0 (Experiment 3), over the
feature-extraction-only model (Experiment 2).

- QWK: 0.7987 (fine-tuned) vs 0.7910 (feature extraction) — a marginal
  global difference.
- Grade 3 recall: 0.690 (fine-tuned) vs 0.586 (feature extraction) — the
  decisive factor. Grade 3 is the clinically critical class where a false
  negative carries the highest cost, so materially better recall there is
  preferred despite a more mixed per-class profile elsewhere (Grade 1-2
  recall dropped under fine-tuning).

**Grad-CAM finding**: across all grades, model attention concentrates
consistently on the optic disc and surrounding vasculature, rather than
shifting toward grade-specific lesions (microaneurysms, haemorrhages, hard
exudates) as severity increases. This is a plausible explanation for the
confusion between adjacent grades in the confusion matrix, and motivated a
stricter safety net in the rule engine (`src/rules/triage.py`): Grades 1-3
require confidence ≥0.75, not just the base 0.60 threshold, before
bypassing human review.


### Retrieval evaluation note

Manual verification with 10 domain-relevant queries showed 8/10 correct
top-1 retrieval. Two queries (urgent referral, proliferative DR)
retrieved the correct chunk at rank 2 rather than rank 1, likely due to
semantic overlap between sections and the small corpus size (7 chunks
total). A mitigation was tested — prefixing each chunk with its section
title to sharpen the embedding signal — but this slightly regressed
performance (7/10) rather than improving it, as short prefixes were
over-weighted relative to longer chunk bodies by the embedding model.
The original chunking was retained. Mitigated at the generation stage
by retrieving top-3 chunks (not just top-1), ensuring the correct
source remains in context even when not ranked first.


### RAG evaluation (formal, 18-query eval set)

Hit rate @3: 100% (18/18) — the correct chunk is always present in the
top-3 retrieved results, which is what the generation stage actually
uses.
Top-1 accuracy: 77.8% (14/18) — consistent with the earlier 10-query
baseline (80%), confirming the pattern is stable rather than sampling
noise. All 4 top-1 misses trace to the same cause: characteristic_findings_by_grade
is the longest, most content-dense section, giving it a ranking
advantage on cross-cutting queries against shorter, more specific
sections. This is an accepted, documented limitation of the small
corpus (7 sections) rather than a retrieval bug, mitigated by using
top-3 (not top-1) at the generation stage.

**Faithfulness check** (2 cases, manual review): both generated reports
(Grade 3 urgent, Grade 0 routine) were fully grounded in their source
chunks — no invented claims, no drift into content from other grades
despite grade_definitions_icdrss covering all five. Minor observation:
in one case, source_chunks listed all 3 retrieved chunks while the
narrative guideline_applied text only named 2 of them explicitly — a
citation-completeness gap, not a faithfulness violation, since the
underlying content remained accurate.


### Model loading fix — quantization_config incompatibility (Block 18)

`models/efficientnetb0_finetuned.keras`, saved from Colab, failed to
load locally with an error deserializing the Dense layer:
`Unrecognized keyword arguments passed to Dense: {'quantization_config': None}`.

Root cause: the Colab environment's Keras build supports a
`quantization_config` parameter on Dense layers and wrote it into the
saved config.json, but the latest stable Keras available on PyPI
(3.12.3, and 3.12.4 — both tried, neither helped) doesn't recognise
that key on load. This isn't a "too old" version problem — both
tensorflow (2.21.0) and keras (3.12.4) were already at the newest
release available; there is no newer stable version to upgrade to.

**Fix**: copied the file to
`models/efficientnetb0_finetuned_patched.keras`, stripped the
`quantization_config` key from the Dense layer's config.json inside
that copy only (a .keras file is a zip archive), and re-zipped it. The
patched copy loads successfully with the correct architecture
(EfficientNetB0 + Dropout + Dense(5), matching the original training
notebook's saved parameter counts exactly). The original file was never
modified — confirmed via file size and the fact that it was only ever
opened in read mode.

Both `.keras` files are excluded from git via `.gitignore` (regenerable
model artifacts) — this note exists so the fix is reproducible without
relying on the files themselves being committed.