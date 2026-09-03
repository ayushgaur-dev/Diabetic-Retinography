"""RAG evaluation (Block 16): retrieval hit rate + manual faithfulness review.

Two separate concerns, two separate functions:

1. evaluate_hit_rate() — quantitative, automated. Extends the 10 queries
   already used in test_retrieval.py (Block 14) with 8 more for full
   7-section coverage (the original 10 never touched
   `characteristic_findings_by_grade`), reports hit_rate@3 and top-1
   accuracy against a hand-labelled ground truth.

2. evaluate_faithfulness() — qualitative, manual. Deliberately NOT scored
   by another LLM call: at this project's scale (2 cases), a human
   reading generated text against its source chunks side by side is more
   trustworthy than an unverified LLM-judge, and avoids compounding one
   model's potential errors with another's. Makes real generate_report()
   API calls (gpt-4o-mini) — not mocked.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.generator import generate_report
from src.rag.indexer import CHUNKS_PATH, load_index, retrieve
from src.rag.test_retrieval import TEST_QUERIES

# ---------------------------------------------------------------------------
# 1. Hit rate @3 / top-1 accuracy
# ---------------------------------------------------------------------------

# Ground-truth correct chunk_id for each of the 10 queries already used in
# test_retrieval.py (TEST_QUERIES), in the same order — determined by
# manual review during Block 14's retrieval verification.
_BASELINE_LABELS = [
    "recommended_follow_up_intervals",
    "urgent_referral_criteria",
    "grade_definitions_icdrss",
    "image_quality_criteria",
    "macular_oedema_and_fundus_photography_limitations",
    "screening_context",
    "grade_definitions_icdrss",
    "grade_definitions_icdrss",
    "recommended_follow_up_intervals",
    "image_quality_criteria",
]

# New queries added for full 7-section coverage — the original 10 never
# targeted `characteristic_findings_by_grade` at all.
_ADDITIONAL_QUERIES = [
    ("What do microaneurysms look like on fundus imaging and what do they indicate?",
     "characteristic_findings_by_grade"),
    ("What is the clinical significance of hard exudates located near the macula?",
     "macular_oedema_and_fundus_photography_limitations"),
    ("What are cotton wool spots and what do they indicate?",
     "characteristic_findings_by_grade"),
    ("Should pregnant patients with diabetes be screened more frequently?",
     "screening_context"),
    ("What are the risks of non-mydriatic (non-dilated) image capture compared to pupillary dilation?",
     "image_quality_criteria"),
    ("How do individual patient risk factors like HbA1c or diabetes duration affect the recommended follow-up interval?",
     "recommended_follow_up_intervals"),
    ("What should happen if a patient reports sudden vision loss even though their DR grade looks mild?",
     "urgent_referral_criteria"),
    ("How does high myopia complicate the interpretation of hard exudates near the macula?",
     "macular_oedema_and_fundus_photography_limitations"),
]

assert len(TEST_QUERIES) == len(_BASELINE_LABELS), \
    "TEST_QUERIES in test_retrieval.py changed length — update _BASELINE_LABELS to match"

EVAL_SET = list(zip(TEST_QUERIES, _BASELINE_LABELS)) + _ADDITIONAL_QUERIES


def evaluate_hit_rate(top_k=3):
    """Retrieve top_k chunks for every (query, correct_chunk_id) pair in
    EVAL_SET and report hit_rate@top_k plus top-1 accuracy.

    top-1 accuracy is reported separately so it can be compared directly
    against the known 8/10 (80.0%) baseline from the original 10-query
    test_retrieval.py run.
    """
    index, chunk_mapping = load_index()
    valid_chunk_ids = {v["chunk_id"] for v in chunk_mapping.values()}

    unknown = [cid for _, cid in EVAL_SET if cid not in valid_chunk_ids]
    assert not unknown, f"EVAL_SET references chunk_id(s) not in the index: {unknown}"

    results = []
    for query, correct_chunk_id in EVAL_SET:
        retrieved = retrieve(query, index, chunk_mapping, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        hit_at_k = correct_chunk_id in retrieved_ids
        hit_at_1 = bool(retrieved_ids) and retrieved_ids[0] == correct_chunk_id
        results.append({
            "query": query,
            "correct_chunk_id": correct_chunk_id,
            "retrieved_ids": retrieved_ids,
            "hit_at_k": hit_at_k,
            "hit_at_1": hit_at_1,
        })

    n = len(EVAL_SET)
    hits_at_k = sum(r["hit_at_k"] for r in results)
    hits_at_1 = sum(r["hit_at_1"] for r in results)
    hit_rate_at_k = hits_at_k / n * 100
    top1_accuracy = hits_at_1 / n * 100

    print(f"Evaluation set size: {n} queries covering all 7 guideline sections")
    print(f"\nHit rate @{top_k}:  {hits_at_k}/{n} ({hit_rate_at_k:.1f}%)")
    print(f"Top-1 accuracy: {hits_at_1}/{n} ({top1_accuracy:.1f}%)")
    print("(Baseline from the original 10-query test_retrieval.py run: 8/10 = 80.0% top-1)")

    print("\nPer-query detail:")
    for r in results:
        mark_k = "PASS" if r["hit_at_k"] else "FAIL"
        mark_1 = "PASS" if r["hit_at_1"] else "FAIL"
        print(f"  [top{top_k}:{mark_k}  top1:{mark_1}] {r['query']}")
        print(f"      expected={r['correct_chunk_id']!r}  retrieved={r['retrieved_ids']}")

    return {
        "n_queries": n,
        "hit_rate_at_k": hit_rate_at_k,
        "top1_accuracy": top1_accuracy,
        "results": results,
    }


# ---------------------------------------------------------------------------
# 2. Faithfulness — manual side-by-side review
# ---------------------------------------------------------------------------

# The same two real scenarios already exercised against generate_report()
# directly: Grade 3 urgent referral, and Grade 0 routine virtual follow-up.
FAITHFULNESS_CASES = [
    {
        "label": "Grade 3 — urgent referral",
        "triage_decision": {
            "predicted_grade": 3,
            "confidence": 0.90,
            "requires_human_review": False,
            "action": "urgent_referral",
            "interval_months": 1,
            "notes": [],
        },
        "chunk_ids": [
            "grade_definitions_icdrss",
            "urgent_referral_criteria",
            "recommended_follow_up_intervals",
        ],
    },
    {
        "label": "Grade 0 — routine virtual follow-up",
        "triage_decision": {
            "predicted_grade": 0,
            "confidence": 0.88,
            "requires_human_review": False,
            "action": "virtual_followup",
            "interval_months": 12,
            "notes": [],
        },
        "chunk_ids": [
            "grade_definitions_icdrss",
            "recommended_follow_up_intervals",
        ],
    },
]


def _load_chunk_texts():
    """chunk_id -> full text, read straight from the persisted index —
    not hand-copied excerpts, so faithfulness review is against the
    actual retrievable source, not an approximation of it."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunk_mapping = json.load(f)
    return {v["chunk_id"]: v["text"] for v in chunk_mapping.values()}


def evaluate_faithfulness():
    """Call generate_report() on each FAITHFULNESS_CASES entry (real API
    calls) and print the generated findings/guideline_applied alongside
    their full source chunk text, for a human to check by eye: does
    every claim in the generated text trace back to the source, with no
    invented detail? No automated faithfulness score is computed here —
    that's a deliberate choice, not an oversight (see module docstring).
    """
    chunk_texts = _load_chunk_texts()

    for case in FAITHFULNESS_CASES:
        retrieved_chunks = [
            {"chunk_id": cid, "text": chunk_texts[cid]} for cid in case["chunk_ids"]
        ]
        report = generate_report(case["triage_decision"], retrieved_chunks)

        print("=" * 80)
        print(f"CASE: {case['label']}")
        print("=" * 80)
        td = case["triage_decision"]
        print(f"Triage decision: grade={td['predicted_grade']}, action={td['action']}, "
              f"interval={td['interval_months']} months, confidence={td['confidence']}")

        print("\n--- GENERATED: findings ---")
        print(report.get("findings"))
        print("\n--- GENERATED: guideline_applied ---")
        print(report.get("guideline_applied"))
        print(f"\n--- GENERATED: source_chunks = {report.get('source_chunks')} ---")

        print("\n--- SOURCE chunk text (compare every claim above against this) ---")
        for cid in case["chunk_ids"]:
            print(f"\n[{cid}]")
            print(chunk_texts[cid])

        print("\nManual check: does every claim in 'findings' and "
              "'guideline_applied' above trace back to the source text? "
              "Flag anything that looks invented or unsupported.\n")


if __name__ == "__main__":
    print("### Hit rate @3 / top-1 accuracy ###\n")
    evaluate_hit_rate()

    print("\n\n### Faithfulness review (manual — makes real API calls) ###\n")
    evaluate_faithfulness()
