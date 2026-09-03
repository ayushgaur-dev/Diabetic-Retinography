"""Manual retrieval verification for the guidelines FAISS index (Block 14).

Loads the persisted index + chunk mapping from data/guidelines/index/,
runs 10 domain-relevant test queries, and prints the top-3 retrieved
chunks for each so retrieval quality can be checked by eye before an LLM
is layered on top. GATE: do not proceed to report generation until this
passes — if retrieval is broken, debugging the LLM instead wastes hours.

This is a standalone script, not a pytest test suite (despite the
`test_` filename, requested to match this file's purpose in the
workflow) — run it directly, not via `pytest`. All work happens inside
main(), guarded by `if __name__ == "__main__"`, so importing this module
(e.g. if pytest happens to collect it) has no side effects.

load_index() and retrieve() live in indexer.py, not here — they're
reused by eval_rag.py and the Streamlit app too, so this "test_" named
script isn't the right home for shared, non-test logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.indexer import load_index, retrieve

TEST_QUERIES = [
    "What follow-up interval is recommended for a patient with no diabetic retinopathy?",
    "When should a patient be referred urgently to an ophthalmologist?",
    "What are the criteria for severe non-proliferative diabetic retinopathy?",
    "How should poor image quality or an ungradable image be handled?",
    "What is diabetic macular oedema and why can't a single fundus photo diagnose it?",
    "At what point after a diabetes diagnosis should retinopathy screening begin?",
    "What distinguishes mild from moderate non-proliferative retinopathy?",
    "What findings indicate proliferative diabetic retinopathy?",
    "Why is the follow-up interval for severe NPDR shorter than standard screening protocols?",
    "What acquisition factors, such as pupil dilation, affect fundus photography quality?",
]


def _print_results(query, results):
    print(f"\nQuery: {query}")
    for r in results:
        preview = " ".join(r["text"].split())
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"  [{r['rank']}] {r['chunk_id']}  (L2 distance={r['distance']:.4f})")
        print(f"      {preview}")


def main():
    index, chunk_mapping = load_index()
    print(f"Loaded index: {index.ntotal} vectors, {len(chunk_mapping)} chunks")

    for query in TEST_QUERIES:
        results = retrieve(query, index, chunk_mapping, top_k=3)
        _print_results(query, results)


if __name__ == "__main__":
    main()
