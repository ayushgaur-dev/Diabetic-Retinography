"""FAISS vector index builder for the clinical guidelines corpus.

Chunks `data/guidelines/icdrss_guidelines.md` along its `## ` section
boundaries — the document was deliberately written so each `## ` section
is a self-contained unit, so section-based chunking (rather than fixed
token windows) keeps each chunk semantically coherent. Nested `### `
subsections stay attached to their parent `## ` chunk. Any section that
still exceeds ~500 tokens is split further with ~50 token overlap.

Chunks are embedded with sentence-transformers (all-MiniLM-L6-v2) via
`embed_texts()`, which is the single entry point for turning text into
vectors — reuse it at query time too. Using a different embedding model
(or a different wrapper) to embed queries than the one used to build the
index returns noise instead of relevant matches.

Persists a flat L2 FAISS index plus a position -> {chunk_id, text}
mapping to `data/guidelines/index/`, so retrieved FAISS row indices can
be mapped back to their source text (and citable chunk_id) at query time.
"""

import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDELINES_PATH = REPO_ROOT / "data" / "guidelines" / "icdrss_guidelines.md"
INDEX_DIR = REPO_ROOT / "data" / "guidelines" / "index"
INDEX_PATH = INDEX_DIR / "guidelines.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

_model = None  # lazily-loaded, module-level cache


def get_embedding_model():
    """Load (and cache) the sentence-transformers model.

    Called by embed_texts() below. Kept as its own function so a future
    query/retrieval script can share the exact same loaded model instead
    of re-instantiating it — the point isn't performance, it's making
    "same model at index and query time" the only code path available.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts):
    """Embed a list of strings with the shared embedding model.

    This is THE function to call for embeddings, at both index-build
    time (here) and query time (in the retriever, next). Never
    instantiate SentenceTransformer directly elsewhere.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype("float32")


def _approx_token_count(text):
    """Rough token estimate via whitespace word count.

    No tokenizer library is in requirements.txt, so this is a proxy
    (roughly 0.75-1 word per token for English) — good enough for a
    chunk-sizing heuristic, not an exact token count.
    """
    return len(text.split())


def _slugify(heading):
    """Turn a heading like 'Image Quality Criteria' into 'image_quality_criteria'."""
    slug = heading.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug.strip("_")


def _split_oversized_chunk(chunk_id, text, max_tokens=MAX_CHUNK_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS):
    """Split one chunk's text into ~max_tokens windows with overlap.

    Splits on whitespace-delimited words — the same unit _approx_token_count
    uses — so window sizes stay consistent with the threshold that
    triggered the split. Sub-chunks are labelled `{chunk_id}_part1`,
    `{chunk_id}_part2`, ...
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [(chunk_id, text)]

    parts = []
    start = 0
    part_num = 1
    step = max_tokens - overlap_tokens
    while start < len(words):
        end = min(start + max_tokens, len(words))
        parts.append((f"{chunk_id}_part{part_num}", " ".join(words[start:end])))
        if end == len(words):
            break
        start += step
        part_num += 1
    return parts


def parse_sections(markdown_text):
    """Split markdown into self-contained chunks along `## ` boundaries.

    Content before the first `## ` heading (the document title + intro
    paragraph) is document metadata, not retrievable clinical content —
    it's dropped rather than indexed as its own chunk. Nested `### `
    subsections stay inside their parent `## ` chunk — only `## ` starts
    a new chunk.

    Returns a list of (chunk_id, text) tuples, one per `## ` section, in
    document order.
    """
    lines = markdown_text.splitlines()
    section_starts = [i for i, line in enumerate(lines) if line.startswith("## ")]

    raw_sections = []

    for i, start in enumerate(section_starts):
        end = section_starts[i + 1] if i + 1 < len(section_starts) else len(lines)
        section_text = "\n".join(lines[start:end]).strip()
        heading = lines[start][3:].strip()  # strip leading "## "
        raw_sections.append((_slugify(heading), section_text))

    return raw_sections


def build_chunks(markdown_text):
    """Parse markdown into sections, then split any oversized section.

    Returns the final ordered list of (chunk_id, text) chunks ready to
    embed and index.
    """
    chunks = []
    for chunk_id, text in parse_sections(markdown_text):
        if _approx_token_count(text) <= MAX_CHUNK_TOKENS:
            chunks.append((chunk_id, text))
        else:
            chunks.extend(_split_oversized_chunk(chunk_id, text))
    return chunks


def build_index(guidelines_path=GUIDELINES_PATH, index_dir=INDEX_DIR):
    """Full indexing pipeline: parse -> chunk -> embed -> persist to disk."""
    guidelines_path = Path(guidelines_path)
    index_dir = Path(index_dir)

    markdown_text = guidelines_path.read_text(encoding="utf-8")
    chunks = build_chunks(markdown_text)

    chunk_ids = [chunk_id for chunk_id, _ in chunks]
    assert len(set(chunk_ids)) == len(chunk_ids), "Duplicate chunk_id detected"

    chunk_texts = [text for _, text in chunks]
    embeddings = embed_texts(chunk_texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "guidelines.faiss"))

    # Keys are the FAISS row index (as a string) so a search result's
    # returned index maps straight back to {chunk_id, text} — chunk_id
    # itself is kept as a field for human-readable citation.
    chunk_mapping = {
        str(i): {"chunk_id": chunk_ids[i], "text": chunk_texts[i]}
        for i in range(len(chunks))
    }
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_mapping, f, indent=2, ensure_ascii=False)

    token_counts = [_approx_token_count(text) for text in chunk_texts]
    print(f"Parsed {len(chunks)} chunks from {guidelines_path}")
    print(f"Chunk sizes (approx. tokens): min={min(token_counts)}, "
          f"max={max(token_counts)}, mean={sum(token_counts) / len(token_counts):.0f}")
    print(f"Embedding dimension: {embeddings.shape[1]} ({EMBEDDING_MODEL_NAME})")
    print(f"FAISS index saved to: {index_dir / 'guidelines.faiss'}")
    print(f"Chunk mapping saved to: {index_dir / 'chunks.json'}")

    return index, chunk_mapping


def load_index(index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
    """Load the persisted FAISS index and its chunk_id/text mapping.

    Shared by test_retrieval.py, eval_rag.py, and the Streamlit app —
    the single place this happens, so all three read the same on-disk
    format the same way.
    """
    index = faiss.read_index(str(index_path))
    with open(chunks_path, encoding="utf-8") as f:
        chunk_mapping = json.load(f)
    return index, chunk_mapping


def retrieve(query, index, chunk_mapping, top_k=3):
    """Embed a query with embed_texts() and return its top_k nearest chunks.

    Each result dict has rank, distance, chunk_id, and text.
    """
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx == -1:  # FAISS pads with -1 if top_k > number of indexed vectors
            continue
        entry = chunk_mapping[str(idx)]
        results.append({
            "rank": rank,
            "distance": float(dist),
            "chunk_id": entry["chunk_id"],
            "text": entry["text"],
        })
    return results


if __name__ == "__main__":
    build_index()
