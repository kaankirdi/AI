"""Day 06 — RAG with real semantic embeddings (CPU-friendly, small download).

`rag_from_scratch.py` built the pipeline with TF-IDF and showed its ceiling: it
matches *words*, not *meaning*, so a query phrased differently from the source
document retrieves poorly. This script swaps in a real sentence-embedding model
(`all-MiniLM-L6-v2`, ~90 MB, runs fine on a laptop CPU) and keeps everything else
identical — same chunking, same cosine search, same grounded-prompt assembly.

The point is to feel the difference embeddings make on semantically-phrased
queries, not lexically-matching ones.

Install (CPU is fine):
    pip install sentence-transformers

Run:
    python rag_semantic.py
"""
from __future__ import annotations

import sys

import numpy as np

# Reuse the corpus, chunker, store, and prompt builder from the scratch version
# so the ONLY thing that changes is the embedder.
from rag_from_scratch import CORPUS, VectorStore, build_prompt, chunk_text


def load_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(
            "sentence-transformers is not installed.\n"
            "  pip install sentence-transformers\n"
            "(CPU wheels are fine — the model is ~90 MB and runs on a laptop.)"
        )
    # First run downloads the weights; afterwards it is cached and offline.
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(model, text: str) -> np.ndarray:
    """Embed one text into an L2-normalized vector (so dot product == cosine)."""
    return model.encode(text, normalize_embeddings=True).astype(np.float64)


def demo() -> None:
    model = load_model()

    store = VectorStore()
    for doc_id, text in CORPUS:
        for j, chunk in enumerate(chunk_text(text)):
            chunk_id = doc_id if j == 0 else f"{doc_id}#{j}"
            store.add(chunk_id, chunk, embed(model, chunk))

    # These queries share almost no vocabulary with their target documents —
    # the case where TF-IDF fell over and semantics should win.
    queries = [
        ("How do you shrink a model without any further training?", "gptq/awq"),
        ("Why does generating long outputs eat so much memory?", "kv_cache"),
        ("What trick lets you fine-tune with very few trainable weights?", "lora"),
        ("How is text turned into vectors that capture meaning?", "embeddings"),
    ]

    for q, expected in queries:
        results = store.search(embed(model, q), k=3)
        print("=" * 74)
        print(f"Q: {q}")
        print(f"   (expected top source: {expected})")
        print(f"{'rank':>4}  {'score':>6}  source")
        print("-" * 32)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"{rank:>4}  {score:>6.3f}  {doc.doc_id}")

    q = queries[0][0]
    print("\n" + "=" * 74)
    print("ASSEMBLED PROMPT (ready for an LLM to generate a grounded answer):\n")
    print(build_prompt(q, store.search(embed(model, q), k=2)))


if __name__ == "__main__":
    demo()
