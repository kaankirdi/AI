"""Day 06 — Retrieval-Augmented Generation, from scratch (CPU, NumPy only).

RAG = retrieve relevant context from a knowledge base, then hand it to an LLM so
the answer is *grounded* in real documents instead of the model's parametric
memory. This file builds the whole retrieval half by hand — no model download,
no vector-DB dependency — so the mechanics are visible:

    chunk  ->  embed  ->  index  ->  retrieve (cosine)  ->  assemble prompt

The embedder here is a deterministic TF-IDF vectorizer. That is enough to show
the pipeline end to end, and it also exposes TF-IDF's weakness (it matches
*words*, not *meaning*) — which is exactly what Day 06's second script fixes with
real semantic embeddings.

Run:
    python rag_from_scratch.py
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# A tiny knowledge base. Deliberately small so you can eyeball every retrieval.
# --------------------------------------------------------------------------- #
CORPUS = [
    ("nf4", "NF4 is a 4-bit data type used by QLoRA. Its levels are spaced to "
            "match a normal distribution, so it stores neural-network weights "
            "more accurately than plain 4-bit integers."),
    ("lora", "LoRA freezes the pretrained weights and trains two small low-rank "
             "matrices A and B instead. This cuts the number of trainable "
             "parameters to well under one percent of the full model."),
    ("kv_cache", "The KV cache stores the key and value tensors for every past "
                 "token during autoregressive decoding. It grows linearly with "
                 "sequence length and dominates memory in long-context inference."),
    ("gptq", "GPTQ is a post-training quantization method. It quantizes weights "
             "one column at a time and uses second-order information to correct "
             "the error introduced in the columns that remain."),
    ("awq", "AWQ is activation-aware weight quantization. It notices that a small "
            "fraction of weight channels are far more important, and scales them "
            "up before quantizing so their precision is preserved."),
    ("embeddings", "An embedding maps a piece of text to a dense vector so that "
                   "texts with similar meaning land close together. Retrieval "
                   "works by comparing these vectors with cosine similarity."),
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


def chunk_text(text: str, max_tokens: int = 40, overlap: int = 8) -> list[str]:
    """Split long text into overlapping windows of whole words.

    Real corpora hold documents far larger than a model's context. Chunking
    keeps each retrievable unit small and self-contained; the overlap stops a
    sentence that straddles a boundary from being lost. Our demo docs already
    fit in one chunk, but the machinery is here so the pipeline is honest.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    chunks, start = [], 0
    step = max_tokens - overlap
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_tokens]))
        start += step
    return chunks


# --------------------------------------------------------------------------- #
# TF-IDF embedder — deterministic, dependency-free, CPU-only.
# --------------------------------------------------------------------------- #
@dataclass
class TfidfEmbedder:
    """A bag-of-words TF-IDF vectorizer with L2-normalized output vectors.

    term frequency (tf)  : how often a term appears in a document
    inverse doc freq (idf): down-weights terms common across the whole corpus
    Normalizing to unit length turns the dot product into cosine similarity.
    """

    vocab: dict[str, int] = field(default_factory=dict)
    idf: np.ndarray | None = None

    def fit(self, documents: list[str]) -> "TfidfEmbedder":
        # Build the vocabulary and count in how many documents each term appears.
        doc_freq: Counter[str] = Counter()
        for doc in documents:
            for term in set(tokenize(doc)):
                doc_freq[term] += 1
        self.vocab = {term: i for i, term in enumerate(sorted(doc_freq))}

        # Smoothed idf: log((1 + N) / (1 + df)) + 1  — the sklearn convention.
        n_docs = len(documents)
        self.idf = np.ones(len(self.vocab), dtype=np.float64)
        for term, i in self.vocab.items():
            self.idf[i] = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
        return self

    def transform(self, text: str) -> np.ndarray:
        """Embed one text into an L2-normalized TF-IDF vector."""
        assert self.idf is not None, "call fit() before transform()"
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        counts = Counter(tokenize(text))
        for term, count in counts.items():
            idx = self.vocab.get(term)
            if idx is not None:  # out-of-vocabulary query words are ignored
                vec[idx] = count * self.idf[idx]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# --------------------------------------------------------------------------- #
# Vector store — the in-memory index a real system would swap for FAISS / a DB.
# --------------------------------------------------------------------------- #
@dataclass
class Document:
    doc_id: str
    text: str
    vector: np.ndarray


class VectorStore:
    """Holds document vectors and does brute-force cosine top-k search.

    Vectors are unit-normalized, so cosine similarity is a single matrix-vector
    product. FAISS / Chroma / pgvector do the same thing with an approximate
    index once you have millions of vectors — the idea is identical.
    """

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._matrix: np.ndarray | None = None

    def add(self, doc_id: str, text: str, vector: np.ndarray) -> None:
        self._docs.append(Document(doc_id, text, vector))
        self._matrix = None  # invalidate the stacked cache

    def _stacked(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.vstack([d.vector for d in self._docs])
        return self._matrix

    def search(self, query_vec: np.ndarray, k: int = 3) -> list[tuple[Document, float]]:
        scores = self._stacked() @ query_vec           # cosine, vectors are unit-norm
        top = np.argsort(scores)[::-1][:k]
        return [(self._docs[i], float(scores[i])) for i in top]


# --------------------------------------------------------------------------- #
# Prompt assembly — the "augmented" in Retrieval-Augmented Generation.
# --------------------------------------------------------------------------- #
def build_prompt(question: str, retrieved: list[tuple[Document, float]]) -> str:
    """Stitch retrieved chunks into a grounded, cite-able prompt for an LLM.

    The instruction to answer *only* from the context — and to say when the
    context is insufficient — is what turns retrieval into reduced hallucination.
    """
    context_blocks = [
        f"[{i + 1}] (source: {doc.doc_id})\n{doc.text}"
        for i, (doc, _score) in enumerate(retrieved)
    ]
    context = "\n\n".join(context_blocks)
    return (
        "Answer the question using ONLY the context below. "
        "Cite sources as [n]. If the context does not contain the answer, "
        "say you don't know.\n\n"
        f"### Context\n{context}\n\n"
        f"### Question\n{question}\n\n"
        "### Answer\n"
    )


def demo() -> None:
    embedder = TfidfEmbedder().fit([text for _id, text in CORPUS])

    store = VectorStore()
    for doc_id, text in CORPUS:
        for j, chunk in enumerate(chunk_text(text)):
            chunk_id = doc_id if j == 0 else f"{doc_id}#{j}"
            store.add(chunk_id, chunk, embedder.transform(chunk))

    queries = [
        "How does QLoRA store weights in 4 bits?",   # lexical overlap -> should hit nf4
        "What makes long context expensive in memory?",  # -> kv_cache
        "Which method keeps the important weight channels precise?",  # -> awq
        "How do you compress a model without retraining it?",  # semantic gap -> TF-IDF struggles
    ]

    for q in queries:
        results = store.search(embedder.transform(q), k=3)
        print("=" * 74)
        print(f"Q: {q}")
        print(f"{'rank':>4}  {'score':>6}  source")
        print("-" * 32)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"{rank:>4}  {score:>6.3f}  {doc.doc_id}")
        top_doc, top_score = results[0]
        if top_score == 0.0:
            print("  -> no lexical overlap: TF-IDF retrieved nothing relevant.")

    # Show the assembled prompt for the first query so the "augment" step is concrete.
    q = queries[0]
    print("\n" + "=" * 74)
    print("ASSEMBLED PROMPT (fed to an LLM — GPU or an API — for generation):\n")
    print(build_prompt(q, store.search(embedder.transform(q), k=2)))


if __name__ == "__main__":
    demo()
