"""Day 07 — End-to-end RAG with a real generative LLM (CPU-friendly small model).

`rag_agent_from_scratch.py` faked the last two steps: an extractive "generator"
and a rule-based policy. This script keeps Day 06/07's retrieval + MMR re-ranking
exactly as-is and plugs a *real* instruction-tuned LLM onto the retrieved context,
so it actually writes a grounded, cited answer.

A ~0.5B instruct model runs on a laptop CPU (slowly). Swap in any of the quantized
models from Days 02–05 on a GPU for real speed — the pipeline is unchanged.

Install:
    pip install transformers torch

Run:
    python rag_agent_llm.py
    python rag_agent_llm.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import os
import sys

# Day 06 retrieval + Day 07 re-ranking, reused unchanged.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "day-06-rag-basics"))
from rag_from_scratch import CORPUS, TfidfEmbedder, VectorStore, build_prompt, chunk_text
from rag_agent_from_scratch import grounding_score, mmr_rerank


def build_store() -> tuple[TfidfEmbedder, VectorStore]:
    embedder = TfidfEmbedder().fit([text for _id, text in CORPUS])
    store = VectorStore()
    for doc_id, text in CORPUS:
        for j, chunk in enumerate(chunk_text(text)):
            cid = doc_id if j == 0 else f"{doc_id}#{j}"
            store.add(cid, chunk, embedder.transform(chunk))
    return embedder, store


def load_generator(model_name: str):
    try:
        from transformers import pipeline
    except ImportError:
        sys.exit(
            "transformers is not installed.\n"
            "  pip install transformers torch\n"
            "(A ~0.5B instruct model runs on CPU; use a GPU + a Day 02–05 "
            "quantized model for speed.)"
        )
    # First run downloads the weights; then it is cached and offline.
    return pipeline("text-generation", model=model_name)


def answer(question: str, embedder, store, generator) -> str:
    """Full RAG: retrieve -> MMR re-rank -> grounded prompt -> LLM generates."""
    q_vec = embedder.transform(question)
    candidates = [doc for doc, _score in store.search(q_vec, k=5)]
    contexts = mmr_rerank(q_vec, candidates, k=3)

    prompt = build_prompt(question, [(d, 0.0) for d in contexts])
    messages = [{"role": "user", "content": prompt}]
    out = generator(messages, max_new_tokens=120, do_sample=False)
    generated = out[0]["generated_text"][-1]["content"].strip()

    ground = grounding_score(generated, contexts)
    used = ", ".join(d.doc_id for d in contexts)
    return f"{generated}\n   (retrieved: {used} | grounding={ground:.2f})"


def demo(model_name: str) -> None:
    embedder, store = build_store()
    generator = load_generator(model_name)

    questions = [
        "How do you shrink a model without any further training?",
        "Why is decoding long sequences memory-hungry?",
        "In one sentence, what problem does LoRA solve?",
    ]
    for q in questions:
        print("=" * 74)
        print(f"Q: {q}")
        print(answer(q, embedder, store, generator))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="any chat/instruct model id from the Hugging Face Hub",
    )
    demo(parser.parse_args().model)
