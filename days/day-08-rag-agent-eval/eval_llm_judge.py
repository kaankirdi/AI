"""Day 08 — LLM-as-a-judge: scoring RAG answers with a model (CPU-friendly).

`eval_from_scratch.py` measures faithfulness with word overlap — cheap, honest, and
blind to paraphrase. It marks a correct *reworded* answer as unfaithful because none
of the exact tokens match. The production fix is **LLM-as-a-judge**: hand a second
model the (question, context, answer) triple and ask *it* whether the answer is
supported and on-topic. The judge understands meaning, so a paraphrase that the
overlap metric punished scores correctly.

This keeps Day 06/07 retrieval + MMR unchanged, generates an answer with a small
instruct model, and then has the *same* model grade its own output on two axes:

    faithfulness — every claim traceable to the retrieved context? (catches drift)
    relevance    — does it actually answer the question?           (catches evasion)

The judge is prompted to return strict JSON so the scores are machine-readable — the
same trick that turns any LLM into an automated eval. A ~0.5B model is enough to
demonstrate the mechanics; a stronger judge (or a Days 02–05 quantized model on a
GPU) gives scores you would actually trust.

Install:
    pip install transformers torch

Run:
    python eval_llm_judge.py
    python eval_llm_judge.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Day 06 retrieval + Day 07 re-ranking, reused unchanged.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "day-06-rag-basics"))
sys.path.insert(0, os.path.join(_HERE, "..", "day-07-rag-agent"))
from rag_from_scratch import CORPUS, TfidfEmbedder, VectorStore, build_prompt, chunk_text
from rag_agent_from_scratch import mmr_rerank


def build_store() -> tuple[TfidfEmbedder, VectorStore]:
    embedder = TfidfEmbedder().fit([text for _id, text in CORPUS])
    store = VectorStore()
    for doc_id, text in CORPUS:
        for j, chunk in enumerate(chunk_text(text)):
            cid = doc_id if j == 0 else f"{doc_id}#{j}"
            store.add(cid, chunk, embedder.transform(chunk))
    return embedder, store


def load_llm(model_name: str):
    try:
        from transformers import pipeline
    except ImportError:
        sys.exit(
            "transformers is not installed.\n"
            "  pip install transformers torch\n"
            "(A ~0.5B instruct model runs on CPU; use a GPU + a Day 02–05 "
            "quantized model for speed.)"
        )
    return pipeline("text-generation", model=model_name)


def generate_answer(question: str, embedder, store, llm) -> tuple[str, list]:
    """Full RAG: retrieve -> MMR -> grounded prompt -> LLM writes the answer."""
    q_vec = embedder.transform(question)
    candidates = [doc for doc, _s in store.search(q_vec, k=5)]
    contexts = mmr_rerank(q_vec, candidates, k=3)
    prompt = build_prompt(question, [(d, 0.0) for d in contexts])
    out = llm([{"role": "user", "content": prompt}], max_new_tokens=120, do_sample=False)
    answer = out[0]["generated_text"][-1]["content"].strip()
    return answer, contexts


_JUDGE_TEMPLATE = (
    "You are a strict evaluator of a retrieval-augmented answer. Score it on:\n"
    "  faithfulness: 0.0-1.0, is EVERY claim supported by the context below?\n"
    "  relevance:    0.0-1.0, does it actually answer the question?\n"
    "Reply with ONLY a JSON object: "
    '{{"faithfulness": <float>, "relevance": <float>, "reason": "<short>"}}\n\n'
    "### Context\n{context}\n\n"
    "### Question\n{question}\n\n"
    "### Answer to grade\n{answer}\n\n"
    "### JSON\n"
)


def judge(question: str, answer: str, contexts: list, llm) -> dict:
    """Ask the LLM to grade an answer and parse its JSON verdict.

    Robust parsing matters: even a well-prompted small model wraps its JSON in
    prose. We pull out the first {...} block and fall back to a neutral score if
    the model refuses to comply — an eval harness must never crash on one bad row.
    """
    context = "\n\n".join(f"(source: {d.doc_id}) {d.text}" for d in contexts)
    prompt = _JUDGE_TEMPLATE.format(context=context, question=question, answer=answer)
    out = llm([{"role": "user", "content": prompt}], max_new_tokens=120, do_sample=False)
    raw = out[0]["generated_text"][-1]["content"]

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            verdict = json.loads(match.group(0))
            return {
                "faithfulness": float(verdict.get("faithfulness", 0.0)),
                "relevance": float(verdict.get("relevance", 0.0)),
                "reason": str(verdict.get("reason", "")).strip(),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return {"faithfulness": 0.0, "relevance": 0.0, "reason": f"unparseable: {raw[:60]!r}"}


def demo(model_name: str) -> None:
    embedder, store = build_store()
    llm = load_llm(model_name)

    questions = [
        "How do you shrink a model without any further training?",
        "Why is decoding long sequences memory-hungry?",
        "In one sentence, what problem does LoRA solve?",
    ]

    faith, relev = [], []
    for q in questions:
        answer, contexts = generate_answer(q, embedder, store, llm)
        verdict = judge(q, answer, contexts, llm)
        faith.append(verdict["faithfulness"])
        relev.append(verdict["relevance"])
        print("=" * 74)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print(f"  judge -> faithfulness={verdict['faithfulness']:.2f} "
              f"relevance={verdict['relevance']:.2f}  ({verdict['reason']})")

    print("=" * 74)
    print(f"MEAN  faithfulness={sum(faith) / len(faith):.2f}  "
          f"relevance={sum(relev) / len(relev):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="any chat/instruct model id from the Hugging Face Hub",
    )
    demo(parser.parse_args().model)
