"""Day 08 — Evaluating a RAG + agent system, from scratch (CPU, NumPy only).

Days 06–07 *built* a retrieval → re-rank → answer → agent pipeline. This asks the
question that decides whether any of it ships: **is it any good, and how would you
know it got worse?** No model download — every metric is computed by hand so you
can see exactly what each number rewards and where it lies to you.

You cannot improve what you cannot measure, and a RAG system fails in three
different places, so it needs three different rulers:

    retrieval  ->  did we fetch the right chunks?     recall@k, MRR, nDCG@k
    generation ->  is the answer supported + on-topic? faithfulness, relevance
    agent      ->  did the policy pick the right tool? routing accuracy

Each needs a small **labeled** set: queries paired with the answer key (which docs
are relevant, which tool is correct). That golden set — not the model — is the hard
part of evaluation in the real world; here it is a handful of hand-written rows so
the arithmetic stays inspectable.

Run:
    python eval_from_scratch.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

# Days 06 (retrieval) and 07 (rerank/answer/agent) own the system under test.
# Put both on the path and import the pieces verbatim — Day 08 only *scores* them.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "day-06-rag-basics"))
sys.path.insert(0, os.path.join(_HERE, "..", "day-07-rag-agent"))

from rag_from_scratch import CORPUS, TfidfEmbedder, VectorStore, chunk_text  # noqa: E402
from rag_agent_from_scratch import (  # noqa: E402
    choose_tool,
    extractive_answer,
    grounding_score,
    mmr_rerank,
)


# --------------------------------------------------------------------------- #
# The golden set. In production this is human-labeled and it is the whole ball
# game; here it is tiny and hand-written so every score can be checked by eye.
#   relevant : the doc_ids that genuinely answer the query (the retrieval key)
#   tool     : the tool the agent policy *should* choose (the routing key)
# --------------------------------------------------------------------------- #
EVAL_SET = [
    {"query": "How does QLoRA store weights in 4 bits?",
     "relevant": {"nf4"}, "tool": "knowledge_base"},
    {"query": "What makes long-context inference expensive in memory?",
     "relevant": {"kv_cache"}, "tool": "knowledge_base"},
    {"query": "Which quantization method keeps the important weight channels precise?",
     "relevant": {"awq"}, "tool": "knowledge_base"},
    {"query": "How does LoRA cut the number of trainable parameters?",
     "relevant": {"lora"}, "tool": "knowledge_base"},
    {"query": "How does GPTQ correct the error it introduces while quantizing?",
     "relevant": {"gptq"}, "tool": "knowledge_base"},
    {"query": "What is 4096 * 32 / 1024?",
     "relevant": set(), "tool": "calculator"},
]


# --------------------------------------------------------------------------- #
# 1. Retrieval metrics — computed by hand from a ranked list of doc_ids.
# --------------------------------------------------------------------------- #
def _base_id(chunk_id: str) -> str:
    """Strip a chunk suffix ('nf4#1' -> 'nf4') so scoring is per source document."""
    return chunk_id.split("#", 1)[0]


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Of the top-k retrieved, what fraction are relevant?"""
    if k == 0:
        return 0.0
    topk = ranked[:k]
    hits = sum(doc_id in relevant for doc_id in topk)
    return hits / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Of all relevant docs, what fraction made it into the top-k?"""
    if not relevant:
        return 1.0  # nothing to find -> vacuously perfect
    hits = sum(doc_id in relevant for doc_id in ranked[:k])
    return hits / len(relevant)


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant doc is in the top-k, else 0.0."""
    return float(any(doc_id in relevant for doc_id in ranked[:k]))


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / (rank of the first relevant hit). Rewards putting the answer *first*.

    Averaged over queries this is the classic MRR — the single number that best
    tracks "does the right chunk land at the top?"
    """
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance.

    DCG discounts a hit by log2(rank+1), so a relevant doc at rank 3 is worth less
    than at rank 1. Dividing by the ideal DCG (all hits stacked at the top) puts
    every query on a 0–1 scale, which is what makes nDCG averageable across queries
    with different numbers of relevant docs.
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
# 2. Generation metrics — faithfulness (already built in Day 07) + relevance.
# --------------------------------------------------------------------------- #
def answer_relevance(answer: str, query: str, embedder: TfidfEmbedder) -> float:
    """Cosine similarity between the answer and the question.

    Faithfulness asks "is the answer supported by the sources?"; relevance asks the
    orthogonal question "does it actually address what was asked?" A perfectly
    grounded answer to the *wrong* question scores high on faithfulness and low here
    — which is exactly why you need both.
    """
    a_vec = embedder.transform(answer)
    q_vec = embedder.transform(query)
    return float(a_vec @ q_vec)  # both are unit-norm, so the dot product is cosine


# --------------------------------------------------------------------------- #
# 3. The harness: build the system once, then score it against the golden set.
# --------------------------------------------------------------------------- #
def build_system() -> tuple[TfidfEmbedder, VectorStore]:
    embedder = TfidfEmbedder().fit([text for _id, text in CORPUS])
    store = VectorStore()
    for doc_id, text in CORPUS:
        for j, chunk in enumerate(chunk_text(text)):
            cid = doc_id if j == 0 else f"{doc_id}#{j}"
            store.add(cid, chunk, embedder.transform(chunk))
    return embedder, store


def evaluate_retrieval(embedder, store, k: int = 3) -> None:
    """Score retrieval on every knowledge-base query and macro-average."""
    kb_rows = [r for r in EVAL_SET if r["relevant"]]
    agg = {"P@k": [], "R@k": [], "Hit@k": [], "RR": [], "nDCG@k": []}

    print("=" * 74)
    print(f"RETRIEVAL  (top-k = {k}, macro-averaged over {len(kb_rows)} queries)")
    print(f"{'P@k':>6} {'R@k':>6} {'Hit@k':>6} {'RR':>6} {'nDCG':>6}  query")
    print("-" * 74)
    for row in kb_rows:
        q_vec = embedder.transform(row["query"])
        ranked = [_base_id(doc.doc_id) for doc, _s in store.search(q_vec, k=k)]
        rel = row["relevant"]
        p = precision_at_k(ranked, rel, k)
        r = recall_at_k(ranked, rel, k)
        h = hit_at_k(ranked, rel, k)
        rr = reciprocal_rank(ranked, rel)
        n = ndcg_at_k(ranked, rel, k)
        for name, val in zip(agg, (p, r, h, rr, n)):
            agg[name].append(val)
        print(f"{p:>6.2f} {r:>6.2f} {h:>6.2f} {rr:>6.2f} {n:>6.2f}  {row['query'][:38]}")

    print("-" * 74)
    means = {name: float(np.mean(vals)) for name, vals in agg.items()}
    print(f"{means['P@k']:>6.2f} {means['R@k']:>6.2f} {means['Hit@k']:>6.2f} "
          f"{means['RR']:>6.2f} {means['nDCG@k']:>6.2f}  << MEAN "
          f"(the 'RR' column averaged is MRR)")


def evaluate_generation(embedder, store, k: int = 3) -> None:
    """Score the extractive answer on faithfulness and relevance.

    Includes one deliberately hallucinated answer so you can watch faithfulness
    catch what relevance can't, and vice-versa.
    """
    kb_rows = [r for r in EVAL_SET if r["relevant"]]
    faith, relev = [], []

    print("\n" + "=" * 74)
    print("GENERATION  (faithfulness = grounded in sources, relevance = on-topic)")
    print(f"{'faith':>6} {'relev':>6}  answer")
    print("-" * 74)
    for row in kb_rows:
        q_vec = embedder.transform(row["query"])
        candidates = [doc for doc, _s in store.search(q_vec, k=5)]
        reranked = mmr_rerank(q_vec, candidates, k=k)
        answer, _src = extractive_answer(row["query"], embedder, reranked)
        f = grounding_score(answer, reranked)
        rv = answer_relevance(answer, row["query"], embedder)
        faith.append(f)
        relev.append(rv)
        print(f"{f:>6.2f} {rv:>6.2f}  {answer[:52]}")

    # A planted failure: a fluent answer to query[0] that the sources never support.
    row = kb_rows[0]
    q_vec = embedder.transform(row["query"])
    reranked = mmr_rerank(q_vec, [d for d, _s in store.search(q_vec, k=5)], k=k)
    bogus = "QLoRA stores its weights as 8-bit floating point values on the GPU."
    print("-" * 74)
    print(f"{grounding_score(bogus, reranked):>6.2f} "
          f"{answer_relevance(bogus, row['query'], embedder):>6.2f}  "
          f"[PLANTED HALLUCINATION] {bogus[:34]}")

    print("-" * 74)
    print(f"{np.mean(faith):>6.2f} {np.mean(relev):>6.2f}  << MEAN over real answers")


def evaluate_agent() -> None:
    """Score the agent's tool-routing policy: predicted tool vs the golden tool."""
    correct = 0
    print("\n" + "=" * 74)
    print("AGENT ROUTING  (did the policy pick the right tool?)")
    print(f"{'gold':>13}  {'chosen':>13}   query")
    print("-" * 74)
    for row in EVAL_SET:
        chosen, _inp = choose_tool(row["query"])
        ok = chosen == row["tool"]
        correct += ok
        mark = "✓" if ok else "✗"
        print(f"{row['tool']:>13}  {chosen:>13} {mark} {row['query'][:34]}")
    print("-" * 74)
    print(f"routing accuracy: {correct}/{len(EVAL_SET)} = {correct / len(EVAL_SET):.2f}")


def main() -> None:
    embedder, store = build_system()
    evaluate_retrieval(embedder, store, k=3)
    evaluate_generation(embedder, store, k=3)
    evaluate_agent()


if __name__ == "__main__":
    main()
