"""Day 07 — End-to-end RAG + a tool-using agent, from scratch (CPU, NumPy only).

Day 06 stopped at retrieval. This closes the loop into an *answer*, and then wraps
the whole thing in a tool-using agent — all without an LLM download, so every
decision is inspectable:

    retrieve (Day 06)  ->  re-rank (MMR)  ->  answer (extractive)  ->  grounding score
                                                        |
                                          wrapped as a tool the agent can call

Two things stand in for a real LLM here, on purpose:
  * the *generator* is extractive — it selects supporting sentences instead of
    writing new ones (Day 07's second script swaps in a real generative model);
  * the *agent policy* is rule-based — it routes each question to a tool by
    inspecting it, where a real ReAct agent would let the LLM emit the same
    Thought / Action / Observation trace.

Keeping both mechanical makes the *plumbing* — re-ranking, grounding, the tool
loop — the thing you actually see. Run:

    python rag_agent_from_scratch.py
"""
from __future__ import annotations

import ast
import operator
import os
import re
import sys

import numpy as np

# Day 06 owns the retrieval stack; put its folder on the path and import from it
# so Day 07 only adds the steps *after* retrieval.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "day-06-rag-basics")
)

# Reuse Day 06's retrieval stack verbatim — only the steps *after* retrieval are new.
from rag_from_scratch import (
    CORPUS,
    TfidfEmbedder,
    VectorStore,
    chunk_text,
    tokenize,
)


# --------------------------------------------------------------------------- #
# 1. Re-ranking with Maximal Marginal Relevance (MMR).
# --------------------------------------------------------------------------- #
def mmr_rerank(
    query_vec: np.ndarray,
    candidates: list,
    k: int = 3,
    lambda_: float = 0.7,
) -> list:
    """Pick k documents that are relevant *and* non-redundant.

    Plain top-k retrieval happily returns three near-duplicate chunks. MMR trades
    a little relevance for diversity so the context covers more ground:

        MMR(d) = λ · sim(d, query) − (1 − λ) · max sim(d, already-picked)

    Every vector is unit-norm, so each `sim` is one dot product.
    """
    selected: list = []
    remaining = list(candidates)
    while remaining and len(selected) < k:
        best, best_score = None, -np.inf
        for doc in remaining:
            relevance = float(doc.vector @ query_vec)
            redundancy = max(
                (float(doc.vector @ s.vector) for s in selected), default=0.0
            )
            score = lambda_ * relevance - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = doc, score
        selected.append(best)
        remaining.remove(best)
    return selected


# --------------------------------------------------------------------------- #
# 2. Extractive "generation" — stands in for an LLM, no download needed.
# --------------------------------------------------------------------------- #
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def extractive_answer(
    question: str, embedder: TfidfEmbedder, contexts: list, max_sentences: int = 2
) -> tuple[str, list[str]]:
    """Answer by selecting the sentences most similar to the question.

    Returns the answer string (with [source] tags) and the list of source ids it
    drew from. A real generator would paraphrase these; extraction keeps the demo
    honest and 100% traceable to the source text.
    """
    q_vec = embedder.transform(question)
    scored = []
    for doc in contexts:
        for sent in _SENT_RE.split(doc.text.strip()):
            if not sent:
                continue
            sim = float(embedder.transform(sent) @ q_vec)
            scored.append((sim, sent.strip(), doc.doc_id))

    scored.sort(key=lambda t: t[0], reverse=True)
    picked = [s for s in scored[:max_sentences] if s[0] > 0]
    if not picked:
        return "I don't know — the knowledge base has nothing relevant.", []

    answer = " ".join(f"{sent} [{src}]" for _sim, sent, src in picked)
    sources = list(dict.fromkeys(src for _sim, _sent, src in picked))
    return answer, sources


def grounding_score(answer: str, contexts: list) -> float:
    """Fraction of the answer's content words that appear in the retrieved context.

    A blunt proxy for "is this answer supported by the sources?" — 1.0 means every
    content token is traceable, low values flag a drifting / hallucinated answer.
    """
    context_tokens = set()
    for doc in contexts:
        context_tokens.update(tokenize(doc.text))
    ans_tokens = [t for t in tokenize(answer) if len(t) > 2]  # skip [ ] and stopwordy bits
    if not ans_tokens:
        return 0.0
    supported = sum(t in context_tokens for t in ans_tokens)
    return supported / len(ans_tokens)


# --------------------------------------------------------------------------- #
# 3. Tools the agent can call.
# --------------------------------------------------------------------------- #
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate an arithmetic AST with a whitelist — never Python's eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """A math tool. Real agents pair an LLM (bad at arithmetic) with one of these."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
        return f"could not evaluate {expression!r}"


class KnowledgeBaseTool:
    """Wraps the whole RAG pipeline (retrieve -> rerank -> answer) as one tool."""

    def __init__(self) -> None:
        self.embedder = TfidfEmbedder().fit([text for _id, text in CORPUS])
        self.store = VectorStore()
        for doc_id, text in CORPUS:
            for j, chunk in enumerate(chunk_text(text)):
                cid = doc_id if j == 0 else f"{doc_id}#{j}"
                self.store.add(cid, chunk, self.embedder.transform(chunk))

    def __call__(self, query: str) -> str:
        q_vec = self.embedder.transform(query)
        candidates = [doc for doc, _score in self.store.search(q_vec, k=5)]
        reranked = mmr_rerank(q_vec, candidates, k=3)
        answer, _sources = extractive_answer(query, self.embedder, reranked)
        ground = grounding_score(answer, reranked)
        return f"{answer}   (grounding={ground:.2f})"


# --------------------------------------------------------------------------- #
# 4. The agent loop — ReAct-style, rule-based policy.
# --------------------------------------------------------------------------- #
_MATH_RE = re.compile(r"^[\d\s()+\-*/.%^]+$")


def choose_tool(question: str) -> tuple[str, str]:
    """The 'policy'. A real ReAct agent gets this decision from an LLM's output.

    Returns (tool_name, tool_input).
    """
    # Extract a bare arithmetic expression if the question is (or contains) one.
    math_match = re.search(r"[\d.]+(?:\s*[-+*/%^]\s*[\d.]+)+", question)
    if _MATH_RE.match(question.strip()) or (
        math_match and any(w in question.lower() for w in ("what is", "calculate", "compute", "="))
    ):
        return "calculator", (math_match.group(0) if math_match else question).replace("^", "**")
    return "knowledge_base", question


def run_agent(questions: list[str]) -> None:
    tools = {"knowledge_base": KnowledgeBaseTool(), "calculator": calculator}

    for q in questions:
        print("=" * 74)
        print(f"User: {q}")
        tool_name, tool_input = choose_tool(q)
        # The Thought/Action/Observation trace an LLM agent would emit itself.
        print(f"  Thought: this needs the {tool_name} tool.")
        print(f"  Action: {tool_name}({tool_input!r})")
        observation = tools[tool_name](tool_input)
        print(f"  Observation: {observation}")
        print(f"  Final Answer: {observation}")


def demo() -> None:
    questions = [
        "What data type does QLoRA use to store weights?",
        "Why does long-context inference use so much memory?",
        "What is 4096 * 32 / 1024?",           # routed to the calculator
        "Which quantization method is activation-aware?",
    ]
    run_agent(questions)


if __name__ == "__main__":
    demo()
