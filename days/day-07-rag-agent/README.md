# Day 07 — End-to-End RAG + a Tool-Using Agent

> **Runs on:** `rag_agent_from_scratch.py` → any CPU, no download.
> `rag_agent_llm.py` → CPU (slow) or GPU, one small model download.

## 🎯 Goal
Turn Day 06's retrieval into a full answer, then wrap it in an agent. Add the two
pieces a real RAG system needs past "fetch some chunks": **re-ranking** for better
context and a **grounding** check to catch drift — plus a **tool-using agent loop**
that decides *which* tool to reach for.

## 🧠 Theory (short)
Day 06 got us `retrieve`. A production pipeline adds two steps around generation:

```
retrieve (top-n)  ->  re-rank (MMR, top-k)  ->  generate (LLM)  ->  grounding check
```

- **MMR re-ranking** — top-k by similarity often returns near-duplicates. *Maximal
  Marginal Relevance* trades a little relevance for diversity so the context covers
  more ground:

  ```
  MMR(d) = λ · sim(d, query) − (1 − λ) · max sim(d, already-picked)
  ```

- **Grounding score** — a blunt "is this answer supported by the sources?" proxy:
  the fraction of the answer's content words that appear in the retrieved context.
  Low grounding flags a hallucinated or drifting answer.

An **agent** goes one level up: instead of always running RAG, a *policy* looks at
the question and picks a **tool** — the knowledge base, a calculator, a web search —
runs it, reads the result, and repeats. That Thought → Action → Observation loop is
**ReAct**. The LLM is bad at arithmetic; pairing it with a calculator tool is the
canonical fix.

## 💻 Code
- [`rag_agent_from_scratch.py`](./rag_agent_from_scratch.py) — **CPU, no download.**
  Imports Day 06's retrieval, then adds MMR re-ranking, an *extractive* generator
  (selects supporting sentences — an honest, traceable stand-in for an LLM), a
  grounding score, and a rule-based ReAct agent routing between a `knowledge_base`
  tool and a safe `calculator`.
- [`rag_agent_llm.py`](./rag_agent_llm.py) — **CPU/GPU.** Same retrieval + MMR, but
  a real instruction-tuned LLM writes the grounded, cited answer. Swap in a Day
  02–05 quantized model on a GPU for speed.

```bash
python rag_agent_from_scratch.py              # pure NumPy, runs anywhere
pip install transformers torch
python rag_agent_llm.py                        # real generation (default: Qwen2.5-0.5B-Instruct)
```

## 📊 Result (from-scratch)
```
User: What data type does QLoRA use to store weights?
  Thought: this needs the knowledge_base tool.
  Action: knowledge_base('What data type does QLoRA use to store weights?')
  Observation: NF4 is a 4-bit data type used by QLoRA. [nf4] Its levels are spaced
               to match a normal distribution... [nf4]   (grounding=1.00)

User: What is 4096 * 32 / 1024?
  Thought: this needs the calculator tool.
  Action: calculator('4096 * 32 / 1024')
  Observation: 128.0
```

The agent routes the knowledge question to RAG (grounding **1.00** — every word
traces to a source) and the arithmetic to the calculator, which the LLM path would
otherwise get wrong.

## 🔍 Observations
- **MMR matters most when the corpus has overlap** — with near-duplicate chunks,
  plain top-k wastes the context window on redundancy MMR would skip.
- **Grounding is a cheap guardrail.** Even this word-overlap proxy separates a
  supported answer (≈1.0) from one that wandered into an off-topic chunk (<0.6);
  a real system gates the response on it.
- **Extraction can't paraphrase** — that's the ceiling `rag_agent_llm.py` lifts.
  The *retrieval quality* still caps the LLM: bad chunks in → confident nonsense out.
- **Tools beat parametric knowledge for exact facts** — arithmetic, dates, lookups.
  The agent's whole job is knowing when to stop guessing and call one.
- The policy here is rule-based; on a GPU/API the LLM emits the same
  Thought/Action/Observation lines and you parse them — identical loop, smarter router.

## ➡️ Next step
Phase 3 continues: replace the in-memory store with a real **vector database**
(FAISS / Chroma), add an **embedding-based re-ranker** (cross-encoder), and give the
agent **multi-hop** reasoning — chaining several tool calls before it answers.
