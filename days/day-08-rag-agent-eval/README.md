# Day 08 — Evaluating a RAG + Agent System

> **Runs on:** `eval_from_scratch.py` → any CPU, no download.
> `eval_llm_judge.py` → CPU (slow) or GPU, one small model download.

## 🎯 Goal
Days 06–07 *built* a retrieval → re-rank → answer → agent pipeline. Now measure it.
You can't improve — or even notice a regression in — what you don't score, so this
day builds the three rulers a RAG system actually needs, by hand, and points them
at the system from the previous two days.

## 🧠 Theory (short)
A RAG system fails in three different places, so one number can't grade it:

```
retrieval  ->  did we fetch the right chunks?      recall@k · MRR · nDCG@k
generation ->  is the answer supported + on-topic?  faithfulness · relevance
agent      ->  did the policy pick the right tool?  routing accuracy
```

- **Retrieval** — against a labeled set (query → which docs are relevant):
  - **recall@k** — of the relevant docs, how many made the top-k. *Did we fetch it?*
  - **MRR** — mean of `1 / rank of the first hit`. *Did it land near the top?*
  - **nDCG@k** — like MRR but discounts every hit by `log2(rank+1)` and normalizes
    to 0–1, so queries with different numbers of relevant docs stay comparable.

- **Generation** — two orthogonal axes; you need both:
  - **faithfulness** — is every claim supported by the retrieved context? (catches
    hallucination / drift)
  - **relevance** — does the answer actually address the question? (catches a
    perfectly-grounded answer to the *wrong* question)

- **Agent** — **routing accuracy**: predicted tool vs the correct tool, over the set.

The catch: word-overlap faithfulness is blind to paraphrase — a correctly *reworded*
answer scores low because no exact tokens match. The production fix is
**LLM-as-a-judge**: a second model reads `(question, context, answer)` and grades it,
returning strict JSON so the verdict is machine-readable.

## 💻 Code
- [`eval_from_scratch.py`](./eval_from_scratch.py) — **CPU, no download.** Imports the
  Day 06/07 system verbatim and scores it: retrieval metrics (recall@k, precision@k,
  hit@k, MRR, nDCG@k), generation metrics (faithfulness, relevance), and agent
  routing accuracy — every metric implemented by hand. Includes a **planted
  hallucination** so you can watch faithfulness catch it.
- [`eval_llm_judge.py`](./eval_llm_judge.py) — **CPU/GPU.** Generates an answer with a
  real instruct model, then has an LLM grade faithfulness + relevance and parse its
  JSON verdict. Robust to a small model wrapping its JSON in prose.

```bash
python eval_from_scratch.py                    # pure NumPy, runs anywhere
pip install transformers torch
python eval_llm_judge.py                        # LLM-as-a-judge (default: Qwen2.5-0.5B-Instruct)
```

## 📊 Result (from-scratch)
```
RETRIEVAL  (top-k = 3, macro-averaged over 5 queries)
   P@k    R@k  Hit@k     RR   nDCG  query
  0.33   1.00   1.00   1.00   1.00  << MEAN (the 'RR' column averaged is MRR)

GENERATION  (faithfulness = grounded in sources, relevance = on-topic)
 faith  relev  answer
  1.00   0.44  << MEAN over real answers
  0.60   0.39  [PLANTED HALLUCINATION] QLoRA stores its weights as 8-bit ...

AGENT ROUTING  (did the policy pick the right tool?)
  routing accuracy: 6/6 = 1.00
```

Retrieval nails **recall/MRR/nDCG = 1.00** (the right chunk is always retrieved and
ranked first), while **P@k = 0.33** is honest: with one relevant doc among three
retrieved, two-thirds of the context is padding. The planted hallucination drops
faithfulness **1.00 → 0.60** — the metric earns its keep.

## 🔍 Observations
- **Precision@k and recall@k tell different stories.** Perfect recall with low
  precision means "we found it, buried in filler" — that filler is context-window
  budget and a distraction for the generator. Tune `k` against both, not one.
- **Faithfulness ≠ relevance, and you need both.** A fluent answer to the *wrong*
  question scores high on faithfulness and low on relevance; a drifting answer does
  the opposite. Grading one axis hides the other failure.
- **The word-overlap judge is blind to paraphrase** — it punishes a correct reword.
  That's the exact ceiling `eval_llm_judge.py` lifts by asking a model that
  understands meaning, at the cost of speed, money, and the judge's own biases.
- **Your golden set is the real bottleneck.** The metrics are trivial arithmetic;
  the labeled `(query, relevant docs, correct tool)` rows are what's expensive — and
  a lazy label set makes every score downstream a comfortable lie.
- **An LLM judge needs its own eval.** Before trusting it, check its scores against
  human labels on a sample — an unvalidated judge just launders the generator's
  errors into confident numbers.

## ➡️ Next step
Phase 3 continues: swap the in-memory store for a real **vector database**
(FAISS / Chroma), add a **cross-encoder re-ranker**, and re-run this harness to prove
the upgrade actually moved recall and faithfulness — regression-testing the RAG
pipeline the same way you'd test any other system.
