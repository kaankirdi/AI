# Day 06 — Retrieval-Augmented Generation (RAG) Basics

> **Runs on:** `rag_from_scratch.py` → any CPU, no download. `rag_semantic.py` →
> any CPU, one small (~90 MB) model download.

## 🎯 Goal
Start Phase 3 (systems). Build the **retrieval** half of RAG by hand so the moving
parts are visible, then swap the hand-rolled embedder for a real semantic model
and feel exactly what changes.

## 🧠 Theory (short)
An LLM only knows what was in its training data, frozen at some cutoff, and it will
happily hallucinate the rest. **RAG** fixes this by fetching relevant text at query
time and putting it *in the prompt*, so the answer is grounded in real sources:

```
chunk  ->  embed  ->  index  ->  retrieve (cosine top-k)  ->  augment prompt  ->  generate
```

- **Chunk** — split documents into small, self-contained passages.
- **Embed** — map each chunk to a vector. Similar meaning → nearby vectors.
- **Index** — store the vectors (here: a NumPy matrix; in prod: FAISS / a vector DB).
- **Retrieve** — embed the question, take the top-k by cosine similarity.
- **Augment** — paste those chunks into the prompt with a "answer only from this" instruction.
- **Generate** — an LLM writes the final, cite-able answer.

The quality of retrieval sets the ceiling for the whole system: **garbage in →
grounded-sounding garbage out.** So the embedder matters, which is the whole point
of comparing the two scripts.

## 💻 Code
- [`rag_from_scratch.py`](./rag_from_scratch.py) — **CPU, no download.** A
  deterministic **TF-IDF** embedder, an in-memory `VectorStore` doing brute-force
  cosine search, chunking with overlap, and grounded-prompt assembly. The whole
  pipeline in NumPy.
- [`rag_semantic.py`](./rag_semantic.py) — **CPU.** Same pipeline, but the embedder
  is `sentence-transformers` `all-MiniLM-L6-v2`. Only the embedding step changes.

```bash
python rag_from_scratch.py                    # pure NumPy, runs anywhere
pip install sentence-transformers
python rag_semantic.py                        # real embeddings, ~90 MB model
```

## 📊 Result (from-scratch, TF-IDF)
```
Q: How does QLoRA store weights in 4 bits?
rank   score  source
   1   0.353  nf4          <- correct
   2   0.131  gptq
   3   0.073  kv_cache

Q: How do you compress a model without retraining it?
rank   score  source
   1   0.181  lora         <- WRONG (should be gptq / awq)
   2   0.139  gptq
   3   0.124  nf4
```

The first query shares words with its source (`4-bit`, `weights`, `QLoRA`) so
TF-IDF nails it. The last one is phrased with **synonyms** — "compress" and
"without retraining" never literally appear in the GPTQ/AWQ docs — so lexical
matching drifts to the wrong document. That failure is the motivation for
`rag_semantic.py`: real embeddings match *meaning*, and MiniLM ranks GPTQ/AWQ
first on exactly these reworded queries.

## 🔍 Observations
- **Cosine on unit vectors is just a dot product** — retrieval is one matmul.
  FAISS / pgvector only add an *approximate* index for scale; the idea is the same.
- **TF-IDF is a strong, free baseline** when queries reuse the corpus vocabulary,
  and it needs zero model. Reach for it before assuming you need embeddings.
- **Embeddings buy you paraphrase robustness**, which is what real users type.
- The **grounding instruction** ("answer only from the context, else say you don't
  know") is doing the anti-hallucination work — retrieval just supplies the facts.
- Chunk size is a real knob: too big wastes context and dilutes relevance, too
  small severs the sentence that held the answer. Overlap hedges the boundaries.

## ➡️ Next step
Retrieval is solved but crude. **Day 07** closes the loop into a real answer:
plug an LLM onto these retrieved chunks (a local quantized model from Days 02–05,
or an API), add a re-ranking step, and measure grounding — the first end-to-end
RAG **agent**.
