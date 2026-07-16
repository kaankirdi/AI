<div align="center">

# 🧠 AI Engineering Journey

**A daily, hands-on log of building deep expertise in modern AI systems.**

Real, runnable code — not tutorials copied from elsewhere. Each day tackles one
focused topic in LLM engineering, starting with model **quantization**, and builds
toward RAG, fine-tuning, and agentic systems.

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="python" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="pytorch" />
<img src="https://img.shields.io/badge/Focus-LLM%20Engineering-4F46E5?style=flat" alt="focus" />
<img src="https://img.shields.io/badge/License-MIT-0EA5E9?style=flat" alt="license" />

</div>

---

## Why this repo

I learn best by building. This repository is my public, incremental record of
going deep on the parts of modern AI that actually ship to production: making
large models **smaller, faster, and cheaper** without wrecking their quality, and
then wiring them into useful systems.

Every entry lives under [`days/`](./days) and follows the same shape:

> **Goal → Theory (short) → Code → Result / observation → Next step**

If a script needs a GPU it says so at the top; otherwise it runs on any laptop.

---

## 📈 Progress

| Day | Topic | Runs on | Status |
|-----|-------|---------|--------|
| [01](./days/day-01-quantization-from-scratch) | Integer quantization from scratch (int8, sym/asym) | CPU | ✅ Done |
| [02](./days/day-02-bitsandbytes-4bit) | 4-bit inference with bitsandbytes (NF4) | GPU | ✅ Done |
| [03](./days/day-03-gptq-vs-awq) | GPTQ vs AWQ — post-training quantization | GPU | ✅ Done |
| [04](./days/day-04-kv-cache-quantization) | KV-cache quantization for long-context inference | CPU (sim) | ✅ Done |
| 05 | LoRA / QLoRA fine-tuning | GPU | 🔜 Planned |
| 06 | Retrieval-Augmented Generation (RAG) basics | CPU | 🔜 Planned |
| 07 | Building a tool-using agent | CPU | 🔜 Planned |

---

## 🗺️ Roadmap

**Phase 1 — Efficient inference (current):** quantization from first principles →
bitsandbytes → GPTQ/AWQ → KV-cache & long context.

**Phase 2 — Adaptation:** LoRA, QLoRA, PEFT, dataset curation, evaluation.

**Phase 3 — Systems:** RAG pipelines, embeddings & vector stores, agents, tool use,
serving & benchmarking.

---

## 🚀 Setup

```bash
git clone https://github.com/kaankirdi/AI.git
cd AI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the first day (works on CPU, no model download needed):

```bash
python days/day-01-quantization-from-scratch/int8_quant.py
```

---

## 📚 References

Curated papers and links live in [`resources.md`](./resources.md).

---

<div align="center">
<sub>Maintained by <a href="https://github.com/kaankirdi">@kaankirdi</a> · Learning in public, one commit at a time.</sub>
</div>
