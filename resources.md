# 📚 Resources

Curated references for the topics in this repo. Papers first, then practical guides.

## Quantization — foundations
- **LLM.int8()** — Dettmers et al., 2022 — 8-bit matmul with outlier handling.
  https://arxiv.org/abs/2208.07339
- **GPTQ** — Frantar et al., 2022 — accurate post-training quantization to 3–4 bits.
  https://arxiv.org/abs/2210.17323
- **AWQ** — Lin et al., 2023 — activation-aware weight quantization.
  https://arxiv.org/abs/2306.00978
- **QLoRA** — Dettmers et al., 2023 — 4-bit NF4 + LoRA fine-tuning.
  https://arxiv.org/abs/2305.14314
- **SmoothQuant** — Xiao et al., 2022 — migrating activation outliers into weights.
  https://arxiv.org/abs/2211.10438

## KV-cache & long context
- **KIVI** — 2-bit asymmetric KV cache quantization. https://arxiv.org/abs/2402.02750
- **H2O** — heavy-hitter oracle for KV cache eviction. https://arxiv.org/abs/2306.14048

## Retrieval-Augmented Generation
- **RAG** — Lewis et al., 2020 — the original retrieval-augmented generation paper.
  https://arxiv.org/abs/2005.11401
- **Dense Passage Retrieval (DPR)** — Karpukhin et al., 2020 — dual-encoder retrieval.
  https://arxiv.org/abs/2004.04906
- **Sentence-BERT** — Reimers & Gurevych, 2019 — sentence embeddings via siamese BERT.
  https://arxiv.org/abs/1908.10084
- Sentence-Transformers docs: https://www.sbert.net/
- FAISS — efficient similarity search: https://github.com/facebookresearch/faiss

## Practical guides / libraries
- Hugging Face — Quantization docs: https://huggingface.co/docs/transformers/quantization
- bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes
- AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
- AutoAWQ: https://github.com/casper-hansen/AutoAWQ
- llama.cpp (GGUF quant formats): https://github.com/ggerganov/llama.cpp

## Background math
- A gentle intro to quantization (scale & zero-point):
  https://huggingface.co/blog/hf-bitsandbytes-integration
