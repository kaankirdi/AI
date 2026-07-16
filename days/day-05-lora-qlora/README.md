# Day 05 — LoRA / QLoRA Fine-Tuning

> **Runs on:** `lora_from_scratch.py` → any CPU. `qlora_finetune.py` → a CUDA GPU.

## 🎯 Goal
Start Phase 2 (adapting models). Understand **why** low-rank adaptation works with
a from-scratch NumPy demo, then see the real **QLoRA** training script that fine-tunes
a quantized model on a single GPU.

## 🧠 Theory (short)
Full fine-tuning updates every weight: for a `d×k` matrix that's `d·k` parameters,
times every layer — huge. **LoRA** freezes the pretrained `W` and learns a low-rank
update instead:

```
W' = W + (α/r) · B · A       B: (d×r),  A: (r×k),   r ≪ d,k
```

Only `B` and `A` train → `r·(d+k)` parameters, often **<1%** of the model. Because
`W` is untouched, one base model can host many hot-swappable adapters.

**QLoRA** stacks three ideas to make this fit on consumer hardware:
1. **4-bit NF4** frozen base weights (Day 02) — cuts base memory ~4×.
2. **LoRA** adapters in bf16 on top — the only trainable params.
3. **Paged optimizer + gradient checkpointing** — smooths memory spikes.

Together: fine-tune a **7B** model on a **single ~16 GB GPU**.

## 💻 Code
- [`lora_from_scratch.py`](./lora_from_scratch.py) — **CPU**. Shows, via truncated
  SVD, how a weight update is captured at increasing rank, counts LoRA vs full
  params, and verifies the efficient forward path `y = Wx + (α/r)·B·(A·x)`.
- [`qlora_finetune.py`](./qlora_finetune.py) — **GPU**. Real QLoRA pipeline with
  `transformers` + `peft` + `bitsandbytes` + `trl.SFTTrainer`.

```bash
python lora_from_scratch.py                                   # CPU demo
# On a GPU box:
pip install transformers peft bitsandbytes accelerate datasets trl
python qlora_finetune.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## 📊 Result (CPU demo)
```
Weight matrix: 768x768   full fine-tune params: 589,824

 rank r   LoRA params  % of full   rel. error
----------------------------------------------
      1         1,536      0.26%       0.8412
      2         3,072      0.52%       0.6759
      4         6,144      1.04%       0.0476
      8        12,288      2.08%       0.0471
     32        49,152      8.33%       0.0444

Forward paths agree (rank 8): max abs diff = 1.02e-05
```

The error **collapses at rank 4** — the update really does live in a low-dimensional
subspace, which is exactly why LoRA works. Past that, extra rank barely helps.

## 🔍 Observations
- Pick `r` at the "elbow" — here rank 4–8. Bigger `r` mostly wastes parameters.
- `alpha/r` scaling keeps the update magnitude stable as you change `r`.
- The adapter you save is only a few MB — cheap to store, share, and swap.

## ➡️ Next step
Phase 2 continues, then Phase 3 (systems): **Day 06** builds a Retrieval-Augmented
Generation (RAG) pipeline — embeddings, a vector store, and grounded answers.
