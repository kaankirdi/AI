# Day 03 — GPTQ vs AWQ (Post-Training Quantization)

> **Runs on:** a CUDA GPU. Needs `auto-gptq` and/or `autoawq`.

## 🎯 Goal
Move past fixed codebooks (NF4) to **calibration-based** 4-bit methods, and
understand *why* two of the most-used ones — GPTQ and AWQ — take opposite routes
to the same goal.

## 🧠 Theory (short)
Both use a small **calibration dataset** to quantize better than a naive round.

**GPTQ** — *error compensation.* It quantizes weights one column at a time and
uses second-order (Hessian) information to adjust the *remaining, not-yet-quantized*
weights so they absorb the rounding error just introduced. Think of it as
"quantize, then correct everyone downstream." Very accurate, layer-local, no
retraining.

**AWQ** — *protect the salient weights.* It observes that only ~0.1–1% of weight
channels (identified by the **activation** magnitude flowing through them) really
matter. Before quantizing, it scales those channels up (and the matching
activations down), so the important weights land on finer quantization steps.
Simpler, very fast, and often better on instruction-tuned models.

| | GPTQ | AWQ |
|---|------|-----|
| Signal used | weight Hessian | activation magnitude |
| Core idea | compensate rounding error | scale up salient channels |
| Calibration cost | higher (per-column solve) | lower |
| Typical use | general PTQ to 3–4 bit | LLMs, esp. chat models |
| Inference kernels | ExLlama / Marlin | GEMM / GEMV |

Neither retrains the model — this is all **post-training**.

## 💻 Code
[`quantize_compare.py`](./quantize_compare.py) — quantize the same model with both
methods and save each. Uses a tiny inline calibration set for clarity; in practice
you'd feed a few hundred samples resembling your target domain.

```bash
# On a CUDA machine:
pip install auto-gptq autoawq transformers accelerate
python quantize_compare.py --model facebook/opt-125m --method both
```

## 📊 What to expect
On small models the accuracy gap between GPTQ and AWQ is minor; the real
differences show up as (a) **calibration time** — AWQ is usually faster — and
(b) which **inference kernel** you can use downstream. Both comfortably beat naive
round-to-nearest 4-bit.

## 🔍 Observations
- Calibration data **matters**: quantize with text that resembles your workload.
- `group_size=128` (a scale per 128 weights) is the common sweet spot between
  quality and overhead.
- AWQ tends to be the safer default for chat/instruct models; GPTQ is the veteran
  with the widest tooling support.

## ➡️ Next step
Weights are handled — but during long-context inference the **KV cache** becomes
the memory bottleneck. **Day 04** quantizes the cache itself.
