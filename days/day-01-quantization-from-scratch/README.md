# Day 01 — Integer Quantization From Scratch

> **Runs on:** any CPU · no GPU, no PyTorch, no model download.

## 🎯 Goal
Build a rock-solid intuition for what quantization *actually does* by
implementing int8 quantization by hand — both **symmetric** and **asymmetric** —
with nothing but NumPy.

## 🧠 Theory (short)
Quantization maps a continuous float range onto a small set of integers. The map
is defined by a **scale** `s` and a **zero-point** `z`:

```
q = round(x / s) + z         (quantize)
x ≈ (q - z) * s              (dequantize)
```

- **Symmetric (int8, range [-127, 127]):** `z = 0`. The scale is set from the
  largest absolute value: `s = max(|x|) / 127`. Simple and fast (no zero-point
  in the matmul), but it *wastes* codes when the data is not centered at zero.
- **Asymmetric (uint8, range [0, 255]):** uses the true min/max, so `z ≠ 0`.
  It fits skewed distributions (e.g. post-ReLU activations) more tightly, at the
  cost of carrying a zero-point.

The tension is always the same: **outliers stretch the scale**, which coarsens
the step size for everyone else. That single fact motivates almost every
advanced method (per-channel scales, LLM.int8() outlier handling, SmoothQuant).

## 💻 Code
[`int8_quant.py`](./int8_quant.py) — the two schemes plus MSE / max-error metrics
and a small demo on a weight-like tensor with deliberate outliers.

```bash
python int8_quant.py
```

## 📊 Result
On a tensor with a `+3.7` and a `-2.1` outlier:

| scheme | scale | zero-point | MSE | max error |
|--------|-------|-----------|-----|-----------|
| symmetric | 0.0291 | 0 | 8.10e-05 | 0.0143 |
| asymmetric | 0.0228 | 92 | 5.66e-05 | 0.0113 |

Asymmetric wins here because the data is **not** centered at zero — it spends its
256 codes on the actual `[min, max]` range instead of a symmetric window, giving
a ~30% smaller scale and lower error. Memory drops **4×** (float32 → int8).

## 🔍 Observations
- The error is dominated by how big the **scale** is, and the scale is dominated
  by the **outliers**. Clip one outlier and both schemes improve immediately.
- Symmetric is preferred for **weights** (roughly zero-centered, and it keeps the
  matmul cheap); asymmetric shines for **activations** (often one-sided).

## ➡️ Next step
Real 4-bit inference: **Day 02** loads an actual LLM in NF4 with `bitsandbytes`
and measures the VRAM/latency payoff.
