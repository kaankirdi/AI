# Day 04 — KV-Cache Quantization

> **Runs on:** any CPU (NumPy simulation — no model needed).

## 🎯 Goal
Understand why the **KV cache** becomes the memory bottleneck in long-context
inference, and show — with a runnable simulation — how quantizing it buys back
memory at almost no quality cost.

## 🧠 Theory (short)
In autoregressive decoding, each generated token attends to the Key/Value vectors
of **all previous tokens**. To avoid recomputing them, they're stored in the
**KV cache**. Its size grows linearly with context:

```
kv_cache_bytes ≈ 2 · layers · seq_len · n_kv_heads · head_dim · dtype_bytes
                 ^-- K and V
```

At long context this can exceed the model weights themselves. Since K/V vectors
are just activations, we can quantize them. The standard recipe:

- **Per-token** granularity: quantize each newly appended token's vector with its
  own scale/zero-point (cheap — you only have one new token per step).
- **int8** is nearly lossless; research methods (**KIVI**, **KVQuant**) go to
  2-bit by treating **keys per-channel** and **values per-token**, because the two
  have different outlier structure.

## 💻 Code
[`kv_cache_demo.py`](./kv_cache_demo.py) — builds a 2048-token fake cache,
quantizes it to per-token int8, runs scaled-dot-product attention with both the
fp32 and quantized caches, and reports memory + output error.

```bash
python kv_cache_demo.py
```

## 📊 Result
```
context length         : 2048 tokens, head_dim 128
KV cache (fp16)        : 1024.0 KB
KV cache (int8+scales) : 544.0 KB (1.88x smaller)
attention cosine sim   : 0.999974
attention rel. L2 error: 0.7249%
```

The attention output is essentially unchanged (cosine ≈ 1.0) while the cache is
~**1.9× smaller** vs fp16 — and the gap only grows against fp32. That saved memory
translates directly into **longer context** or **bigger batches** on the same GPU.

## 🔍 Observations
- The scale/zero-point overhead is real but small; coarser granularity (per-tensor)
  saves more memory but hurts quality — per-token is the usual balance.
- Keys are more outlier-prone along the channel dimension than values — which is
  exactly why KIVI quantizes keys per-channel and values per-token.

## ➡️ Next step
Phase 1 (efficient inference) is done: weights **and** cache are handled. **Day 05**
starts Phase 2 — adapting models cheaply with **LoRA / QLoRA** fine-tuning, which
builds directly on the NF4 weights from Day 02.
