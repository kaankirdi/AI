"""
Day 4 — KV-cache quantization (simulation)
==========================================

During autoregressive generation, every past token's Key and Value vectors are
cached so they aren't recomputed. For long contexts this **KV cache** can dwarf
the model weights in memory. Quantizing it (commonly to int8, or even int4/int2
in methods like KIVI) is one of the highest-leverage tricks for long-context LLMs.

This script *simulates* the effect with NumPy: build a fake KV cache, quantize it
per-token (asymmetric int8), run a scaled-dot-product attention with both the
fp32 and quantized caches, and measure the memory saving and the error in the
attention output. Runnable on any CPU.

Run:  python kv_cache_demo.py
"""
from __future__ import annotations

import numpy as np


def quantize_per_token_int8(x: np.ndarray):
    """Asymmetric int8 quantization with one (scale, zero_point) per token (row).

    x shape: (seq_len, head_dim). Returns (q_uint8, scale[seq_len], zp[seq_len]).
    Per-token granularity mirrors real KV-cache quantizers, which quantize each
    newly appended token independently.
    """
    qmin, qmax = 0, 255
    x_min = x.min(axis=-1, keepdims=True)
    x_max = x.max(axis=-1, keepdims=True)

    scale = np.where(x_max > x_min, (x_max - x_min) / (qmax - qmin), 1.0)
    zero_point = np.round(qmin - x_min / scale)
    zero_point = np.clip(zero_point, qmin, qmax)

    q = np.round(x / scale + zero_point)
    q = np.clip(q, qmin, qmax).astype(np.uint8)
    return q, scale, zero_point


def dequantize_per_token_int8(q, scale, zero_point):
    return (q.astype(np.float32) - zero_point) * scale


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(query, keys, values):
    """Single-query scaled dot-product attention. query:(d,) keys/values:(seq,d)."""
    d = query.shape[-1]
    scores = keys @ query / np.sqrt(d)          # (seq,)
    weights = softmax(scores)                    # (seq,)
    return weights @ values                      # (d,)


def _demo() -> None:
    rng = np.random.default_rng(0)
    seq_len, head_dim = 2048, 128

    # Fake cached keys/values for a long context, plus the current query.
    keys = rng.normal(size=(seq_len, head_dim)).astype(np.float32)
    values = rng.normal(size=(seq_len, head_dim)).astype(np.float32)
    query = rng.normal(size=(head_dim,)).astype(np.float32)

    # Reference attention output (fp32 cache).
    out_fp32 = attention(query, keys, values)

    # Quantize the cache to int8 and recompute.
    kq, ks, kz = quantize_per_token_int8(keys)
    vq, vs, vz = quantize_per_token_int8(values)
    keys_q = dequantize_per_token_int8(kq, ks, kz)
    values_q = dequantize_per_token_int8(vq, vs, vz)
    out_int8 = attention(query, keys_q, values_q)

    # Memory: fp16 is the usual baseline for a live cache; int8 halves it,
    # plus a small per-token scale/zero-point overhead.
    fp16_bytes = keys.nbytes + values.nbytes      # if stored as fp16
    fp16_bytes //= 2                               # float32 array -> fp16 size
    int8_bytes = kq.nbytes + vq.nbytes
    overhead = ks.nbytes + kz.nbytes + vs.nbytes + vz.nbytes

    cos = float(
        out_fp32 @ out_int8 / (np.linalg.norm(out_fp32) * np.linalg.norm(out_int8))
    )
    rel_err = float(np.linalg.norm(out_fp32 - out_int8) / np.linalg.norm(out_fp32))

    print(f"context length         : {seq_len} tokens, head_dim {head_dim}")
    print(f"KV cache (fp16)        : {fp16_bytes/1024:.1f} KB")
    print(f"KV cache (int8+scales) : {(int8_bytes + overhead)/1024:.1f} KB "
          f"({fp16_bytes / (int8_bytes + overhead):.2f}x smaller)")
    print(f"attention cosine sim   : {cos:.6f}")
    print(f"attention rel. L2 error: {rel_err:.4%}")
    print("\nTakeaway: int8 KV cache ~halves cache memory while the attention")
    print("output stays almost identical (cosine ~1.0). Methods like KIVI push")
    print("this to 2-bit with per-channel keys + per-token values.")


if __name__ == "__main__":
    _demo()
