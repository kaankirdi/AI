"""
Day 5 — LoRA from scratch (the math, on CPU)
============================================

LoRA (Low-Rank Adaptation) freezes the pretrained weight W and learns a *low-rank*
update instead of a full one:

    full fine-tune:   W' = W + ΔW            ΔW is (d x k)  ->  d*k params
    LoRA:             W' = W + (α/r)·B·A     B:(d x r), A:(r x k) -> r*(d+k) params

Because r ≪ d, k, LoRA trains a tiny fraction of the parameters while leaving W
untouched (so one base model can host many swappable adapters).

This script demonstrates the core idea with NumPy — no training loop, no GPU:
  1. Any target update ΔW is best approximated at rank r by its truncated SVD;
     that IS the optimal (B, A). We use it to show how error falls as r grows.
  2. We count parameters and show the compression vs a full fine-tune.
  3. We verify the efficient forward path y = Wx + (α/r)·B·(A·x).

Run:  python lora_from_scratch.py
"""
from __future__ import annotations

import numpy as np


def lora_lowrank(delta: np.ndarray, r: int):
    """Optimal rank-r factorization of `delta` via truncated SVD.

    Returns (B, A) with delta ≈ B @ A, B:(d,r), A:(r,k).
    (In real LoRA, B and A are *learned* by gradient descent — the SVD here just
    gives us the best-possible rank-r target to benchmark against.)
    """
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    B = U[:, :r] * S[:r]      # fold singular values into B  -> (d, r)
    A = Vt[:r, :]             # (r, k)
    return B, A


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def _demo() -> None:
    rng = np.random.default_rng(0)
    d, k = 768, 768                 # a Transformer-ish projection matrix
    alpha = 16

    W = rng.normal(size=(d, k)).astype(np.float32)

    # A target task update: mostly low-rank structure + a little full-rank noise,
    # which is roughly what fine-tuning updates look like in practice.
    core_B = rng.normal(size=(d, 4))
    core_A = rng.normal(size=(4, k))
    target_delta = (core_B @ core_A) * 0.02 + rng.normal(size=(d, k)) * 0.002

    full_params = d * k
    print(f"Weight matrix: {d}x{k}   full fine-tune params: {full_params:,}\n")
    print(f"{'rank r':>7}{'LoRA params':>14}{'% of full':>11}{'rel. error':>13}")
    print("-" * 46)
    for r in (1, 2, 4, 8, 16, 32):
        B, A = lora_lowrank(target_delta, r)
        lora_params = r * (d + k)
        approx = B @ A
        print(f"{r:>7}{lora_params:>14,}{100 * lora_params / full_params:>10.2f}%"
              f"{rel_error(target_delta, approx):>13.4f}")

    # --- verify the efficient forward path ---
    r = 8
    B, A = lora_lowrank(target_delta, r)
    scaling = alpha / r
    x = rng.normal(size=(k,)).astype(np.float32)

    y_merged = (W + scaling * (B @ A)) @ x          # merge then multiply
    y_efficient = W @ x + scaling * (B @ (A @ x))   # keep low-rank at inference
    print(f"\nForward paths agree (rank {r}): "
          f"max abs diff = {np.max(np.abs(y_merged - y_efficient)):.2e}")

    print("\nTakeaway: at rank 8 LoRA trains ~2% of the parameters and already")
    print("captures most of the update. QLoRA adds one more trick — the frozen W")
    print("is stored in 4-bit NF4 (see Day 02) — so a 7B model fine-tunes on a")
    print("single consumer GPU. See qlora_finetune.py for the real training script.")


if __name__ == "__main__":
    _demo()
