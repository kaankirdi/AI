"""
Day 1 — Integer Quantization From Scratch
==========================================

Goal:  understand *exactly* what happens when we quantize a float32 tensor to
       int8, by implementing both schemes by hand with only NumPy.

  - Symmetric  int8: values map to [-127, 127], zero-point fixed at 0.
  - Asymmetric uint8: values map to [0, 255] with a learned zero-point.

Runs on:  any CPU. No GPU, no PyTorch, no model download.

Run:      python int8_quant.py
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Symmetric quantization  (signed int8, zero-point = 0)
# --------------------------------------------------------------------------- #
def quantize_symmetric(x: np.ndarray, num_bits: int = 8):
    """Quantize `x` symmetrically. Returns (q_int8, scale)."""
    qmax = 2 ** (num_bits - 1) - 1  # 127 for int8
    max_abs = float(np.max(np.abs(x)))
    scale = max_abs / qmax if max_abs > 0 else 1.0

    q = np.round(x / scale)
    q = np.clip(q, -qmax, qmax).astype(np.int8)
    return q, scale


def dequantize_symmetric(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


# --------------------------------------------------------------------------- #
# Asymmetric quantization  (unsigned uint8, learned zero-point)
# --------------------------------------------------------------------------- #
def quantize_asymmetric(x: np.ndarray, num_bits: int = 8):
    """Quantize `x` asymmetrically. Returns (q_uint8, scale, zero_point)."""
    qmin, qmax = 0, 2 ** num_bits - 1  # 0..255 for uint8
    x_min, x_max = float(np.min(x)), float(np.max(x))

    # Guard against a constant tensor (x_max == x_min).
    scale = (x_max - x_min) / (qmax - qmin) if x_max > x_min else 1.0

    # zero_point is the integer that x=0.0 maps to.
    zero_point = round(qmin - x_min / scale)
    zero_point = int(np.clip(zero_point, qmin, qmax))

    q = np.round(x / scale + zero_point)
    q = np.clip(q, qmin, qmax).astype(np.uint8)
    return q, scale, zero_point


def dequantize_asymmetric(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _demo() -> None:
    rng = np.random.default_rng(seed=42)

    # A weight-like tensor: mostly small values with a couple of outliers,
    # which is exactly the regime where symmetric vs asymmetric matters.
    x = rng.normal(loc=0.1, scale=0.5, size=(4, 8)).astype(np.float32)
    x[0, 0] = 3.7   # positive outlier
    x[3, 7] = -2.1  # negative outlier

    print("Original float32 tensor:")
    print(np.round(x, 3), "\n")

    # --- symmetric ---
    q_s, scale_s = quantize_symmetric(x)
    x_s = dequantize_symmetric(q_s, scale_s)

    # --- asymmetric ---
    q_a, scale_a, zp_a = quantize_asymmetric(x)
    x_a = dequantize_asymmetric(q_a, scale_a, zp_a)

    fp32_bytes = x.nbytes
    int8_bytes = q_s.nbytes  # same element count, 1 byte each

    print("=" * 62)
    print(f"{'scheme':<14}{'scale':>12}{'zero_pt':>10}{'MSE':>13}{'max_err':>12}")
    print("-" * 62)
    print(f"{'symmetric':<14}{scale_s:>12.5f}{0:>10}{mse(x, x_s):>13.2e}{max_abs_error(x, x_s):>12.4f}")
    print(f"{'asymmetric':<14}{scale_a:>12.5f}{zp_a:>10}{mse(x, x_a):>13.2e}{max_abs_error(x, x_a):>12.4f}")
    print("=" * 62)

    print(f"\nMemory:  float32 = {fp32_bytes} bytes  ->  int8 = {int8_bytes} bytes"
          f"  ({fp32_bytes / int8_bytes:.1f}x smaller)")
    print("Note: real kernels also store the scale (and zero-point), a tiny")
    print("per-tensor/per-channel overhead — negligible for large weight matrices.")


if __name__ == "__main__":
    _demo()
