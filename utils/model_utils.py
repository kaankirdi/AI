"""Shared helpers used across the daily experiments.

Kept dependency-light: only NumPy is required for the CPU examples. The Torch
helpers are imported lazily so that Day 1 (pure NumPy) runs without Torch.
"""
from __future__ import annotations

import time
from contextlib import contextmanager


def bytes_human(n: int) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


@contextmanager
def timer(label: str = "block"):
    """Context manager that prints wall-clock time for a code block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[timer] {label}: {elapsed * 1000:.2f} ms")


def mse(a, b) -> float:
    """Mean squared error between two array-likes (NumPy or Torch)."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def gpu_mem_mb() -> float:
    """Currently allocated CUDA memory in MB (0.0 if no GPU / no Torch)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0
