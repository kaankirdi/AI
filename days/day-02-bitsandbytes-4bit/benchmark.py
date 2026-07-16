"""
Day 2 — VRAM & latency: fp16 vs 4-bit NF4
=========================================

Loads the same model twice — once in fp16, once in 4-bit NF4 — and reports the
memory footprint and a rough tokens/sec for each. This makes the quantization
trade-off concrete instead of theoretical.

Runs on:  a CUDA GPU.

Run:      python benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
from __future__ import annotations

import argparse
import time


def load(model_id: str, four_bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {"device_map": "auto"}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def measure(model, tokenizer, prompt: str, max_new_tokens: int) -> float:
    """Return tokens/sec for a single greedy generation."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = out.shape[-1] - inputs["input_ids"].shape[-1]
    return generated / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = "Summarize why 4-bit quantization saves memory."

    print(f"{'variant':<12}{'footprint (GB)':>16}{'tokens/sec':>14}")
    print("-" * 42)
    for label, four_bit in (("fp16", False), ("nf4-4bit", True)):
        model = load(args.model, four_bit)
        gb = model.get_memory_footprint() / (1024 ** 3)
        tps = measure(model, tokenizer, prompt, args.max_new_tokens)
        print(f"{label:<12}{gb:>16.2f}{tps:>14.1f}")
        del model


if __name__ == "__main__":
    main()
