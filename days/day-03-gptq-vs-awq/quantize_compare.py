"""
Day 3 — GPTQ vs AWQ: post-training quantization
===============================================

Two calibration-based 4-bit methods, side by side. Unlike NF4 (a fixed codebook),
both use a small calibration dataset to decide *how* to quantize each layer.

  - GPTQ: quantizes weights column-by-column, using second-order (Hessian) info
          to compensate the not-yet-quantized weights for each rounding error.
  - AWQ:  observes that a few *salient* weight channels (picked by activation
          magnitude) matter most, and scales them up before quantizing so they
          survive with more precision.

Runs on:  a CUDA GPU. Requires `auto-gptq` and/or `autoawq` (see requirements.txt).
          This file is a runnable reference — quantizing takes a few minutes.

Run:      python quantize_compare.py --model facebook/opt-125m
"""
from __future__ import annotations

import argparse


def quantize_gptq(model_id: str, out_dir: str) -> None:
    """Quantize to 4-bit with GPTQ and save."""
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)

    # A tiny calibration set — in practice use a few hundred domain-like samples.
    calibration = [
        tokenizer("Quantization reduces the memory footprint of neural networks."),
        tokenizer("Large language models can be compressed with little quality loss."),
    ]

    model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
    model.quantize(calibration)
    model.save_quantized(out_dir)
    print(f"[gptq] saved 4-bit model to {out_dir}")


def quantize_awq(model_id: str, out_dir: str) -> None:
    """Quantize to 4-bit with AWQ and save."""
    from transformers import AutoTokenizer
    from awq import AutoAWQForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoAWQForCausalLM.from_pretrained(model_id)
    quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}

    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[awq] saved 4-bit model to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--method", choices=["gptq", "awq", "both"], default="both")
    args = parser.parse_args()

    if args.method in ("gptq", "both"):
        quantize_gptq(args.model, out_dir="./opt-125m-gptq-4bit")
    if args.method in ("awq", "both"):
        quantize_awq(args.model, out_dir="./opt-125m-awq-4bit")


if __name__ == "__main__":
    main()
