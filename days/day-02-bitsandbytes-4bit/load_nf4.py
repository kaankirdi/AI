"""
Day 2 — 4-bit inference with bitsandbytes (NF4)
===============================================

Load a real LLM in 4-bit NormalFloat (NF4) and generate text. NF4 is the data
type introduced in the QLoRA paper: a 4-bit code whose levels are spaced to match
a normal distribution — a good fit for LLM weights.

Runs on:  a CUDA GPU. On CPU this script will still import but generation is
          impractically slow; read it as a reference and run it on Colab / a GPU box.

Run:      python load_nf4.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
from __future__ import annotations

import argparse


def build_nf4_config():
    """A standard NF4 4-bit config with double quantization + fp16 compute."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 (vs "fp4")
        bnb_4bit_use_double_quant=True,     # quantize the quant constants too
        bnb_4bit_compute_dtype=torch.float16,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", default="Explain quantization in one sentence.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("[warn] No CUDA GPU detected — bitsandbytes 4-bit needs a GPU.")
        print("       Run this on Colab (free T4) or any CUDA machine.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=build_nf4_config(),
        device_map="auto",
    )

    # Report the memory footprint of the quantized weights.
    footprint_gb = model.get_memory_footprint() / (1024 ** 3)
    print(f"[info] model footprint: {footprint_gb:.2f} GB (4-bit NF4)")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print("\n" + tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
