# Day 02 — 4-bit Inference with bitsandbytes (NF4)

> **Runs on:** a CUDA GPU (free Colab T4 is enough). CPU can read the code but
> generation is impractically slow.

## 🎯 Goal
Go from the hand-rolled int8 of Day 1 to **real 4-bit inference** on an actual
LLM, and measure the payoff in VRAM and speed.

## 🧠 Theory (short)
**NF4 (NormalFloat4)** is a 4-bit datatype from the QLoRA paper. Instead of
evenly spaced levels, its 16 codes are placed at the quantiles of a normal
distribution — because LLM weights are ~normally distributed, this puts more
resolution where the mass actually is. Two extra tricks:

- **Double quantization:** the per-block scale constants are themselves quantized,
  shaving another ~0.4 bits/param.
- **fp16 compute dtype:** weights are *stored* in 4-bit but *dequantized on the
  fly* to fp16 for the matmul, so quality stays high.

This is the same weight format QLoRA fine-tunes on top of (that's Day 5).

## 💻 Code
- [`load_nf4.py`](./load_nf4.py) — build a `BitsAndBytesConfig`, load a model in
  NF4, print its footprint, and generate.
- [`benchmark.py`](./benchmark.py) — load the model in **fp16** and **NF4** and
  compare footprint + tokens/sec side by side.

```bash
pip install torch transformers accelerate bitsandbytes
python benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## 📊 Expected result
For a ~1.1B model on a T4 you should see roughly:

| variant | footprint | tokens/sec |
|---------|-----------|-----------|
| fp16 | ~2.2 GB | baseline |
| nf4-4bit | ~0.7 GB | similar or slightly slower |

The headline: **~3× less memory** for a small quality hit. 4-bit is often
*not* faster per token on small models (dequant overhead), but it lets you fit
much larger models — or longer context — on the same card.

## 🔍 Observations
- `bnb_4bit_use_double_quant=True` is nearly free quality-wise; keep it on.
- Memory savings grow with model size; on a 7B model NF4 is the difference
  between "fits on a 12 GB card" and "does not".

## ➡️ Next step
NF4 is a *fixed* codebook. **Day 03** looks at data-driven post-training methods —
GPTQ and AWQ — that use calibration data to squeeze more quality out of 4 bits.
