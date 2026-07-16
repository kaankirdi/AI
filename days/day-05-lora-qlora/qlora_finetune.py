"""
Day 5 — QLoRA fine-tuning (real training script)
================================================

QLoRA = 4-bit NF4 frozen base weights (Day 02) + LoRA adapters (this day) + a few
memory tricks (paged optimizers, gradient checkpointing). The result: fine-tune a
7B model on a single ~16 GB GPU.

Runs on:  a CUDA GPU. Needs transformers, peft, bitsandbytes, accelerate, datasets, trl.
          Reference script — a real run takes minutes to hours depending on data.

Run:      python qlora_finetune.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
from __future__ import annotations

import argparse


def build_model_and_tokenizer(model_id: str):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # 1) Load the base model in 4-bit NF4 (frozen).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)  # enables grad checkpointing etc.

    # 2) Attach LoRA adapters to the attention projections.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # typically <1% of all params

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset", default="mlabonne/guanaco-llama2-1k")
    parser.add_argument("--output_dir", default="./qlora-out")
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = build_model_and_tokenizer(args.model)
    dataset = load_dataset(args.dataset, split="train")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",     # paged optimizer — the QLoRA memory trick
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    trainer.train()

    # Save only the small adapter — not the full base model.
    trainer.model.save_pretrained(args.output_dir)
    print(f"[done] LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
