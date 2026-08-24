#!/usr/bin/env python3
"""
OKS Model Fine-Tuning — QLoRA Training for Book Knowledge QA
==============================================================
Fine-tunes a base model using QLoRA on the OKS-generated training data
to create a specialized Book Knowledge QA model.

After training, rebuilds model.pkl with the fine-tuned adapter path.

Base Model:  Qwen/Qwen2.5-7B-Instruct (or 3B for lower VRAM)
Method:      QLoRA — 4-bit NF4 quantization + LoRA adapters
Trainer:     Hugging Face TRL SFTTrainer
Task:        Factual English QA grounded in 20 books via OKS

Usage:
  python -X utf8 scripts/train_oks_model.py
  python -X utf8 scripts/train_oks_model.py --model Qwen/Qwen2.5-3B-Instruct --batch-size 1
"""

import os
import sys
import json
import argparse
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert AI assistant with comprehensive knowledge of all 20 books "
    "written by Shahbuddin Rathod, the renowned Gujarati humorist and philosopher. "
    "Answer questions accurately, thoroughly, and strictly in English, citing which "
    "book(s) the information comes from. Use the structured knowledge base (OKS) "
    "to provide factual, detailed responses about characters, themes, stories, "
    "opinions, and statistics from the books."
)


def print_banner(text):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def load_oks_sft_dataset(sft_path):
    """Load oks_sft_data.jsonl and format for Qwen2.5 chat template."""
    records = []
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            
            instruction = rec.get("instruction", "").strip()
            output = rec.get("output", "").strip()
            if not instruction or not output:
                continue

            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>"
            )
            records.append({"text": text})
    return records


def main():
    parser = argparse.ArgumentParser(
        description="OKS Model Fine-Tuning — QLoRA for Book Knowledge QA"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                       help="HuggingFace base model ID")
    parser.add_argument("--data-dir", default="tuning",
                       help="Folder containing oks_sft_data.jsonl")
    parser.add_argument("--output-dir", default="models/book-knowledge-oks-v1",
                       help="Where to save the trained LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent if script_dir.name == "scripts" else script_dir

    def resolve_path(given, default):
        p = Path(given)
        if p.is_absolute():
            return p
        c1 = (base_dir / p).resolve()
        if c1.exists():
            return c1
        c2 = (base_dir / default).resolve()
        if c2.exists():
            return c2
        return c1

    data_dir = resolve_path(args.data_dir, "tuning")
    output_dir = resolve_path(args.output_dir, "models/book-knowledge-oks-v1")

    # ── 1. Import libraries ─────────────────────────────────────────────
    print_banner("Importing Libraries")
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall training requirements:")
        print("  pip install transformers peft trl bitsandbytes accelerate datasets")
        sys.exit(1)

    # ── 2. GPU detection ────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No CUDA GPU detected. Training will be extremely slow.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"✅ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    # ── 3. Load OKS training data ──────────────────────────────────────
    print_banner("Loading OKS Training Data")
    sft_file = data_dir / "oks_sft_data.jsonl"

    if not sft_file.exists():
        alt = base_dir / "tuning" / "oks_sft_data.jsonl"
        if alt.exists():
            sft_file = alt
        else:
            print(f"❌ OKS SFT data not found at: {sft_file}")
            print("   Run: python -X utf8 scripts/generate_oks_tuning_data.py")
            sys.exit(1)

    records = load_oks_sft_dataset(sft_file)
    print(f"✅ Loaded {len(records)} OKS training examples")

    if len(records) < 20:
        print(f"⚠️  Only {len(records)} examples. Generate more for better accuracy.")

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"   Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # ── 4. Quantization ────────────────────────────────────────────────
    print_banner("Setting Up QLoRA")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ── 5. Load base model ──────────────────────────────────────────────
    print(f"Loading: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 6. LoRA Config ────────────────────────────────────────────────────
    print_banner("Configuring LoRA Adapter")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ── 7. Training arguments ───────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=args.lr,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        dataset_text_field="text",
        report_to="none",
    )

    # ── 8. Start Trainer ──────────────────────────────────────────────────
    print_banner("Starting Fine-Tuning")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )
    trainer.train()

    # ── 9. Save ─────────────────────────────────────────────────────────
    print_banner("Saving Fine-Tuned OKS Model")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"✅ OKS adapter saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Rebuild model.pkl: python -X utf8 scripts/build_model.py")
    print(f"  2. Chat: python -X utf8 scripts/chat_oks.py")


if __name__ == "__main__":
    main()
