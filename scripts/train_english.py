#!/usr/bin/env python3
"""
Phase 5 (English): QLoRA Fine-Tuning — English Book Knowledge QA Model
=======================================================================
Base Model : Qwen/Qwen2.5-7B-Instruct (or Qwen/Qwen2.5-3B-Instruct)
Method     : QLoRA — 4-bit NF4 quantization + LoRA adapters (PEFT)
Trainer    : Hugging Face TRL SFTTrainer
Task       : English QA grounded in Shahbuddin Rathod's 20 books

Usage:
  # Default run
  python -X utf8 scripts/train_english.py

  # 3B Model (Low VRAM GPUs)
  python -X utf8 scripts/train_english.py --model Qwen/Qwen2.5-3B-Instruct --batch-size 1 --grad-accum 16

  # Custom directory run
  python -X utf8 scripts/train_english.py --data-dir ../tuning --output-dir ../models/book-qa-english-v1
"""

import os
import sys
import json
import argparse
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant with full access to the stories, philosophy, "
    "anecdotes, and wisdom from Shahbuddin Rathod's books. "
    "Your objective is to answer user questions accurately, thoroughly, and strictly in English, "
    "reflecting the author's principles, humor, and teachings."
)


def print_banner(text):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def load_english_sft_dataset(sft_path: Path):
    """Load english_sft_data.jsonl and format for Qwen2.5 chat template."""
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

            # Chat format
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>"
            )
            records.append({"text": text})
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 (English) — QLoRA Fine-Tuning (English Book Knowledge Model)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace base model ID (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--data-dir",
        default="../tuning",
        help="Folder containing english_sft_data.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="../models/book-qa-english-v1",
        help="Where to save the trained LoRA adapter",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size (use 1-2 for smaller VRAM)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum token length per training example",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent if script_dir.name == "scripts" else script_dir

    def resolve_path(given_path, default_folder_name):
        p = Path(given_path)
        if p.is_absolute():
            return p
        candidate1 = (base_dir / p).resolve()
        if candidate1.exists():
            return candidate1
        candidate2 = (base_dir / default_folder_name).resolve()
        if candidate2.exists():
            return candidate2
        return candidate1

    data_dir = resolve_path(args.data_dir, "tuning")
    output_dir = resolve_path(args.output_dir, "models/book-qa-english-v1")


    # ── 1. Dependency imports ──────────────────────────────────────────────
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
        from trl import SFTTrainer
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install training requirements first:")
        print("  pip install transformers peft trl bitsandbytes accelerate datasets")
        sys.exit(1)

    # ── 2. GPU detection ──────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No CUDA GPU detected. Training on CPU will be extremely slow.")
        vram_gb = 0
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"✅ GPU Detected : {gpu_name}")
        print(f"✅ VRAM         : {vram_gb:.1f} GB")

        if vram_gb < 8 and args.model == "Qwen/Qwen2.5-7B-Instruct":
            print("\n⚠️  Your GPU has less than 8 GB VRAM.")
            print("   Recommend switching to 3B model:")
            print("   python -X utf8 scripts/train_english.py --model Qwen/Qwen2.5-3B-Instruct --batch-size 1 --grad-accum 16\n")

    # ── 3. Load Data ──────────────────────────────────────────────────────
    print_banner("Loading English Training Data")
    sft_file = data_dir / "english_sft_data.jsonl"

    if not sft_file.exists():
        # Fallback check in tuning directory if relative path mismatch
        alt_sft_file = base_dir / "tuning" / "english_sft_data.jsonl"
        if alt_sft_file.exists():
            sft_file = alt_sft_file
        else:
            print(f"❌ English SFT data file not found at: {sft_file}")
            print("   Run scripts/generate_english_tuning_data.py first to create the dataset.")
            sys.exit(1)

    records = load_english_sft_dataset(sft_file)
    print(f"✅ Loaded {len(records)} English training examples")

    if len(records) < 50:
        print(f"\n⚠️  WARNING: Only {len(records)} training examples available.")
        print("   Recommend generating more dataset records for optimal accuracy.")

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"   Train split : {len(train_dataset)}")
    print(f"   Eval split  : {len(eval_dataset)}")

    # ── 4. Quantization Config (QLoRA) ────────────────────────────────────
    print_banner("Setting Up QLoRA (4-bit Quantization)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ── 5. Load Base Model ────────────────────────────────────────────────
    print(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 7. Training Arguments ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        fp16=False,
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        report_to="none",
    )

    # ── 8. Start Trainer ──────────────────────────────────────────────────
    print_banner("Starting Fine-Tuning")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    # ── 9. Save Final Adapter ─────────────────────────────────────────────
    print_banner("Saving Fine-Tuned Model")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"✅ Adapter saved successfully to: {output_dir}")
    print("\nNext step — test inference in English using chat_english.py:")
    print(f"  python -X utf8 scripts/chat_english.py --adapter-dir {output_dir}")


if __name__ == "__main__":
    main()
