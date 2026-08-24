#!/usr/bin/env python3
"""
Phase 5: QLoRA Fine-Tuning — Shahbuddin Rathod Author Voice
============================================================
Base Model : Qwen/Qwen2.5-7B-Instruct  (supports Gujarati natively)
Method     : QLoRA — 4-bit NF4 quantization + LoRA adapters (PEFT)
Trainer    : Hugging Face TRL SFTTrainer
GPU Target : RTX 3090 (24 GB) — batch_size=4, grad_accum=8  → ~2-3 hrs
             RTX 4050 (6 GB)  — batch_size=1, grad_accum=16 → ~6-8 hrs

Usage:
  # Default run (RTX 3090)
  python -X utf8 scripts/train.py

  # Smaller GPU (RTX 4050 6GB) — use 3B model
  python -X utf8 scripts/train.py --model Qwen/Qwen2.5-3B-Instruct --batch-size 1 --grad-accum 16

  # Custom run
  python -X utf8 scripts/train.py --data-dir ../data/tuning --output-dir ../models/rathod-v1 --epochs 3
"""

import os
import sys
import json
import argparse
from pathlib import Path


def print_banner(text):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def load_sft_dataset(sft_path: Path):
    """Load sft_data.jsonl and format for Qwen2.5 chat template."""
    records = []
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            instruction = rec.get("instruction", "").strip()
            output = rec.get("output", "").strip()
            if not instruction or not output:
                continue

            # Qwen2.5 chat template format
            text = (
                "<|im_start|>system\n"
                "તમે શાહ‍બુદ્દીન રાઠોડ છો — ગુજરાતના પ્રસિદ્ધ હાસ્યયોગ અને ફિલસૂફ. "
                "તમારા ઉત્તરો હંમેશા ટૂંકા, ટપ, ટૂ ધ પૉઇન્ટ, સૂક્ષ્મ હ્યુમર, "
                "પ્રત્યક્ષ ભાષા અને ગુજરાતી ઉક્તિઓ-દ્રષ્ટાંતો સાથે આપો.<|im_end|>\n"
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>"
            )
            records.append({"text": text})
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 — QLoRA Fine-Tuning (Rathod Author Voice)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace base model ID (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--data-dir",
        default="../data/tuning",
        help="Folder containing sft_data.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="../models/rathod-voice-v1",
        help="Where to save the LoRA adapter",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device training batch size (use 1-2 for 6GB GPU)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum token length per training example")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (higher = more capacity, more VRAM)")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

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
        print("   Strongly recommend running on an NVIDIA GPU.")
        vram_gb = 0
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"✅ GPU Detected : {gpu_name}")
        print(f"✅ VRAM         : {vram_gb:.1f} GB")

        # Auto-recommend settings for small GPUs
        if vram_gb < 8 and args.model == "Qwen/Qwen2.5-7B-Instruct":
            print("\n⚠️  Your GPU has less than 8 GB VRAM.")
            print("   Recommend switching to 3B model:")
            print("   python -X utf8 scripts/train.py --model Qwen/Qwen2.5-3B-Instruct --batch-size 1 --grad-accum 16")
            print("\nContinuing anyway — will attempt with current settings.\n")

    # ── 3. Load Data ──────────────────────────────────────────────────────
    print_banner("Loading Training Data")
    data_dir = Path(args.data_dir).resolve()
    sft_file = data_dir / "sft_data.jsonl"

    if not sft_file.exists():
        print(f"❌ SFT data not found: {sft_file}")
        print("   Run generate_tuning_data.py first to create the training dataset.")
        sys.exit(1)

    records = load_sft_dataset(sft_file)
    print(f"✅ Loaded {len(records)} formatted training examples")

    if len(records) < 100:
        print(f"\n⚠️  WARNING: Only {len(records)} examples — model may underfit.")
        print("   Strongly recommend completing full data generation (~2,634 pairs) first.")

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"   Train : {len(train_dataset)}")
    print(f"   Eval  : {len(eval_dataset)}")

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
    print("(First run downloads model files — Qwen2.5-7B ≈ 15 GB)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # safer than flash_attn for wider compatibility
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
    print_banner("Starting Fine-Tuning")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        learning_rate=args.lr,
        fp16=False,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",  # set to "wandb" if you want experiment tracking
        group_by_length=True,
        dataloader_num_workers=0,  # 0 = safer on Windows
    )

    # ── 8. SFT Trainer ───────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
    )

    print("🚀 Training started...")
    print(f"   Effective batch size : {args.batch_size * args.grad_accum}")
    print(f"   Epochs               : {args.epochs}")
    print(f"   Max sequence length  : {args.max_seq_len} tokens")
    print(f"   Output               : {output_dir}")
    trainer.train()

    # ── 9. Save final adapter ────────────────────────────────────────────
    print_banner("Saving LoRA Adapter")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"✅ LoRA adapter saved to: {output_dir}")
    print("   To use the model: run  python -X utf8 scripts/chat.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
