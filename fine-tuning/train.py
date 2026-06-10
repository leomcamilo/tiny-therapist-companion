#!/usr/bin/env python3
"""
Fine-tuning script for Tiny Therapist Companion using Unsloth + QLoRA.

Trains Nemotron-Nano-4B-v1.1 on a synthetic conversational dataset
for empathetic listening, gratitude practice, and guided journaling.

Usage:
    python fine-tuning/train.py
    python fine-tuning/train.py --push_to_hub
    python fine-tuning/train.py --use_wandb
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(file_path: str) -> list[dict]:
    """Load a JSONL dataset file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_into_chat_template(examples: list[dict], tokenizer):
    """Format conversational examples into the model's chat template."""
    formatted_texts = []
    for example in examples:
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)
    return formatted_texts


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Tiny Therapist Companion")
    parser.add_argument(
        "--config",
        type=str,
        default="./fine-tuning/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the final model to HuggingFace Hub",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID for pushing (default: auto-generated)",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    config = load_config(args.config)

    # ── W&B setup ────────────────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(
            project="tiny-therapist-companion",
            name=f"nemotron-nano-4b-lora-r{config['lora']['rank']}",
            config=config,
        )
        os.environ["WANDB_PROJECT"] = "tiny-therapist-companion"

    # ── Unsloth setup ────────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: Unsloth is not installed. Install with:")
        print("  pip install unsloth")
        print("Or follow the guide at https://github.com/unslothai/unsloth")
        sys.exit(1)

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from peft import LoraConfig

    # ── Load model with 4-bit quantization ───────────────────────────────
    model_cfg = config["model"]
    quant_cfg = config["quantization"]

    print(f"Loading base model: {model_cfg['base_model']}")
    print(f"Max sequence length: {model_cfg['max_seq_length']}")
    print(f"4-bit quantization: {quant_cfg['load_in_4bit']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=quant_cfg["load_in_4bit"],
        dtype=None,  # Auto-detect
    )

    # ── Apply LoRA adapter ───────────────────────────────────────────────
    lora_cfg = config["lora"]
    print(f"Applying LoRA: rank={lora_cfg['rank']}, alpha={lora_cfg['alpha']}")
    print(f"Target modules: {lora_cfg['target_modules']}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        use_rslora=False,  # Standard LoRA
        loftq_config=None,
    )

    # ── Load datasets ────────────────────────────────────────────────────
    dataset_cfg = config["dataset"]
    train_data = load_dataset(dataset_cfg["train_file"])
    val_data = load_dataset(dataset_cfg["val_file"])

    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    # ── Set up chat template formatting ──────────────────────────────────
    # We use the tokenizer's built-in chat template for Nemotron
    # The formatting_fn handles conversion from messages to text

    def formatting_prompts_func(examples):
        """Convert batch of messages into formatted text using chat template."""
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    # ── Training arguments ────────────────────────────────────────────────
    train_cfg = config["training"]

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["eval_strategy"],  # Fixed from deprecated evaluation_strategy
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        seed=train_cfg["seed"],
        report_to=["wandb"] if args.use_wandb else ["none"],
        load_best_model_at_end=True if train_cfg["eval_strategy"] != "no" else False,
        metric_for_loss="eval_loss" if train_cfg["eval_strategy"] != "no" else None,
    )

    # ── Create datasets in HuggingFace format ─────────────────────────────
    from datasets import Dataset

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # ── Initialize trainer ────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_prompts_func,
        max_seq_length=model_cfg["max_seq_length"],
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Starting training...")
    print(f"  Model: {model_cfg['base_model']}")
    print(f"  LoRA rank: {lora_cfg['rank']}, alpha: {lora_cfg['alpha']}")
    print(f"  Epochs: {train_cfg['num_train_epochs']}")
    print(f"  Batch size: {train_cfg['per_device_train_batch_size']} x {train_cfg['gradient_accumulation_steps']} accum")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print("=" * 60)

    trainer.train()

    # ── Save model ────────────────────────────────────────────────────────
    output_dir = train_cfg["output_dir"]
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also save merged model for deployment
    merged_dir = os.path.join(output_dir, "merged")
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer)

    # ── Push to Hub ──────────────────────────────────────────────────────
    if args.push_to_hub:
        hub_id = args.hub_model_id or "leomcamilo/tiny-therapist-companion-nemotron-nano-4b"
        print(f"\nPushing LoRA adapter to HuggingFace Hub as: {hub_id}")
        model.push_to_hub(hub_id, tokenizer=tokenizer)

        print(f"Pushing merged model to HuggingFace Hub as: {hub_id}-merged")
        model.push_to_hub_merged(
            f"{hub_id}-merged",
            tokenizer,
        )
        print("✅ Model pushed to Hub successfully!")

    # ── W&B finish ────────────────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.finish()

    print("\n✅ Training complete!")
    print(f"   LoRA adapter saved to: {output_dir}")
    print(f"   Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()