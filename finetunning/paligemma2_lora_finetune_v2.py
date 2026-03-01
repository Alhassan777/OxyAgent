import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import torch
from PIL import Image
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    PaliGemmaForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)


def has_label(suffix: str, label: str) -> bool:
    return f" {label}" in suffix


def dominant_label(suffix: str) -> str:
    swim = suffix.count(" swimming")
    drown = suffix.count(" drowning")
    if swim == 0 and drown == 0:
        return "unknown"
    return "swimming" if swim >= drown else "drowning"


def load_records(json_path: str, data_root: str) -> List[Dict[str, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    kept = []
    for row in rows:
        image_abs = os.path.join(data_root, row["image"])
        if not os.path.exists(image_abs):
            continue
        suffix = row["suffix"]
        kept.append(
            {
                "image_path": image_abs,
                "prefix": row.get("prefix", "detect swimming ; drowning"),
                "suffix": suffix,
                "dominant_label": dominant_label(suffix),
                "has_swimming": has_label(suffix, "swimming"),
                "has_drowning": has_label(suffix, "drowning"),
            }
        )
    return kept


def rebalance_rows(rows: List[Dict[str, str]], max_factor: float = 2.0, seed: int = 42) -> List[Dict[str, str]]:
    by_label = {"swimming": [], "drowning": [], "unknown": []}
    for row in rows:
        by_label[row["dominant_label"]].append(row)

    n_swim = len(by_label["swimming"])
    n_drown = len(by_label["drowning"])
    if n_swim == 0 or n_drown == 0:
        return rows

    minority_label = "swimming" if n_swim < n_drown else "drowning"
    majority_label = "drowning" if minority_label == "swimming" else "swimming"

    n_minority = len(by_label[minority_label])
    n_majority = len(by_label[majority_label])
    target = min(n_majority, int(math.ceil(n_minority * max_factor)))

    random.seed(seed)
    balanced = list(rows)
    while len([r for r in balanced if r["dominant_label"] == minority_label]) < target:
        balanced.append(random.choice(by_label[minority_label]).copy())

    random.shuffle(balanced)
    return balanced


@dataclass
class PaliCollator:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        images = [Image.open(item["image_path"]).convert("RGB") for item in features]
        prefixes = [f"<image> {item['prefix']}" for item in features]
        suffixes = [item["suffix"] for item in features]

        batch = self.processor(
            text=prefixes,
            images=images,
            suffix=suffixes,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        return batch


def print_distribution(name: str, rows: List[Dict[str, str]]) -> None:
    c = Counter(r["dominant_label"] for r in rows)
    swim_boxes = sum(r["suffix"].count(" swimming") for r in rows)
    drown_boxes = sum(r["suffix"].count(" drowning") for r in rows)
    print(f"{name} images by dominant label: {dict(c)}")
    print(f"{name} box counts: swimming={swim_boxes}, drowning={drown_boxes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/paligemma2-3b-pt-224")
    parser.add_argument("--data_root", default="/home/hackathon/colab_upload (1)")
    parser.add_argument("--train_json", default="/home/hackathon/colab_upload (1)/paligemma_train.json")
    parser.add_argument("--val_json", default="/home/hackathon/colab_upload (1)/paligemma_val.json")
    parser.add_argument("--output_dir", default="/home/hackathon/paligemma2-lora-out-v2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--balance_factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    train_rows = load_records(args.train_json, args.data_root)
    val_rows = load_records(args.val_json, args.data_root)
    if not train_rows:
        raise RuntimeError("No valid training rows found.")

    print_distribution("Raw train", train_rows)
    train_rows = rebalance_rows(train_rows, max_factor=args.balance_factor, seed=args.seed)
    print_distribution("Balanced train", train_rows)
    print_distribution("Validation", val_rows)

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows) if val_rows else None

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True if val_ds is not None else False,
        metric_for_best_model="eval_loss" if val_ds is not None else None,
        greater_is_better=False if val_ds is not None else None,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if val_ds is not None else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PaliCollator(processor=processor),
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Done. LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
