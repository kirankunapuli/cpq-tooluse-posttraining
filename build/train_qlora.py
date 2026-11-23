"""QLoRA / LoRA SFT for the CRM tool-use dataset.

Designed for a single 16 GB GPU (Colab T4, Lightning A10, etc.). On Apple silicon the
4-bit path is unavailable, so `--no-4bit` runs bf16 LoRA on MPS with a smaller batch.

Why the defaults look the way they do (these are the defensible choices, not cargo cult):

  assistant_only_loss=True   the only tokens whose loss matters are the assistant's.
                             Training on prompt tokens is the #1 cause of
                             "loss falls, eval is flat".
  packing=False (default)    packing gives a large throughput win but silently leaks
                             attention across examples if the mask is wrong. Turn it on
                             only after confirming the loss-mask boundaries survive.
  max_length=1024            our p99 example is ~600 chars; 1024 tokens is generous and
                             anything longer truncates the answer, which deletes the label.
  epochs=2, lr=2e-4          LoRA tolerates ~10x the full-FT learning rate. With 674
                             examples, more than 3 epochs memorises.
  eval on a held-out split   entity-disjoint, so the number is not inflated by leakage.

Usage
-----
    pip install -r requirements.txt
    python build/train_qlora.py inspect            # tokenise + masking stats, no model
    python build/train_qlora.py train --out runs/lora-v1
    python build/train_qlora.py predict --out runs/lora-v1 --predictions preds.jsonl
    python eval/harness.py --predictions preds.jsonl --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def _load(split: str) -> list[dict]:
    p = DATA / f"split_{split}.jsonl"
    if not p.exists():
        sys.exit(f"{p} missing - run build/generate.py and build/filter_splits.py first")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ inspect

def cmd_inspect(args: argparse.Namespace) -> None:
    """Show token budget and how many tokens are actually trained on.

    This is the check that catches masking bugs before you spend GPU hours.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(args.base_model)
    rows = _load("train")
    total, trained, lens = 0, 0, []
    for r in rows[: args.limit]:
        msgs = r["messages"]
        full = tok.apply_chat_template(msgs, tokenize=True)
        # assistant-only view: render with the generation prompt removed
        prefix = tok.apply_chat_template(msgs[:2], tokenize=True, add_generation_prompt=True)
        total += len(full)
        trained += len(full) - len(prefix)
        lens.append(len(full))
    lens.sort()
    print(f"model            : {args.base_model}")
    print(f"examples         : {args.limit}")
    print(f"tokens total     : {total:,}")
    print(f"tokens in loss   : {trained:,}  ({100 * trained / max(1, total):.1f}%)")
    print(f"  -> {100 - 100 * trained / max(1, total):.1f}% of tokens are masked context")
    print(f"seq len p50/p95/max : {lens[len(lens)//2]} / {lens[int(len(lens)*0.95)-1]} / {lens[-1]}")
    print("If 'tokens in loss' is a small fraction or near 100%, work out why before training.")


# ------------------------------------------------------------------ train

def cmd_train(args: argparse.Namespace) -> None:
    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    train_rows = _load("train")
    valid_rows = _load("valid")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_cfg = None
    if args.use_4bit:
        if not torch.cuda.is_available():
            sys.exit("--use-4bit needs CUDA; pass --no-4bit on Apple silicon")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto" if quant_cfg else None,
    )

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_length=1024,
        packing=False,
        assistant_only_loss=True,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=13,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(valid_rows),
        peft_config=lora,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\nsaved adapter to {args.out}")
    print("next: predict, then score with eval/harness.py --selftest-compatible predictions")


# ------------------------------------------------------------------ predict

PROMPT_SUFFIX = ("\nRespond with a JSON array of tool calls and nothing else.")


def cmd_predict(args: argparse.Namespace) -> None:
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(args.out)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, args.out).eval()

    rows = _load(args.split)
    out = []
    for r in rows:
        msgs = r["messages"][:2]
        msgs = [dict(m) for m in msgs]
        msgs[-1]["content"] += PROMPT_SUFFIX
        # apply_chat_template returns a BatchEncoding, not a bare tensor; pass input_ids
        # through to generate explicitly or it fails on .shape.
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"].to(model.device)
        extra = {}
        if enc.get("attention_mask") is not None:
            extra["attention_mask"] = enc["attention_mask"].to(model.device)
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids, max_new_tokens=512,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id, **extra)
        text = tok.decode(gen[0][input_ids.shape[-1]:], skip_special_tokens=True)
        out.append({"task_id": r["task_id"], "raw": text})

    with Path(args.predictions).open("w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(out)} predictions to {args.predictions}")
    print(f"now: python eval/harness.py --predictions {args.predictions} --split {args.split}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["inspect", "train", "predict"])
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--out", default="runs/lora-v1")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--use-4bit", dest="use_4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    ap.add_argument("--split", default="test")
    ap.add_argument("--predictions", default="predictions_test.jsonl")
    args = ap.parse_args()

    {"inspect": cmd_inspect, "train": cmd_train, "predict": cmd_predict}[args.command](args)


if __name__ == "__main__":
    main()
