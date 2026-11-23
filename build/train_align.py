"""Continue training from the SFT adapter with DPO or GRPO, then score it.

Both methods start from the same SFT checkpoint and are scored with the same harness
(execute the predicted calls, compare end state) on the same 110 held-out tasks, so the
three arms are directly comparable: base, SFT, and SFT plus alignment.

    python build/train_align.py --method dpo --sft-adapter kirankunapuli/cpq-sft-qwen1.5b
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_eval_colab import (  # noqa: E402
    SUFFIX,
    download,
    evaluate,
    generate_all,
    load_jsonl,
    make_scorer,
    make_sft_config,
    pick_device,
    split_batch,
    _autocast_dtype,
)


def load_merged(base_id: str, adapter_id: str | None, torch, dtype, device="cpu"):
    """Load the base model and merge the SFT adapter into it, on `device`.

    Merging into a full-precision copy rather than layering a second adapter keeps the
    reference model unambiguous: the frozen merged weights are the reference, and DPO or
    GRPO trains a fresh adapter on top of them.

    The model is moved to the device here. Leaving it on the CPU while the eval thinks it
    is generating on the GPU is the difference between ten seconds and ten minutes.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype)
    if adapter_id:
        print(f"loading SFT adapter {adapter_id} and merging it into the base weights")
        model = PeftModel.from_pretrained(model, adapter_id).merge_and_unload()
    model = model.to(device)
    model.eval()
    print(f"policy on {next(model.parameters()).device}")
    return model, tok


def prompt_text(tok, row) -> str:
    """The exact prompt shape the eval harness uses, so training matches scoring."""
    msgs = [dict(m) for m in row["messages"][:2]] if "messages" in row else [dict(m) for m in row["prompt"][:2]]
    msgs[-1]["content"] += SUFFIX
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def make_reward(world_mod, tasks_by_id):
    """Reward 1.0 when the completion replays to the declared end state, else 0.0."""
    parse, score = make_scorer(world_mod)
    world0 = world_mod.canonical_task_set()[0]

    def as_text(c) -> str:
        if isinstance(c, str):
            return c
        if isinstance(c, (list, tuple)):
            out = []
            for m in c:
                out.append(str(m.get("content", "")) if isinstance(m, dict) else str(m))
            return "\n".join(out)
        return str(c)

    def reward(completions=None, task_id=None, **_kw):
        scores = []
        for completion, tid in zip(completions or [], task_id or []):
            calls = parse(as_text(completion))
            task = tasks_by_id.get(tid)
            if calls is None or task is None:
                scores.append(0.0)
                continue
            ok, _ = score(world0, task, calls)
            scores.append(1.0 if ok else 0.0)
        return scores

    return reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["dpo", "grpo"])
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft-adapter", default=None,
                    help="Hub id or local path of the SFT adapter to start from")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO/GRPO KL strength")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--out", default="align_results.json")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer

    device, dtype = pick_device(torch)
    if args.device != "auto":
        device = args.device
        dtype = _autocast_dtype(torch, device)
    print(f"device: {device}, dtype {dtype}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)} "
              f"compute {torch.cuda.get_device_capability(0)}")

    tmp = Path(tempfile.mkdtemp())
    files = download(tmp)
    sys.path.insert(0, str(files["world"].parent))
    import shutil
    shutil.copy(files["world"], tmp / "crm_world.py")
    import crm_world as world_mod

    world, tasks = world_mod.canonical_task_set()
    tasks_by_id = {t.task_id: t for t in tasks}
    test_rows = load_jsonl(files["test"])
    if args.limit_test:
        test_rows = test_rows[: args.limit_test]
    print(f"test={len(test_rows)}")

    model, tok = load_merged(args.base, args.sft_adapter, torch, dtype, device=device)

    # ---- baseline on whatever we start from, so the delta is against this arm's own parent
    print("\n=== STARTING POINT ===")
    start_raws = generate_all(model, tok, test_rows, SUFFIX, args.max_new_tokens)
    res_start = evaluate(world_mod, tasks_by_id, test_rows, start_raws, SUFFIX)
    print(f"start accuracy = {res_start['accuracy']:.3f} "
          f"({res_start['correct']}/{res_start['n']})  errors: {res_start['errors']}")

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM", target_modules="all-linear")

    if args.method == "dpo" and args.lr >= 1e-5:
        print(f"WARNING: --lr {args.lr:g} is at or above the 5e-5 that regressed this model "
              f"from 102/110 to 68/110 in the published run, where invalid identifiers came "
              f"back from 0 to 13. 5e-6 is the untested next point. Pass --lr explicitly if "
              f"you mean to reproduce the regression.")

    steps = max(1, len(load_jsonl(files["train"])) // max(1, args.batch * args.grad_accum))

    if args.method == "dpo":
        pref_path = tmp / "pref_train.jsonl"
        if not pref_path.exists():
            from huggingface_hub import hf_hub_download
            pref_path = Path(hf_hub_download("kirankunapuli/cpq-tooluse-verified",
                                             "pref_train.jsonl", repo_type="dataset"))
        pairs = load_jsonl(pref_path)
        if args.limit_train:
            pairs = pairs[: args.limit_train]
        rows = [{"prompt": prompt_text(tok, p), "chosen": p["chosen"],
                 "rejected": p["rejected"]} for p in pairs]
        print(f"dpo pairs={len(rows)}")
        cfg = make_sft_config(
            DPOConfig, _total_steps=int(steps * args.epochs),
            output_dir="align", num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
            logging_steps=10, save_strategy="no", max_length=args.max_length,
            beta=args.beta, report_to=[], seed=13,
            bf16=(device == "cuda" and dtype == torch.bfloat16),
            fp16=(device == "cuda" and dtype == torch.float16),
        )
        trainer = DPOTrainer(model=model, ref_model=None, args=cfg,
                             train_dataset=Dataset.from_list(rows), peft_config=lora,
                             processing_class=tok)
    else:
        train_rows = load_jsonl(files["train"])
        if args.limit_train:
            train_rows = train_rows[: args.limit_train]
        rows = [{"prompt": prompt_text(tok, r), "task_id": r["task_id"]}
                for r in train_rows]
        print(f"grpo prompts={len(rows)}")
        cfg = make_sft_config(
            GRPOConfig, _total_steps=int(steps * args.epochs),
            output_dir="align", num_train_epochs=args.epochs,
            per_device_train_batch_size=max(args.batch, args.num_generations),
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
            # GRPOConfig has no max_length: prompts are not truncated by a trainer arg,
            # only the sampled completion is capped. Passing max_length here would be
            # dropped silently, which is why the guard in make_sft_config exists.
            logging_steps=10, save_strategy="no",
            max_completion_length=args.max_completion_length,
            num_generations=args.num_generations, temperature=0.9, beta=args.beta,
            report_to=[], seed=13,
            bf16=(device == "cuda" and dtype == torch.bfloat16),
            fp16=(device == "cuda" and dtype == torch.float16),
        )
        trainer = GRPOTrainer(model=model, reward_funcs=[make_reward(world_mod, tasks_by_id)],
                             args=cfg, train_dataset=Dataset.from_list(rows),
                             peft_config=lora, processing_class=tok)

    print("\n=== TRAINING ===")
    trainer.train()
    trainer.save_model("align")
    tok.save_pretrained("align")

    print("\n=== ALIGNED MODEL ===")
    model.eval()
    raws = generate_all(trainer.model, tok, test_rows, SUFFIX, args.max_new_tokens)
    res = evaluate(world_mod, tasks_by_id, test_rows, raws, SUFFIX)
    print(f"aligned accuracy = {res['accuracy']:.3f} "
          f"({res['correct']}/{res['n']})  errors: {res['errors']}")

    from train_eval_colab import paired_ci
    point, lo, hi = paired_ci(res_start["per_item"], res["per_item"])
    print(f"delta vs starting point = {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    summary = {
        "method": args.method,
        "base_model": args.base,
        "sft_adapter": args.sft_adapter,
        "hyperparameters": {"epochs": args.epochs, "lr": args.lr, "beta": args.beta,
                            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                            "batch": args.batch, "grad_accum": args.grad_accum,
                            "max_length": args.max_length},
        "train_rows": len(rows),
        "test_rows": len(test_rows),
        "start": {k: v for k, v in res_start.items() if k != "per_item"},
        "aligned": {k: v for k, v in res.items() if k != "per_item"},
        "delta": {"point": point, "ci": [lo, hi], "n": res_start["n"]},
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
