# Colab / HF Jobs training + evaluation, one self-contained file
#
# Paste this whole file into a Colab cell, or run:
#     python colab_train_eval.py --base Qwen/Qwen2.5-1.5B-Instruct
#
# It downloads the dataset, scores the BASE model on the held-out split, trains a QLoRA
# adapter, scores the TUNED model on the same items, and reports a paired bootstrap CI on
# the delta. Nothing is hardcoded from a previous run: every number it prints it measured.
#
# Requires a CUDA GPU. Colab free T4 is enough (~30-45 min).

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

REPO = "kirankunapuli/cpq-tooluse-verified"
SCHEMA_TOOLS = None  # filled from the downloaded world module


# --------------------------------------------------------------------- data


def download(dst: Path) -> dict[str, Path]:
    """Use local files when running from a clone; fall back to the Hub otherwise.

    The data is deliberately not committed (it regenerates in ~20s), so a fresh clone
    still needs one Hub fetch or one `python build/generate.py`. Preferring local files
    means the notebook and the script exercise the same code you can read.
    """
    here = Path(__file__).resolve().parent
    local = {
        "train": here / "data" / "split_train.jsonl",
        "valid": here / "data" / "split_valid.jsonl",
        "test": here / "data" / "split_test.jsonl",
        "world": here / "build" / "crm_world.py",
    }
    if all(p.exists() for p in local.values()):
        print("using local data/ and build/ (repo clone)")
        return local

    from huggingface_hub import hf_hub_download

    print("local data not found; fetching from the Hub")
    files = {s: Path(hf_hub_download(REPO, f"split_{s}.jsonl", repo_type="dataset"))
             for s in ("train", "valid", "test")}
    files["world"] = Path(hf_hub_download(REPO, "code/crm_world.py", repo_type="dataset"))
    return files


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------- scoring
# Deliberately duplicated (tiny) rather than imported, so this file stands alone.
# The rules are identical to eval/harness.py: replay, then compare the state diff.


def make_scorer(world_mod):
    def parse(raw: str):
        import re
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        i, j = text.find("["), text.rfind("]")
        if i == -1 or j < i:
            return None
        try:
            calls = json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(calls, list) or not all(
                isinstance(c, dict) and "name" in c for c in calls):
            return None
        return calls

    def score(world, task, calls) -> tuple[bool, str]:
        after = world.clone()
        for call in calls:
            try:
                world_mod.execute(after, call)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "unknown" in msg and "_id" in msg:
                    return False, "invalid_id"
                return False, "invalid_argument"
        if not task.check(world, after):
            return False, "goal_not_satisfied"
        return True, "ok"

    return parse, score


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def paired_ci(a: dict, b: dict, iters: int = 5000, seed: int = 0):
    shared = [t for t in a if t in b]
    diffs = [int(b[t]) - int(a[t]) for t in shared]
    if not diffs:
        return 0.0, 0.0, 0.0
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(iters))
    return point, stats[int(0.025 * iters)], stats[int(0.975 * iters)]


# --------------------------------------------------------------------- model


def split_batch(enc):
    """apply_chat_template returns a BatchEncoding, not a bare tensor.

    Depending on the transformers version you get a dict-like with input_ids and
    attention_mask, or just a tensor. Normalise both, because passing the dict straight
    into generate() fails on `.shape`.
    """
    if hasattr(enc, "get") and hasattr(enc, "keys"):
        return enc.get("input_ids"), enc.get("attention_mask")
    return enc, None


def encode_chat(tok, msgs, thinking: bool = False):
    """apply_chat_template across model families and transformers versions.

    Three things vary between checkpoints and versions:
      - return_dict: newer transformers returns a BatchEncoding, older returns a tensor
      - enable_thinking: hybrid reasoning models (Qwen3, some others) emit a long
        reasoning trace before the answer unless it is switched off. For this task that
        trace would eat the generation budget and break JSON parsing, so we ask for it to
        be off where the template understands the flag.
      - the flag is simply absent on non-reasoning models

    Try the richest call first and fall back, rather than branching on model names.
    """
    attempts = [
        {"add_generation_prompt": True, "return_tensors": "pt",
         "return_dict": True, "enable_thinking": thinking},
        {"add_generation_prompt": True, "return_tensors": "pt", "return_dict": True},
        {"add_generation_prompt": True, "return_tensors": "pt"},
        {"add_generation_prompt": True},
    ]
    last = None
    for kwargs in attempts:
        try:
            return tok.apply_chat_template(msgs, **kwargs)
        except (TypeError, KeyError) as e:
            last = e
            continue
        except Exception as e:  # jinja UndefinedError and friends
            if type(e).__name__ in ("UndefinedError", "TemplateError"):
                last = e
                continue
            raise
    raise RuntimeError(f"apply_chat_template failed for every calling convention: {last}")


def make_sft_config(SFTConfig, **kwargs):
    """Instantiate SFTConfig using only the keywords this TRL build accepts.

    TRL and Transformers move arguments around between versions. A Colab runtime with a
    different TRL will raise TypeError on something like warmup_ratio, which is a wasted
    run. Filter against the actual signature and say what was dropped, rather than
    pinning versions that will drift again.
    """
    import inspect

    params = set(inspect.signature(SFTConfig.__init__).parameters)
    has_var_kw = any(
        q.kind is inspect.Parameter.VAR_KEYWORD
        for q in inspect.signature(SFTConfig.__init__).parameters.values()
    )
    internal = {"_total_steps"}
    accepted, dropped = {}, []
    for key, value in kwargs.items():
        if key in internal:
            continue
        if key in params or has_var_kw:
            accepted[key] = value
        else:
            dropped.append(key)

    # Translate warmup_ratio to warmup_steps wherever the latter exists. Newer
    # Transformers deprecates the ratio, older TRL does not accept it at all.
    if "warmup_ratio" in kwargs and "warmup_steps" in params:
        ratio = kwargs["warmup_ratio"]
        accepted.pop("warmup_ratio", None)
        accepted["warmup_steps"] = max(1, int(kwargs.get("_total_steps", 0) * ratio))
        if "warmup_ratio" in dropped:
            dropped.remove("warmup_ratio")
    if "warmup_ratio" in dropped and "warmup_steps" in params:
        steps = max(1, int(kwargs.get("_total_steps", 0) * kwargs["warmup_ratio"]))
        accepted["warmup_steps"] = steps
        dropped.remove("warmup_ratio")
    # Translate names that moved between releases. A silently dropped max_length means the
    # default sequence length truncates every example, which is exactly the failure that
    # made the first run report 0/110 for reasons nobody could see.
    aliases = {"max_length": ("max_seq_length",)}
    for canonical, other_names in aliases.items():
        if canonical not in dropped:
            continue  # accepted as-is, or never requested
        present = [n for n in other_names if n in params]
        if not present:
            raise RuntimeError(
                f"this TRL build accepts neither {canonical} nor {other_names}, so the "
                f"sequence length cannot be set. Refusing to train rather than silently "
                f"truncate every example. Available: {sorted(params)}")
        chosen = present[0]
        accepted[chosen] = kwargs[canonical]
        dropped.remove(canonical)
        print(f"  note: this TRL build calls {canonical} '{chosen}'")

    if dropped:
        print(f"  note: this TRL build has no {dropped}; continuing without them")
    cfg = SFTConfig(**accepted)

    # Only check truncation when a sequence length was actually requested. GRPOConfig has
    # no such argument, so the check must not assume one.
    if "max_length" in kwargs:
        effective = None
        for name in ("max_length", "max_seq_length"):
            if getattr(cfg, name, None) is not None:
                effective = getattr(cfg, name)
                break
        print(f"  sequence length the trainer will use: {effective}")
        if effective != kwargs["max_length"]:
            raise RuntimeError(
                f"trainer resolved max length {effective}, expected "
                f"{kwargs['max_length']}; refusing to train on truncated examples")
    masked = getattr(cfg, "assistant_only_loss", None)
    if not masked:
        print("  WARNING: this TRL build cannot mask the prompt, so loss includes the "
              "system prompt. Training still runs but spends most of its budget on context.")
    return cfg


def _autocast_dtype(torch, device):
    """The half-precision dtype a device actually supports.

    A T4 is Turing, compute 7.5, with no native bf16. torch.cuda.is_bf16_supported() is not a
    usable probe: it reports True on Turing because bf16 is emulated in software, while
    Trainer refuses to start on those cards with "bf16 is only supported on GPUs with
    compute capability of at least 8.0". Gate on the hardware capability instead.
    """
    if device == "cuda":
        try:
            major = torch.cuda.get_device_capability(0)[0]
        except Exception:  # noqa: BLE001
            return torch.float32
        return torch.bfloat16 if major >= 8 else torch.float16
    if device == "mps":
        # bf16 halves the memory footprint versus fp32, which is what makes a 1.5B model
        # plus a 2k-token sequence fit on a 16 GB laptop.
        return torch.bfloat16
    return torch.float32


def pick_device(torch):
    """CUDA first, then Apple Metal, then CPU. Returns (device, dtype)."""
    if torch.cuda.is_available():
        return "cuda", _autocast_dtype(torch, "cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", _autocast_dtype(torch, "mps")
    return "cpu", _autocast_dtype(torch, "cpu")


def load_model(model_id: str, quant_config, torch, device=None, dtype=None):
    """Load with a dtype/torch_dtype fallback across transformers versions.

    device_map is only used for quantised CUDA loads. On MPS and CPU the model is moved
    explicitly, because accelerate's device_map does not target Metal.
    """
    from transformers import AutoModelForCausalLM

    dt = dtype or torch.bfloat16
    kwargs = {}
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        kwargs["device_map"] = "auto"
    last = None
    for key in ("dtype", "torch_dtype"):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **{key: dt}, **kwargs)
        except (TypeError, ValueError) as e:
            last = e
            if key == "torch_dtype":
                raise
            print(f"  ({key} not accepted, trying the other name)")
            continue
        if quant_config is None and device:
            model = model.to(device)
        return model
    raise RuntimeError(f"could not load {model_id}: {last}")


def generate_all(model, tok, rows, suffix: str, max_new_tokens: int) -> dict[str, str]:
    import torch

    out = {}
    model.eval()
    for i, r in enumerate(rows):
        msgs = [dict(m) for m in r["messages"][:2]]
        msgs[-1]["content"] += suffix
        input_ids, attention_mask = split_batch(encode_chat(tok, msgs))
        input_ids = input_ids.to(model.device)
        extra = {}
        if attention_mask is not None:
            extra["attention_mask"] = attention_mask.to(model.device)
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 **extra)
        out[r["task_id"]] = tok.decode(gen[0][input_ids.shape[-1]:],
                                      skip_special_tokens=True)
        if (i + 1) % 20 == 0:
            print(f"  generated {i + 1}/{len(rows)}", flush=True)
    return out


def evaluate(world_mod, tasks_by_id, rows, raws, suffix: str) -> dict:
    world = world_mod.canonical_task_set()[0]
    parse, score = make_scorer(world_mod)
    results, cats = {}, {}
    for r in rows:
        tid = r["task_id"]
        task = tasks_by_id[tid]
        calls = parse(raws.get(tid, ""))
        if calls is None:
            results[tid] = False
            cats["parse_error"] = cats.get("parse_error", 0) + 1
            continue
        ok, cat = score(world, task, calls)
        results[tid] = ok
        if not ok:
            cats[cat] = cats.get(cat, 0) + 1
    k, n = sum(results.values()), len(results)
    lo, hi = wilson(k, n)
    return {"n": n, "correct": k, "accuracy": k / n if n else 0.0,
            "ci": [lo, hi], "errors": cats, "per_item": results}


SUFFIX = "\nRespond with a JSON array of tool calls and nothing else."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="answers here top out around 209 tokens")
    ap.add_argument("--max-length", type=int, default=2560,
                    help="system prompt is ~1791 tokens, so anything under ~2100 "
                         "silently truncates the answer")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--limit-train", type=int, default=0,
                    help="use only the first N training rows, for a smoke test")
    ap.add_argument("--limit-test", type=int, default=0,
                    help="use only the first N test rows, for a smoke test")
    ap.add_argument("--push-to", default=None, help="HF repo id to push the adapter to")
    ap.add_argument("--skip-base", action="store_true",
                    help="reuse a cached baseline instead of re-running it")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    device, dtype = pick_device(torch)
    if args.device != "auto":
        device = args.device
        dtype = _autocast_dtype(torch, device)
    if device == "cuda":
        print(f"device: cuda ({torch.cuda.get_device_name(0)}), dtype {dtype}")
        print(f"  dtype selected: {dtype}, half-precision autocast on")
        # TRL computes NLL with a chunked kernel that accumulates in fp32 through
        # torch.addmm(out_dtype=...). That needs compute capability 8.0 or newer, so a T4
        # (7.5) dies mid-training with "gemm input type BFloat16 and output type float is
        # only supported for CUDA devices with compute capability 8.0 or higher".
        # Newer torch enables that path by version alone, not by hardware, so switch it
        # off here and let it fall back to an upcasting addmm.
        major = torch.cuda.get_device_capability(0)[0]
        if major < 8:
            try:
                from trl.kernels import chunked_logprob

                chunked_logprob._MM_OUT_DTYPE = False
                print(f"  compute {major}.x < 8.0: using the portable fp32 logprob path")
            except Exception as e:  # noqa: BLE001
                print(f"  could not adjust TRL's logprob kernel: {e}")
    else:
        print(f"device: {device}, dtype {dtype}")
    tmp = Path(tempfile.mkdtemp())
    files = download(tmp)
    sys.path.insert(0, str(files["world"].parent))
    import shutil
    shutil.copy(files["world"], tmp / "crm_world.py")
    import crm_world as world_mod

    train_rows = load_jsonl(files["train"])
    test_rows = load_jsonl(files["test"])
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
        print(f"smoke test: training on {len(train_rows)} rows only")
    if args.limit_test:
        test_rows = test_rows[: args.limit_test]
        print(f"smoke test: evaluating on {len(test_rows)} rows only")
    world, tasks = world_mod.canonical_task_set()
    tasks_by_id = {t.task_id: t for t in tasks}
    print(f"train={len(train_rows)} test={len(test_rows)}")

    # The 1791-token system prompt makes every example long, and the LM head logits at
    # 2560 tokens dominate VRAM. On a 16 GB T4, batch 4 at this length OOMs. Cap it
    # rather than let the run die twenty minutes in, and say so.
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if args.max_length >= 2048 and args.batch > 2 and vram_gb < 24:
            print(f"  reducing per-device batch {args.batch} -> 2 (effective stays "
                  f"{args.batch * args.grad_accum}); {args.max_length}-token sequences "
                  f"at batch {args.batch} need more than the {vram_gb:.0f} GB available")
            args.grad_accum = (args.batch * args.grad_accum) // 2
            args.batch = 2

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = None
    if device == "cuda":
        try:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=dtype,
                                     bnb_4bit_use_double_quant=True)
        except Exception as e:  # noqa: BLE001
            print(f"  4-bit unavailable ({e}); loading in {dtype} instead")
    else:
        print("  4-bit needs CUDA; using plain LoRA here")
    base = load_model(args.base, bnb, torch, device=device, dtype=dtype)

    # ---- baseline
    base_path = Path("base_preds.json")
    if args.skip_base and base_path.exists():
        base_raws = json.loads(base_path.read_text())
    else:
        print("\n=== BASE MODEL (no adapter) ===")
        base_raws = generate_all(base, tok, test_rows, SUFFIX, args.max_new_tokens)
        base_path.write_text(json.dumps(base_raws))
    res_base = evaluate(world_mod, tasks_by_id, test_rows, base_raws, SUFFIX)
    lo, hi = res_base["ci"]
    print(f"base accuracy = {res_base['accuracy']:.3f}  "
          f"({res_base['correct']}/{res_base['n']}, 95% CI [{lo:.3f}, {hi:.3f}])")
    print(f"base errors: {res_base['errors']}")

    # ---- train
    print("\n=== TRAINING ===")
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
    steps_per_epoch = max(1, len(train_rows) // max(1, args.batch * args.grad_accum))
    est_total_steps = int(steps_per_epoch * args.epochs)
    cfg = make_sft_config(
        SFTConfig,
        _total_steps=est_total_steps,
        output_dir="adapter",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",
        max_length=args.max_length,  # full conversations top out at 2030 tokens
        packing=False,               # verify loss-mask boundaries before enabling
        assistant_only_loss=True,    # the only tokens that should carry loss
        # Autocast only on CUDA. Metal already runs the weights in bf16 and the Trainer's
        # autocast flags are not supported there.
        bf16=(device == "cuda" and dtype == torch.bfloat16),
        fp16=(device == "cuda" and dtype == torch.float16),
        report_to=[],
        seed=13,
    )
    # TRL 1.x rejects a plain list here; older versions accepted one.
    train_ds = Dataset.from_list(train_rows)
    trainer_kwargs = dict(model=base, args=cfg, train_dataset=train_ds, peft_config=lora)
    try:
        trainer = SFTTrainer(processing_class=tok, **trainer_kwargs)
    except TypeError as e:
        if "processing_class" not in str(e):
            raise
        print("  note: this TRL build calls processing_class 'tokenizer'")
        trainer = SFTTrainer(tokenizer=tok, **trainer_kwargs)
    trainer.train()
    trainer.save_model("adapter")
    tok.save_pretrained("adapter")

    # ---- tuned
    # The trainer leaves its adapter attached to `base`. Wrapping that again would stack
    # two adapters and silently change what "tuned" means, so load a clean copy of the
    # base weights and attach the saved adapter exactly once.
    print("\n=== TUNED MODEL ===")
    del trainer, base
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    clean = load_model(args.base, bnb, torch, device=device, dtype=dtype)
    tuned = PeftModel.from_pretrained(clean, "adapter").eval()
    tuned_raws = generate_all(tuned, tok, test_rows, SUFFIX, args.max_new_tokens)
    res_tuned = evaluate(world_mod, tasks_by_id, test_rows, tuned_raws, SUFFIX)
    lo2, hi2 = res_tuned["ci"]
    print(f"tuned accuracy = {res_tuned['accuracy']:.3f}  "
          f"({res_tuned['correct']}/{res_tuned['n']}, 95% CI [{lo2:.3f}, {hi2:.3f}])")
    print(f"tuned errors: {res_tuned['errors']}")

    # ---- delta
    point, dlo, dhi = paired_ci(res_base["per_item"], res_tuned["per_item"])
    print(f"\npaired delta = {point:+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}] "
          f"n={res_base['n']}")

    # ---- format vs capability
    fmt_base = res_base["errors"].get("parse_error", 0)
    fmt_tuned = res_tuned["errors"].get("parse_error", 0)
    # Items the base model already produced parseable output for: no format headroom,
    # so any accuracy movement there is capability rather than format compliance.
    parse_fn = make_scorer(world_mod)[0]
    base_parsed = {t for t, raw in base_raws.items() if parse_fn(raw) is not None}
    gain_on_parsed = sum(res_tuned["per_item"][t] - res_base["per_item"][t]
                         for t in base_parsed) / max(1, len(base_parsed))
    print(f"parse failures: base {fmt_base} -> tuned {fmt_tuned} "
          f"(of n={res_base['n']})")
    print(f"delta on items the base already parsed ({len(base_parsed)} items): "
          f"{gain_on_parsed:+.3f}  <- capability, not format")

    summary = {
        "base_model": args.base,
        "hyperparameters": {"epochs": args.epochs, "lr": args.lr,
                            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                            "batch": args.batch, "grad_accum": args.grad_accum,
                            "packing": False, "assistant_only_loss": True,
                            "max_length": args.max_length},
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "base": {k: v for k, v in res_base.items() if k != "per_item"},
        "tuned": {k: v for k, v in res_tuned.items() if k != "per_item"},
        "paired_delta": {"point": point, "ci": [dlo, dhi], "n": res_base["n"]},
        "format": {"parse_fail_base": fmt_base, "parse_fail_tuned": fmt_tuned,
                   "delta_on_already_parsed": gain_on_parsed,
                   "n_already_parsed": len(base_parsed)},
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")

    if args.push_to:
        print(f"\n=== PUSHING adapter to {args.push_to} ===")
        tuned.push_to_hub(args.push_to)
        print(f"https://huggingface.co/{args.push_to}")

    print("\nNext: paste results.json into the model card, and put the delta + CI on your CV.")


if __name__ == "__main__":
    main()
