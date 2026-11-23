"""Evaluation harness: score tool-use predictions by *executing* them.

Success is defined as: the predicted calls replay without error against the world AND
the resulting state satisfies the task goal. Nothing about the text is scored, so a
model cannot win by sounding right.

Usage
-----
    python eval/harness.py --selftest
    python eval/harness.py --predictions my_model.jsonl --split test
    python eval/harness.py --predictions a.jsonl --compare b.jsonl --split test

Predictions file format (one JSON object per line):
    {"task_id": "T0042", "calls": [{"name": "...", "arguments": {...}}, ...]}
or
    {"task_id": "T0042", "raw": "```json\n[ ... ]\n```"}

`raw` is parsed leniently so fenced output from a chat model can be scored directly;
parse failures are counted as a separate category rather than silently dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))

from crm_world import canonical_task_set, execute  # noqa: E402

DATA = ROOT / "data"


# ------------------------------------------------------------------ parsing

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_calls(rec: dict) -> tuple[list[dict] | None, str | None]:
    """Return (calls, error_category). error_category is None on success."""
    if isinstance(rec.get("calls"), list):
        return rec["calls"], None
    raw = rec.get("raw")
    if not isinstance(raw, str):
        return None, "no_output"
    text = raw.strip()
    m = _JSON_BLOCK.search(text)
    if m:
        text = m.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None, "parse_error"
    try:
        calls = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None, "parse_error"
    if not isinstance(calls, list) or not all(isinstance(c, dict) and "name" in c for c in calls):
        return None, "parse_error"
    return calls, None


# ------------------------------------------------------------------ scoring

def score_one(world, task, calls: list[dict]) -> tuple[bool, str]:
    after = world.clone()
    for call in calls:
        try:
            execute(after, call)
        except Exception as e:
            msg = str(e)
            if "unknown" in msg and ("opportunity_id" in msg or "case_id" in msg or "account_id" in msg):
                return False, "invalid_id"
            if "unknown tool" in msg:
                return False, "unknown_tool"
            return False, "invalid_argument"
    if not task.check(world, after):
        return False, "goal_not_satisfied"
    return True, "ok"


def evaluate(world, tasks_by_id: dict, split_rows: list[dict],
             preds: dict[str, dict]) -> dict:
    results = []
    for row in split_rows:
        tid = row["task_id"]
        task = tasks_by_id[tid]
        rec = preds.get(tid)
        if rec is None:
            results.append({"task_id": tid, "family": row["family"],
                            "ok": False, "category": "missing_prediction", "steps": 0})
            continue
        calls, err = parse_calls(rec)
        if err:
            results.append({"task_id": tid, "family": row["family"],
                            "ok": False, "category": err, "steps": 0})
            continue
        ok, cat = score_one(world, task, calls)
        results.append({"task_id": tid, "family": row["family"], "ok": ok,
                        "category": cat, "steps": len(calls),
                        "gold_steps": row.get("gold_steps")})

    n = len(results)
    k = sum(r["ok"] for r in results)
    lo, hi = wilson(k, n)
    by_family = defaultdict(lambda: [0, 0])
    for r in results:
        by_family[r["family"]][1] += 1
        by_family[r["family"]][0] += r["ok"]
    return {
        "n": n, "correct": k,
        "accuracy": k / n if n else 0.0,
        "ci95": [lo, hi],
        "by_family": {f: {"correct": v[0], "n": v[1],
                          "accuracy": v[0] / v[1]} for f, v in by_family.items()},
        "error_taxonomy": dict(Counter(r["category"] for r in results if not r["ok"])),
        "mean_steps": sum(r["steps"] for r in results) / n if n else 0.0,
        "results": results,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(a: dict, b: dict, iterations: int = 5000,
                     seed: int = 0) -> tuple[float, float, float]:
    """CI on accuracy(b) - accuracy(a) over the items both systems answered."""
    b_ids = {r["task_id"] for r in b["results"]}
    shared = [r["task_id"] for r in a["results"] if r["task_id"] in b_ids]
    a_map = {r["task_id"]: r["ok"] for r in a["results"]}
    b_map = {r["task_id"]: r["ok"] for r in b["results"]}
    diffs = [int(b_map[t]) - int(a_map[t]) for t in shared]
    if not diffs:
        return (0.0, 0.0, 0.0)
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    stats = []
    for _ in range(iterations):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        stats.append(s)
    stats.sort()
    return (point, stats[int(0.025 * iterations)], stats[int(0.975 * iterations)])


def report(name: str, res: dict) -> None:
    lo, hi = res["ci95"]
    print(f"\n=== {name} ===")
    print(f"n={res['n']}  correct={res['correct']}  accuracy={res['accuracy']:.3f}  "
          f"95% Wilson CI=[{lo:.3f}, {hi:.3f}]  mean_steps={res['mean_steps']:.2f}")
    for f, v in sorted(res["by_family"].items()):
        print(f"  {f:<22} {v['correct']:>4}/{v['n']:<4} = {v['accuracy']:.2f}")
    if res["error_taxonomy"]:
        print("  failures:")
        for cat, cnt in sorted(res["error_taxonomy"].items(), key=lambda kv: -kv[1]):
            print(f"    {cnt:>4}  {cat}")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions")
    ap.add_argument("--compare", help="second predictions file; reports paired delta (second minus first)")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    world, tasks = canonical_task_set()
    tasks_by_id = {t.task_id: t for t in tasks}
    split_rows = load_jsonl(DATA / f"split_{args.split}.jsonl")
    for r in split_rows:
        r["gold_steps"] = len(json.loads(r["messages"][2]["content"]))

    if args.selftest:
        selftest(world, tasks_by_id, split_rows)
        return

    if not args.predictions:
        ap.error("--predictions is required unless --selftest")

    preds = {r["task_id"]: r for r in load_jsonl(Path(args.predictions))}
    res = evaluate(world, tasks_by_id, split_rows, preds)
    report(Path(args.predictions).name, res)

    if args.compare:
        preds_b = {r["task_id"]: r for r in load_jsonl(Path(args.compare))}
        res_b = evaluate(world, tasks_by_id, split_rows, preds_b)
        report(Path(args.compare).name, res_b)
        point, lo, hi = paired_bootstrap(res, res_b)
        print(f"\npaired delta (compare - base): {point:+.3f}  "
              f"95% bootstrap CI=[{lo:+.3f}, {hi:+.3f}]  n_pairs={res['n']}")
        print("  report the CI, not just the point estimate.")


def selftest(world, tasks_by_id, split_rows) -> None:
    """Validate the harness against known-good and known-bad predictions.

    This is a harness test, not a model result. It exists so that a real number from a
    real checkpoint can be trusted once it arrives.
    """
    print("HARNESS SELFTEST, no model involved; these are fixtures.")
    gold = {r["task_id"]: {"task_id": r["task_id"], "calls": json.loads(r["messages"][2]["content"])}
            for r in split_rows}
    res_gold = evaluate(world, tasks_by_id, split_rows, gold)
    report("gold trajectories (expect 1.000)", res_gold)

    def mutate(kind):
        out = {}
        for r in split_rows:
            calls = json.loads(r["messages"][2]["content"])
            calls = [dict(c, arguments=dict(c["arguments"])) for c in calls]
            if kind == "hallucinated_id":
                done = False
                for c in calls:
                    for key in ("opportunity_id", "case_id"):
                        if key in c["arguments"]:
                            c["arguments"][key] = "OPP-0000-9"
                            done = True
                            break
                    if done:
                        break
            elif kind == "dropped_step":
                if len(calls) > 1:
                    calls = calls[:-1]
            elif kind == "extra_task":
                calls.append({"name": "create_task",
                              "arguments": {"account_id": r["account_id"],
                                            "subject": "unprompted", "priority": "Low"}})
            out[r["task_id"]] = {"task_id": r["task_id"], "calls": calls}
        return out

    for kind in ("hallucinated_id", "dropped_step", "extra_task"):
        res = evaluate(world, tasks_by_id, split_rows, mutate(kind))
        report(f"adversarial fixture: {kind} (expect < 1.000)", res)

    res_unparseable = evaluate(world, tasks_by_id, split_rows,
                               {r["task_id"]: {"task_id": r["task_id"], "raw": "I think so!"}
                                for r in split_rows})
    report("unparseable fixture (expect 0.000, all parse_error)", res_unparseable)

    assert res_gold["accuracy"] == 1.0, "gold must score 1.0"
    assert res_unparseable["accuracy"] == 0.0, "garbage must score 0.0"
    print("\nselftest passed: harness accepts gold, rejects garbage and corrupted traces.")


if __name__ == "__main__":
    main()
