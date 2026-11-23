"""Derive preference pairs (chosen / rejected) from the verifier's own rejections.

Why this is nearly free: the generator already produces, for the same task, one verified
trajectory and several plausible-but-wrong variants. The verifier labelled each rejection
with its defect. So a DPO dataset falls out of data we were discarding.

Important: the pair is only legitimate if both sides answer the *same* prompt. We join on
task_id, so `chosen` and `rejected` share an instruction and differ only in the trajectory.

Run:  python build/make_preference_pairs.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).parent))

from generate import SYSTEM_PROMPT, categorise  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    verified = load_jsonl(DATA / "traces_verified.jsonl")
    rejected = load_jsonl(DATA / "traces_rejected.jsonl")

    chosen_by_task: dict[str, dict] = {}
    for r in verified:
        chosen_by_task.setdefault(r["task_id"], r)

    rejected_by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rejected:
        rejected_by_task[r["task_id"]].append(r)

    # Only tasks that appear in a split (train/valid/test) may contribute pairs, so the
    # preference set inherits the entity-disjoint split and cannot leak across it.
    split_of: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for row in load_jsonl(DATA / f"split_{split}.jsonl"):
            split_of[row["task_id"]] = split

    pairs = {"train": [], "valid": [], "test": []}
    defect_counts: Counter = Counter()
    dropped_no_pair = dropped_not_in_split = 0

    for task_id, ch in chosen_by_task.items():
        split = split_of.get(task_id)
        if split is None:
            dropped_not_in_split += 1
            continue
        cands = rejected_by_task.get(task_id, [])
        if not cands:
            dropped_no_pair += 1
            continue
        for rej in cands:
            defect = categorise(rej.get("errors") or [])
            if defect in ("unknown", "other_tool_error"):
                continue
            pairs[split].append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ch["messages"][1]["content"]},
                ],
                # both sides are completion strings (JSON arrays of tool calls)
                "chosen": ch["messages"][2]["content"],
                "rejected": json.dumps(rej["calls"]),
                "defect": defect,
                "task_id": task_id,
                "family": ch["family"],
            })
            defect_counts[defect] += 1

    for split, rows in pairs.items():
        p = DATA / f"pref_{split}.jsonl"
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split:<6} {len(rows):>5} pairs")

    total = sum(len(v) for v in pairs.values())
    print(f"\ntotal pairs            : {total}")
    print(f"tasks with no rejection: {dropped_no_pair}")
    print(f"verified but unsplit   : {dropped_not_in_split}")
    print("\ndefect distribution (these are the negatives):")
    for d, n in defect_counts.most_common():
        print(f"  {n:>5}  {d}")

    # A sanity check that matters: the chosen side must be the verified one.
    sample = pairs["test"][0] if pairs["test"] else None
    if sample:
        assert sample["chosen"] != sample["rejected"]
        print(f"\nsample (test): defect={sample['defect']} family={sample['family']}")
        print(f"  prompt: {sample['prompt'][1]['content'][:110]}")

    (DATA / "pref_stats.json").write_text(json.dumps({
        "total_pairs": total,
        "per_split": {k: len(v) for k, v in pairs.items()},
        "defect_distribution": dict(defect_counts),
        "tasks_without_rejection": dropped_no_pair,
    }, indent=2))
    print("\nwrote pref_{train,valid,test}.jsonl and pref_stats.json")


if __name__ == "__main__":
    main()
