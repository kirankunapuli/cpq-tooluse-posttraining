"""Generate candidate tool-use traces, verify them by execution, keep only survivors.

The generator here is *simulated* so the pipeline runs with no API key and no GPU.
Swap `propose()` for a real model call (or a stronger-teacher distillation call) and
nothing downstream changes: the verifier is the part that decides what is true.

Run:  python build/generate.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from crm_world import (  # noqa: E402
    TOOL_SCHEMAS,
    World,
    canonical_task_set,
    execute,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SYSTEM_PROMPT = (
    "You are an enterprise CRM assistant. You answer by emitting a JSON array of tool "
    "calls, in order, and nothing else. Use only the tools provided. Never invent "
    "identifiers: look them up first.\n\nTools:\n"
    + json.dumps(TOOL_SCHEMAS, indent=2)
)

# Mutations that make a trace *look* plausible but break the end state.
BREAKING_MUTATIONS = [
    "hallucinated_id",
    "wrong_stage",
    "wrong_entity",
    "missing_step",
    "extra_task",
    "partial_escalation",
]


def applicable_mutations(task) -> list[str]:
    """Only offer mutations that can actually alter this task's trace.

    A mutation that silently no-ops makes a broken generator look good, because the
    unchanged trace still verifies. That inflates yield and hides nothing useful.
    """
    names = {c["name"] for c in task.gold_calls}
    args = [c["arguments"] for c in task.gold_calls]
    out = ["extra_task"]
    if any("opportunity_id" in a or "case_id" in a for a in args):
        out.append("hallucinated_id")
    if "update_opportunity_stage" in names:
        out += ["wrong_stage", "missing_step"]
    if names & {"create_task"}:
        out.append("wrong_entity")
    if "escalate_case" in names:
        out.append("partial_escalation")

    # ---- billing-ops mutations: each maps to a real CPQ/billing failure mode
    if "activate_quote" in names:
        out += ["wrong_quantity", "wrong_pricebook"]
        if "request_quote_approval" in names:
            out.append("skip_approval")
    if "amend_subscription_quantity" in names:
        out.append("wrong_subscription_qty")
    return out


def propose(task, other_accounts, rng, p_good: float) -> tuple[list[dict], str]:
    """Return (candidate_calls, label). Label is 'clean' or the mutation applied."""
    if rng.random() < p_good:
        return [dict(c, arguments=dict(c["arguments"])) for c in task.gold_calls], "clean"

    mut = rng.choice(applicable_mutations(task))
    calls = [dict(c, arguments=dict(c["arguments"])) for c in task.gold_calls]

    if mut == "hallucinated_id":
        for c in calls:
            if "opportunity_id" in c["arguments"]:
                c["arguments"]["opportunity_id"] = "OPP-9999-0"
                break
            if "case_id" in c["arguments"]:
                c["arguments"]["case_id"] = "CASE-9999-0"
                break

    elif mut == "wrong_stage":
        for c in calls:
            if c["name"] == "update_opportunity_stage":
                current = c["arguments"]["stage"]
                alternatives = [s for s in ("Qualification", "Proposal", "Negotiation")
                                if s != current]
                c["arguments"]["stage"] = rng.choice(alternatives)
                break

    elif mut == "wrong_entity":
        if other_accounts:
            alien = rng.choice(other_accounts)
            for c in calls:
                if c["name"] == "create_task":
                    c["arguments"]["account_id"] = alien
                    break

    elif mut == "missing_step":
        for i in range(len(calls) - 1, -1, -1):
            if calls[i]["name"] == "update_opportunity_stage":
                calls.pop(i)
                break

    elif mut == "extra_task":
        calls.append({"name": "create_task",
                      "arguments": {"account_id": task.account_id,
                                    "subject": "Unprompted follow-up",
                                    "priority": "Low"}})

    elif mut == "partial_escalation":
        for c in calls:
            if c["name"] == "escalate_case":
                c["arguments"]["priority"] = "High"
                break

    # ---- billing-ops mutations
    elif mut == "skip_approval":
        # The signature CPQ policy failure: activate a discounted quote without
        # routing it for approval. The tool itself refuses, so a correct environment
        # makes this loud rather than silent.
        calls = [c for c in calls if c["name"] != "request_quote_approval"]

    elif mut == "wrong_quantity":
        for c in calls:
            if c["name"] == "add_quote_line_item":
                c["arguments"]["quantity"] = int(c["arguments"]["quantity"]) + 10
                break

    elif mut == "wrong_pricebook":
        for c in calls:
            if c["name"] == "add_quote_line_item":
                pbe = c["arguments"]["pricebook_entry_id"]
                for alt in ("USD", "EUR", "APAC"):
                    candidate = pbe.replace(pbe.split("-")[1], alt, 1)
                    if candidate != pbe:
                        c["arguments"]["pricebook_entry_id"] = candidate
                        break
                break

    elif mut == "wrong_subscription_qty":
        for c in calls:
            if c["name"] == "amend_subscription_quantity":
                c["arguments"]["quantity"] = int(c["arguments"]["quantity"]) + 25
                break

    return calls, mut


# Collapse raw tool errors into a taxonomy that is stable across ids, so the rejection
# table is quotable ("135 wrong-pricebook") instead of 30 near-duplicate strings.
def categorise(errors: list[str]) -> str:
    if not errors:
        return "unknown"
    e = errors[0]
    for key, label in (
        ("pricebook mismatch", "pricebook_mismatch"),
        ("needs approval", "approval_threshold_violated"),
        ("unknown opportunity_id", "hallucinated_opportunity_id"),
        ("unknown case_id", "hallucinated_case_id"),
        ("unknown account_id", "hallucinated_account_id"),
        ("unknown pricebook_entry_id", "hallucinated_pricebook_entry_id"),
        ("unknown subscription_id", "hallucinated_subscription_id"),
        ("unknown invoice_id", "hallucinated_invoice_id"),
        ("only Active subscriptions", "inactive_subscription_amendment"),
        ("not in Draft status", "invalid_quote_state"),
        ("no line items", "empty_quote_activation"),
    ):
        if key in e:
            return label
    if "goal_not_satisfied" in errors:
        return "goal_not_satisfied"
    return "other_tool_error"


def verify(world: World, task, calls: list[dict]) -> tuple[bool, list[str]]:
    """A trace is accepted only if it replays without error AND reaches the goal state."""
    after = world.clone()
    errors: list[str] = []
    for i, call in enumerate(calls):
        try:
            execute(after, call)
        except Exception as e:  # ToolError and malformed-argument variants
            errors.append(f"step {i}: {type(e).__name__}: {e}")
    if errors:
        return False, errors
    if not task.check(world, after):
        return False, ["goal_not_satisfied"]
    return True, []


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    world, tasks = canonical_task_set()

    account_ids = list(world.accounts)
    verified_path = DATA / "traces_verified.jsonl"
    rejected_path = DATA / "traces_rejected.jsonl"

    stats: dict = {
        "tasks": len(tasks),
        "candidates_total": 0,
        "verified_total": 0,
        "by_family": defaultdict(lambda: {"candidates": 0, "verified": 0}),
        "by_label": defaultdict(lambda: {"candidates": 0, "verified": 0}),
        "rejection_reasons": Counter(),
    }

    K_CLEAN = 1   # one canonical trajectory per task
    K_NOISY = 2   # plus two plausible-but-likely-wrong attempts
    with verified_path.open("w") as fv, rejected_path.open("w") as fr:
        for task in tasks:
            others = [a for a in account_ids if a != task.account_id]
            for i in range(K_CLEAN + K_NOISY):
                calls, label = propose(task, others, rng, p_good=1.0 if i < K_CLEAN else 0.0)
                ok, errors = verify(world, task, calls)
                rec = {
                    "task_id": task.task_id,
                    "family": task.family,
                    "difficulty": task.difficulty,
                    "account_id": task.account_id,
                    "instruction": task.instruction,
                    "label": label,
                    "calls": calls,
                    "verified": ok,
                    "errors": errors,
                }
                stats["candidates_total"] += 1
                stats["by_family"][task.family]["candidates"] += 1
                stats["by_label"][label]["candidates"] += 1

                if ok:
                    stats["verified_total"] += 1
                    stats["by_family"][task.family]["verified"] += 1
                    stats["by_label"][label]["verified"] += 1
                    msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task.instruction},
                        {"role": "assistant", "content": json.dumps(calls)},
                    ]
                    fv.write(json.dumps({"messages": msgs, "task_id": task.task_id,
                                         "family": task.family, "label": label,
                                         "account_id": task.account_id}) + "\n")
                else:
                    stats["rejection_reasons"][categorise(errors)] += 1
                    fr.write(json.dumps(rec) + "\n")

    stats["by_family"] = dict(stats["by_family"])
    stats["by_label"] = dict(stats["by_label"])
    stats["rejection_reasons"] = dict(stats["rejection_reasons"])
    stats["yield"] = round(stats["verified_total"] / max(1, stats["candidates_total"]), 4)
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"tasks                : {stats['tasks']}")
    print(f"candidates           : {stats['candidates_total']}")
    print(f"verified             : {stats['verified_total']}  ({stats['yield']:.1%} yield)")
    print("\nby family:")
    for f, v in stats["by_family"].items():
        print(f"  {f:<22} {v['verified']:>4}/{v['candidates']:<4}")
    print("\nby generator label (verified / candidates):")
    for l, v in sorted(stats["by_label"].items(), key=lambda kv: -kv[1]["candidates"]):
        print(f"  {l:<20} {v['verified']:>4}/{v['candidates']:<4}")
    print("\ntop rejection reasons:")
    for r, n in sorted(stats["rejection_reasons"].items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {n:>4}  {r[:90]}")
    print(f"\nwrote {verified_path}")
    print(f"wrote {rejected_path}")

    diagnose_verifier_blindness(world, tasks)


def diagnose_verifier_blindness(world: World, tasks) -> None:
    """Probe my own verifier: what does it *fail to catch*?

    The goal predicate is over end state, so it is blind to wasted or wrong intermediate
    queries whenever they still yield the correct actions. Quantifying that blind spot is
    part of trusting the pipeline, a verifier you have not attacked is a verifier you
    do not know.
    """
    accepted = 0
    probed = 0
    for task in tasks:
        calls = [dict(c, arguments=dict(c["arguments"])) for c in task.gold_calls]
        target = next((c for c in calls if c["name"] == "list_opportunities"), None)
        if target is None:
            continue
        target["arguments"]["min_amount"] = 0
        target["arguments"]["stage"] = "Prospecting"
        probed += 1
        ok, _ = verify(world, task, calls)
        accepted += ok
    if probed:
        print(f"\nverifier blindness probe (drift lookup args): "
              f"{accepted}/{probed} still accepted = "
              f"{accepted / probed:.0%}, end-state checks ignore redundant queries")


if __name__ == "__main__":
    main()
