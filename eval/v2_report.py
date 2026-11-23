"""Report the numbers the screening interview asked for, from the v2 corpus.

Answers, in order:
  1. How many personas, and what is each one's data share and outcome mix
  2. Whether persona authority binds in code rather than in the prompt
  3. Tool-level accuracy potential, per tool and per family
  4. The staleness demonstration: flip a price and show which kind of trace survives

    python eval/v2_report.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))

import revenue_cloud_v2 as v2  # noqa: E402


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    records = v2.build_price_records()
    tasks = v2.make_tasks(records)
    by_persona = defaultdict(list)
    for t in tasks:
        by_persona[t.persona_id].append(t)

    # ---------------------------------------------------------------- 1. personas
    rule("1. PERSONAS: how many, and what each one generates")
    print(f"{'id':4} {'label':18} {'role':16} {'disc':>5} {'tasks':>6} {'share':>7}  outcome mix")
    print("-" * 78)
    for p in v2.PERSONAS:
        ts = by_persona[p.id]
        mix = Counter(t.expected_outcome for t in ts)
        mix_s = " ".join(f"{k}={v}" for k, v in sorted(mix.items()))
        share = len(ts) / len(tasks) * 100
        print(f"{p.id:4} {p.label:18} {p.role:16} {p.max_discount:5.1f} "
              f"{len(ts):6} {share:6.1f}%  {mix_s}")

    print(f"\ntotal tasks: {len(tasks)}")
    print("every persona present:",
          set(by_persona) == {p.id for p in v2.PERSONAS})
    print("every outcome present:",
          {t.expected_outcome for t in tasks} == {"acted", "asked", "refused", "escalated"})

    # ---------------------------------------------------------------- 2. authorisation
    rule("2. PERSONA AUTHORITY: does it bind in code, or only in the prompt?")
    same_request = [
        t for t in tasks
        if t.family == "attribute_quote" and t.declared.get("discount_pct", 0) >= 25
    ]
    print("requests at or above a 25% negotiated discount, by persona:")
    print(f"{'persona':18} {'outcome':10} {'applied disc':>12}")
    print("-" * 78)
    seen = set()
    for t in same_request:
        key = (t.persona_id, t.expected_outcome)
        if key in seen:
            continue
        seen.add(key)
        print(f"{v2.PERSONA_BY_ID[t.persona_id].label:18} {t.expected_outcome:10} "
              f"{t.declared['discount_pct']:11.1f}%")
    ae_acted_high = [t for t in same_request
                     if t.persona_id == "P1" and t.expected_outcome == "acted"
                     and t.declared["discount_pct"] > 20]
    print(f"\nAE tasks that acted above its 20% limit: {len(ae_acted_high)}  (must be 0)")
    print("This is enforced by the generator's authority check, which is code, not a prompt.")

    # ---------------------------------------------------------------- 3. tooling
    rule("3. TOOL-LEVEL SURFACE: what the executor has to get right")
    tool_counts = Counter()
    for t in tasks:
        for c in t.gold_calls:
            tool_counts[c["name"]] += 1
    print(f"{'tool':26} {'gold calls':>11} {'tasks needing it':>17}")
    print("-" * 78)
    for tool, n in tool_counts.most_common():
        n_tasks = sum(1 for t in tasks if any(c["name"] == tool for c in t.gold_calls))
        print(f"{tool:26} {n:11} {n_tasks:17}")

    lookups = [t for t in tasks if t.required_lookups]
    print(f"\ntasks that MUST retrieve before acting: {len(lookups)} "
          f"({len(lookups)/len(tasks)*100:.1f}%)")
    print("a trace that skips the lookup cannot score correct, by construction")

    # ---------------------------------------------------------------- 4. decision table
    rule("4. DECISION TABLE: why the model must not compute prices")
    combos = Counter(r.product_id for r in records.values())
    print(f"price records generated: {len(records)}")
    for pid, n in sorted(combos.items()):
        print(f"  {pid}: {n} attribute combinations")
    print(f"quantity tiers: {[(q, p) for q, p in v2.QUANTITY_TIERS]}")
    print(f"seasonal rule: month {v2.SEASONAL_RULE['month']} adds "
          f"{v2.SEASONAL_RULE['extra_pct']}%")
    distinct = len({r.unit_price for r in records.values()
                    if r.product_id == "PROD-PHONE-X"})
    print(f"\nPROD-PHONE-X has ONE sku and {distinct} distinct prices.")
    print("That is the interviewer's point: one SKU, many prices, and none of them")
    print("is a model decision.")

    # ---------------------------------------------------------------- 5. staleness
    rule("5. STALENESS: flip a price, see which kind of trace survives")
    product, attrs, qty, disc = "PROD-PHONE-X", {"storage": "128GB", "colour": "black"}, 10, 0.0
    old = v2.lookup_price(records, product, attrs, "2026-11-23")
    old_net = round(qty * old.unit_price * (1 - disc / 100.0), 2)
    print(f"before the release: unit {old.unit_price:,.2f}  -> net {old_net:,.2f}")

    flipped = v2.build_price_records(overrides={product: 89900.0})
    new = v2.lookup_price(flipped, product, attrs, "2026-11-23")
    new_net = round(qty * new.unit_price * (1 - disc / 100.0), 2)
    print(f"after the release:  unit {new.unit_price:,.2f}  -> net {new_net:,.2f}")

    print("\ntwo candidate traces for the same request:")
    print(f"  A. memorised   writes net = {old_net:,.2f}   (the value it saw in training)")
    print(f"  B. grounded    calls get_price, writes net = {new_net:,.2f}")

    print(f"\nA against the new price table: WRONG by {new_net - old_net:,.2f}")
    print("B against the new price table: correct, because the number came from the lookup")
    print("\nSame weights, same request, one day apart. The difference is whether the price")
    print("was in the weights or in the index. That is the whole argument for retrieval,")
    print("and it needs no retraining and no GPU to demonstrate.")


if __name__ == "__main__":
    main()
