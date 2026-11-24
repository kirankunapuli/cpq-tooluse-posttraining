"""One request, all the way through, in one run.

The page has six separate panels: a trace walker, a gate, a price flip, an ordering check, a
lifecycle walker and a failure router. Each one is a widget. A reader can play with any of them
and still not see the system, because the thing that makes this a system rather than five
experiments is that they are the same request in that order.

This script is that order. It takes one sentence of human typing and pushes it through every
stage the write-up describes, printing what each stage decided and why. Nothing is mocked and
nothing is re-implemented: it imports the same modules the corpus was built from.

    python build/demo_pipeline.py                 # the happy path
    python build/demo_pipeline.py --break approval
    python build/demo_pipeline.py --break order
    python build/demo_pipeline.py --scope         # an off-topic question
    python build/demo_pipeline.py --ask           # a request with a field missing

The last three exist because a system is only believable when you have watched it say no. The
happy path proves the plumbing; the broken variants prove the guards are wired to the plumbing
and not drawn on a slide next to it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parent
sys.path.insert(0, str(BUILD))

import input_normalizer as N                     # noqa: E402
import revenue_cloud_v2 as V2                    # noqa: E402
import scope_gate as SG                          # noqa: E402

REQUEST = "quote 3 hundred prod phon x 512 gb titanum, 22 pct off, bellows collage"
OFF_TOPIC = "what is the weather in Hyderabad tomorrow"
MISSING = "quote 40 units for bellows college"

PERSONA = V2.PERSONA_BY_ID["P1"]                 # Enterprise AE: 20% without approval
AS_OF = "2026-11-23"


# --------------------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------------------

def stage(n: int, title: str, note: str = "") -> None:
    print()
    print(f"  {'─' * 78}")
    print(f"  {n}. {title.upper()}")
    if note:
        print(f"     {note}")
    print(f"  {'─' * 78}")


def row(label: str, value, note: str = "") -> None:
    line = f"     {label:<14s} {value}"
    if note:
        line += f"    ({note})"
    print(line)


def money(x: float) -> str:
    return f"{x:,.0f}"


# --------------------------------------------------------------------------------------
# the stages
# --------------------------------------------------------------------------------------

def gate(text: str) -> bool:
    """Stage 1. Is this Revenue Cloud work at all?

    The n-gram channel only, so this runs with no model download. The shipped gate is the mean
    of this and an embedding classifier, which is stronger; the decision logic is identical and
    the band is what changes.
    """
    train, _test = SG.build_corpus()
    g = SG.ScopeGate(backend="ngram").fit(train)
    v = g.decide(text)

    stage(1, "scope gate", "a narrow model is still a general model underneath")
    row("decision", v.decision)
    row("probability", round(v.score, 3))
    row("band", f"[{g.low}, {g.high}]", "inside it the gate asks instead of guessing")
    row("layer", v.layer, v.reason)

    if v.decision == "refuse":
        print()
        print(f"     → {V2.REFUSAL_TEMPLATE}")
        return False
    if v.decision == "ask":
        print()
        print("     → the score is too close to the boundary to act on either way, so it asks")
        return False
    return True


def front_door(text: str) -> dict | None:
    """Stage 2. What did they actually mean?

    The dangerous output here is not a refusal, it is a confident wrong read: 300 units
    resolving to 3 units, or "titanum" resolving to black. So the resolver carries a tier, and
    an unresolvable slot becomes a question rather than a guess.
    """
    slots = N.parse(text)
    flat = slots.to_dict()
    flat.update(flat.get("attributes") or {})

    stage(2, "front door", "messy input, and a tier on every slot")
    if flat.get("sku"):
        row("sku", flat["sku"], f"tier {flat.get('sku_tier')}")
    if flat.get("account"):
        row("account", flat["account"], f"tier {flat.get('account_tier')}")
    if flat.get("quantity") is not None:
        row("quantity", int(flat["quantity"]), flat.get("quantity_how") or "")
    if flat.get("percent") is not None:
        row("percent", flat["percent"])
    for k, v in (flat.get("attributes") or {}).items():
        row(k, v, f"tier {flat.get('attribute_tiers', {}).get(k)}")

    if slots.questions:
        print()
        for q in slots.questions:
            print(f"     → ask: {q}")
        print()
        print("     a plan built on a guessed field is worse than no plan, so this stops here")
        return None
    return flat


def price(flat: dict, month: int = 11) -> dict:
    """Stage 3. The number does not live in the weights.

    Exact lookup against the effective-dated decision table, then two rules that are read from
    data rather than computed by the model. This is the whole of Q1 in one function: the model
    contributes the arguments, and the price is fetched.
    """
    records = V2.build_price_records(as_of=AS_OF)
    attrs = {k: v for k, v in (flat.get("attributes") or {}).items()}

    stage(3, "price, from the table and not from memory",
          "an exact lookup, a quantity tier, and a seasonal rule")
    rec = V2.lookup_price(records, flat["sku"], attrs, AS_OF)
    if rec is None:
        print(f"     no record for {flat['sku']} {attrs} at {AS_OF} → this becomes a question")
        return {}
    row("record", rec.record_id)
    row("unit price", money(rec.unit_price))
    row("snapshot", rec.snapshot_id, f"valid from {rec.valid_from}")
    row("attributes", ", ".join(f"{k}={v}" for k, v in sorted(rec.attrs.items())))

    qty = int(flat["quantity"])
    unit = rec.unit_price
    subtotal = unit * qty
    tier_pct = V2.quantity_discount(qty)
    seasonal_pct = V2.SEASONAL_RULE["extra_pct"] if month == V2.SEASONAL_RULE["month"] else 0.0
    discount_pct = float(flat.get("percent") or 0.0)

    after_tier = subtotal * (1 - tier_pct / 100)
    after_season = after_tier * (1 - seasonal_pct / 100)
    net = after_season * (1 - discount_pct / 100)

    print()
    row("list price", f"{money(unit)} x {qty} = {money(subtotal)}")
    row("quantity tier", f"{tier_pct:.1f}%", f"highest tier met at {qty}")
    month_name = ["January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December"][month - 1]
    row("seasonal", f"{seasonal_pct:.1f}%",
        "the December rule, applied" if month == 12 else f"{month_name}: outside the window")
    row("negotiated", f"{discount_pct:.1f}%")
    row("net", money(net))

    return {"unit": unit, "qty": qty, "net": net, "discount_pct": discount_pct,
            "record": rec, "needs_approval": discount_pct > PERSONA.max_discount}


def plan(flat: dict, priced: dict) -> list[dict]:
    """Stage 4. The sequence the interviewer called the code lifecycle."""
    attrs = {k: v for k, v in (flat.get("attributes") or {}).items()}
    calls = [
        {"name": "get_product", "arguments": {"product_id": flat["sku"]}},
        {"name": "get_price", "arguments": {"product_id": flat["sku"], "attributes": attrs}},
        {"name": "get_rules", "arguments": {"product_id": flat["sku"],
                                            "quantity": priced["qty"]}},
        {"name": "create_quote", "arguments": {"account_id": flat["account"],
                                               "pricebook": "USD-Standard"}},
        {"name": "add_quote_line_item", "arguments": {"pricebook_entry_id":
                                                      priced["record"].record_id,
                                                      "quantity": priced["qty"],
                                                      "discount_pct": priced["discount_pct"]}},
    ]
    if priced["needs_approval"]:
        calls.append({"name": "request_quote_approval",
                      "arguments": {"reason": f"discount {priced['discount_pct']:.1f}% exceeds "
                                              f"{PERSONA.max_discount:.0f}% for "
                                              f"{PERSONA.label}"}})
    calls.append({"name": "activate_quote", "arguments": {}})

    stage(4, "the plan", "four stages, and the last one cannot come before the first")
    for i, c in enumerate(calls, 1):
        print(f"     {i}. {c['name']}")
    ok, why = V2.check_stage_order(calls)
    print()
    row("stage order", "ok" if ok else f"rejected: {why}")
    return calls


def judge(flat: dict, priced: dict, calls: list[dict]) -> None:
    """Stage 5. Two independent checks, and they fail differently.

    The ordering check is a sequence property. The scoring check is a goal property. A plan can
    satisfy one and fail the other, which is the reason both exist.
    """
    task = V2.TaskV2(
        task_id="DEMO", persona_id=PERSONA.id, family="attribute_quote",
        instruction="(reconstructed from the request)",
        required_lookups=[{"tool": "get_price", "product_id": flat["sku"],
                           "attributes": (flat.get("attributes") or {})}],
        gold_calls=calls,
        declared={"net": priced["net"], "needs_approval": priced["needs_approval"]},
        expected_outcome="acted",
    )
    score = V2.score_tool_calls(task, calls)

    stage(5, "two checks that fail differently", "sequence is not the same as correctness")
    row("sequence ok", score["sequence_ok"])
    row("provenance ok", score.get("provenance_ok"))
    row("tool precision", score["tool_precision"])
    row("tool recall", score["tool_recall"])
    row("verdict", "accept" if score["correct"] else "reject")
    row("approval", "required" if priced["needs_approval"] else "not required",
        f"{priced['discount_pct']:.1f}% vs {PERSONA.max_discount:.0f}% authority")


def corrupt(flat: dict, priced: dict, calls: list[dict], how: str) -> None:
    """Stage 6. The same plan with one realistic mistake, and which guard catches it."""
    bad = [dict(c, arguments=dict(c.get("arguments") or {})) for c in calls]

    if how == "approval":
        bad = [c for c in bad if c["name"] != "request_quote_approval"]
        caught_by = "the goal check, not the ordering rule"
    elif how == "order":
        # The last layer called before the one it depends on. Same seven tools, one moved.
        line = next(c for c in bad if c["name"] == "add_quote_line_item")
        rest = [c for c in bad if c["name"] != "add_quote_line_item"]
        bad = rest[:1] + [line] + rest[1:]
        caught_by = "the ordering rule"
    else:                                            # price
        # A stale unit price. Every tool name, every position, unchanged.
        for c in bad:
            if c["name"] == "add_quote_line_item":
                c["arguments"]["unit_price"] = 999.0
        caught_by = "the argument check: the right tool, called with a price no lookup returned"

    stage(6, f"the same plan, one defect: {how}", f"expected to be caught by {caught_by}")
    for c in bad:
        print(f"     {c['name']}")
    ok, why = V2.check_stage_order(bad)

    task = V2.TaskV2(
        task_id="DEMO", persona_id=PERSONA.id, family="attribute_quote",
        instruction="(reconstructed from the request)",
        required_lookups=[{"tool": "get_price", "product_id": flat["sku"],
                           "attributes": (flat.get("attributes") or {})}],
        gold_calls=calls,
        declared={"net": priced["net"], "needs_approval": priced["needs_approval"]},
        expected_outcome="acted",
    )
    score = V2.score_tool_calls(task, bad)

    print()
    row("ordering rule", "passed" if ok else f"rejected: {why}")
    failing = [k for k in ("sequence_ok", "args_ok", "provenance_ok")
               if score.get(k) is False]
    row("goal check", "passed" if score["correct"] else "rejected",
        "correct" if score["correct"] else "failed property: " + ", ".join(
            k.replace("_ok", "") for k in failing))

    if how == "approval" and ok:
        # The ordering rule cannot see this one, by design: dropping a step leaves a valid
        # subsequence. This is why the second check exists and why it is not the same check.
        print()
        print("     the ordering rule passes this, because removing a step leaves a valid")
        print("     subsequence. Only the goal check sees it, and that is the argument for")
        print("     having two checks that fail on different properties.")

        # And in the v1 world the tool itself refuses, which is the third line of defence.
        import crm_world as CW
        w = CW.build_world(seed=CW.WORLD_SEED, n_accounts=CW.N_ACCOUNTS)
        acct = next(a for a in w.accounts.values() if a["name"] == "Bellows College Pvt")
        opp = next(o for o in w.opportunities.values() if o["account_id"] == acct["id"])
        pbe = next(e for e in w.pricebook_entries.values()
                   if e["pricebook"] == "USD-Standard" and e["product_id"] == "PROD-CPQ-PLATFORM")
        q = CW.execute(w, {"name": "create_quote", "arguments": 
                           {"opportunity_id": opp["id"], "pricebook": "USD-Standard"}})
        CW.execute(w, {"name": "add_quote_line_item", "arguments": 
                       {"quote_id": q["quote_id"], "pricebook_entry_id": pbe["id"],
                        "quantity": 10, "discount_pct": 25.9}})
        print()
        print("     and the tool layer refuses it before any check runs:")
        try:
            CW.execute(w, {"name": "activate_quote", "arguments": {"quote_id": q["quote_id"]}})
            print("       activate_quote succeeded -- that would be a bug")
        except CW.ToolError as exc:
            print(f"       ToolError: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("request", nargs="?", default=REQUEST)
    ap.add_argument("--scope", action="store_true", help="send an off-topic request instead")
    ap.add_argument("--ask", action="store_true", help="send a request with a field missing")
    ap.add_argument("--break", dest="breaker", choices=("approval", "order", "price"),
                    help="run the happy path, then the same plan with one defect")
    ap.add_argument("--month", type=int, default=11, help="month, for the seasonal rule")
    args = ap.parse_args()

    text = OFF_TOPIC if args.scope else MISSING if args.ask else args.request
    closing = ("one request, one refusal, zero tool calls. the gate answered it."
               if args.scope else
               "one request, one question back, zero tool calls. a missing field is asked for."
               if args.ask else
               "one request, one quote, eleven decisions behind it. that is the system.")

    def done() -> int:
        print()
        print(f"  {closing}")
        print()
        return 0

    print()
    print(f"  request: {text!r}")

    if not gate(text):
        return done()
    flat = front_door(text)
    if flat is None:
        return done()
    priced = price(flat, month=args.month)
    if not priced:
        return done()
    calls = plan(flat, priced)
    judge(flat, priced, calls)
    if args.breaker:
        corrupt(flat, priced, calls, args.breaker)
    return done()


if __name__ == "__main__":
    raise SystemExit(main())
