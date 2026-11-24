"""Reproduce every number the write-up quotes, in one run.

The page makes about forty quantitative claims. A reader has no way to tell which of them are
measured and which are remembered, so this script is the answer: every claim below is asserted
against the artefact that produces it, and the run fails if any of them drifts.

That is the whole reason it exists. A number in a document is a promise, and a promise with no
executable check is a hope. If I change the discount threshold, or add a task family, or retrain,
this script is what tells me the write-up is now lying.

Three kinds of check, and the difference matters when you quote the result:

  ASSERT   the value is fixed. It must not move, because a document quotes it.
  COMPUTE  the value is derived here from the artefacts, and printed so it can be compared.
  SKIP     a dependency is absent. Says so, does not fail, and does not pretend.

The ML-dependent harnesses (retrieval over embeddings, the scope gate ensemble, Laya) are skipped
on a bare interpreter and run when the stack is present. Nothing here needs a GPU.

Run:

    python build/verify_all.py            # full check, skips what it cannot run
    python build/verify_all.py --strict   # a skip is a failure

Exit code is 0 only when every check that ran passed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

sys.path.insert(0, str(BUILD))


# --------------------------------------------------------------------------------------
# Tiny harness. Deliberately not pytest: this must run anywhere with no install.
# --------------------------------------------------------------------------------------

@dataclass
class Check:
    claim: str
    value: str
    source: str
    kind: str            # ASSERT | COMPUTE | SKIP
    ok: bool = True

    def line(self) -> str:
        mark = "FAIL" if not self.ok else {"ASSERT": "ok  ", "COMPUTE": "mea ", "SKIP": "skip"}[
            self.kind]
        return f"  {mark} {self.claim:46s} {self.value:>22s}   {self.source}"


CHECKS: list[Check] = []


def _assert(claim: str, measured, expected, source: str) -> None:
    ok = measured == expected
    CHECKS.append(Check(claim, str(measured), source, "ASSERT", ok))
    if not ok:
        CHECKS.append(Check("  expected", str(expected), source, "FAILED", False))


def _compute(claim: str, measured, source: str) -> None:
    CHECKS.append(Check(claim, str(measured), source, "COMPUTE"))


def _skip(claim: str, why: str, source: str) -> None:
    CHECKS.append(Check(claim, why, source, "SKIP"))


def _have(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def _jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _run(script: str, *args: str, timeout: int = 600) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(BUILD / script), *args],
                          capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    return proc.returncode, proc.stdout + proc.stderr


# --------------------------------------------------------------------------------------
# 1. The world
# --------------------------------------------------------------------------------------

def check_world() -> None:
    from crm_world import N_ACCOUNTS, TOOL_SCHEMAS, WORLD_SEED, build_world

    world = build_world(seed=WORLD_SEED, n_accounts=N_ACCOUNTS)
    _assert("accounts", len(world.accounts), 520, "crm_world.build_world")
    _assert("opportunities", len(world.opportunities), 1302, "crm_world.build_world")
    _assert("products", len(world.products), 8, "crm_world.build_world")
    _assert("price book entries", len(world.pricebook_entries), 24, "crm_world.build_world")
    _assert("subscriptions", len(world.subscriptions), 1020, "crm_world.build_world")
    _assert("invoices", len(world.invoices), 508, "crm_world.build_world")
    _assert("cases", len(world.cases), 799, "crm_world.build_world")
    _assert("tools in the schema", len(TOOL_SCHEMAS), 16, "crm_world.TOOL_SCHEMAS")

    # The three price books must hold the same products at fixed ratios, because the page says
    # the relationship is a rule and a test can assert it.
    by_product: dict[str, dict[str, float]] = {}
    for entry in world.pricebook_entries.values():
        by_product.setdefault(entry["product_id"], {})[entry["pricebook"]] = entry["unit_price"]
    ratios = {(round(v["EUR-Standard"] / v["USD-Standard"], 4),
               round(v["APAC-Partner"] / v["USD-Standard"], 4))
              for v in by_product.values() if len(v) == 3}
    _assert("price book ratios (EUR, APAC) of USD", ratios, {(0.92, 0.78)}, "pricebook_entries")

    phone = world.pricebook_entries["PBE-USD-DATA-ZERO"]
    _assert("Data Platform Zero Copy unit price", int(phone["unit_price"]), 12000, "PBE-USD-DATA-ZERO")


def check_attribute_pricing() -> None:
    """16 attribute combinations, 12 distinct prices, one SKU."""
    import itertools

    import revenue_cloud_v2 as V2

    base = V2.BASE_PRICE["PROD-PHONE-X"]
    deltas = [a for a in V2.ATTRIBUTE_TABLE if a.product_id == "PROD-PHONE-X"]
    storage = [a for a in deltas if a.attribute == "storage"]
    colour = [a for a in deltas if a.attribute == "colour"]
    combos = list(itertools.product(storage, colour))
    prices = {base + s.delta + c.delta for s, c in combos}
    _assert("PROD-PHONE-X attribute combinations", len(combos), 16, "revenue_cloud_v2.ATTRIBUTE_TABLE")
    _assert("PROD-PHONE-X distinct prices", len(prices), 12, "revenue_cloud_v2.ATTRIBUTE_TABLE")
    _assert("price records in the decision table", len(V2.build_price_records()), 22,
            "revenue_cloud_v2.build_price_records")


# --------------------------------------------------------------------------------------
# 2. The dataset
# --------------------------------------------------------------------------------------

def check_dataset() -> None:
    stats = json.loads((DATA / "stats.json").read_text())
    _assert("proposals generated", stats["candidates_total"], 3300, "data/stats.json")
    _assert("canonical traces accepted", stats["verified_total"], 1100, "data/stats.json")
    _assert("tasks", stats["tasks"], 1100, "data/stats.json")

    labels = stats["by_label"]
    _assert("canonical plans that passed", labels["clean"]["verified"], 1100, "data/stats.json")
    mutated = {k: v for k, v in labels.items() if k != "clean"}
    _assert("mutated proposals", sum(v["candidates"] for v in mutated.values()), 2200,
            "data/stats.json")
    _assert("mutated proposals that passed", sum(v["verified"] for v in mutated.values()), 0,
            "data/stats.json")
    _assert("defect labels", len(labels), 11, "data/stats.json")

    reasons = stats["rejection_reasons"]
    _assert("rejections explained by reason", sum(reasons.values()), 2200, "data/stats.json")
    tool_level = (reasons["hallucinated_opportunity_id"] + reasons["pricebook_mismatch"]
                  + reasons["hallucinated_case_id"] + reasons["approval_threshold_violated"])
    _compute("caught by a tool refusing", tool_level, "data/stats.json")
    _compute("caught by state comparison", sum(reasons.values()) - tool_level, "data/stats.json")

    for split, rows, accounts in (("train", 872, 180), ("valid", 108, 23), ("test", 110, 22)):
        data = _jsonl(DATA / f"split_{split}.jsonl")
        _assert(f"{split} rows", len(data), rows, f"data/split_{split}.jsonl")
        _assert(f"{split} accounts", len({r["account_id"] for r in data}), accounts,
                f"data/split_{split}.jsonl")

    train = {r["account_id"] for r in _jsonl(DATA / "split_train.jsonl")}
    test = {r["account_id"] for r in _jsonl(DATA / "split_test.jsonl")}
    _assert("accounts shared between train and test", len(train & test), 0, "data/split_*.jsonl")

    pref = json.loads((DATA / "pref_stats.json").read_text())
    _assert("preference pairs", pref["total_pairs"], 2180, "data/pref_stats.json")
    _assert("tasks with no rejection to pair against", pref["tasks_without_rejection"], 0,
            "data/pref_stats.json")


# --------------------------------------------------------------------------------------
# 3. The four training runs
# --------------------------------------------------------------------------------------

def check_training() -> None:
    dpo = json.loads((RESULTS / "dpo_lr5e6.json").read_text())
    _assert("DPO 5e-6 start accuracy (of 110)", dpo["start"]["correct"], 102, "results/dpo_lr5e6.json")
    _assert("DPO 5e-6 end accuracy (of 110)", dpo["aligned"]["correct"], 102, "results/dpo_lr5e6.json")

    grpo = json.loads((RESULTS / "grpo_ckpt15.json").read_text())
    frac = grpo["observed"]["frac_reward_zero_std"]
    _assert("GRPO unanimous sample groups", round(frac, 4), 0.8875, "results/grpo_ckpt15.json")
    _assert("GRPO end accuracy (of 110)", grpo["aligned"]["correct"], 102, "results/grpo_ckpt15.json")
    _compute("GRPO steps run of planned", f"{grpo['steps_done']}/{grpo['steps_planned']}",
             "results/grpo_ckpt15.json")


# --------------------------------------------------------------------------------------
# 4. The scope gate
# --------------------------------------------------------------------------------------

def check_scope_gate() -> None:
    import random

    import scope_gate as SG

    train, test = SG.build_corpus()
    fams = sorted({r[2] for r in train})
    val_fams = set(random.Random(11).sample(fams, max(4, len(fams) // 4)))
    fit_rows = [r for r in train if r[2] not in val_fams]

    _assert("held-out utterances", len(test), 240, "scope_gate.build_corpus")
    _assert("held-out template families", len({r[2] for r in test}), 8, "scope_gate.build_corpus")
    _assert("free-text probes", len(SG.PROBES), 12, "scope_gate.PROBES")
    _assert("templates shared with training",
            len({r[2] for r in test} & {r[2] for r in fit_rows}), 0, "scope_gate.build_corpus")

    def tally(gate):
        tp = fp = tn = fn = 0
        for text, label, _tpl in test:
            v = gate.decide(text)
            if v.decision == "act":
                tp += label == "in"
                fp += label == "out"
            else:
                tn += label == "out"
                fn += label == "in"
        return tp, fp, tn, fn

    tp, fp, tn, fn = tally(SG.ScopeGate(backend="ngram").fit(train))
    _assert("ngram gate, in scope acted on", tp, 120, "scope_gate.ScopeGate(ngram)")
    _assert("ngram gate, out of scope refused", tn, 120, "scope_gate.ScopeGate(ngram)")
    _assert("ngram gate, out of scope leaked", fp, 0, "scope_gate.ScopeGate(ngram)")

    # The numbers the write-up quotes are the shipped ensemble's, and that needs sklearn.
    # Asserting them from a bare interpreter would fail for the wrong reason, so say which
    # gate produced the number instead of pretending there is one.
    if SG.sklearn_available():
        ens = SG.ScopeGate(backend="ensemble").fit(train)
        tp, fp, tn, fn = tally(ens)
        _assert("ensemble gate, in scope acted on", tp, 120, "scope_gate.ScopeGate(ensemble)")
        _assert("ensemble gate, out of scope refused", tn, 120, "scope_gate.ScopeGate(ensemble)")
        _assert("ensemble gate, out of scope leaked", fp, 0, "scope_gate.ScopeGate(ensemble)")
        _assert("ensemble gate, shipped band",
                (round(ens.low, 3), round(ens.high, 3)), (0.535, 0.655),
                "scope_gate.ScopeGate(ensemble)")
        ng = SG.ScopeGate(backend="ngram").fit(train)
        _assert("ngram gate, shipped band", (round(ng.low, 3), round(ng.high, 3)),
                (0.695, 0.815), "scope_gate.ScopeGate(ngram)")
        emb = SG.ScopeGate(backend="embed").fit(train)
        _assert("embedding gate, shipped band", (round(emb.low, 3), round(emb.high, 3)),
                (0.365, 0.485), "scope_gate.ScopeGate(embed)")
    else:
        _skip("ensemble arm (the numbers in the write-up)",
              "sklearn absent; above is the n-gram arm alone", "build/scope_gate.py")

    if SG.laya_available():
        _compute("decision model available", "laya", "scope_gate.laya_available")
    else:
        _skip("decision model arms", "laya not installed", "build/laya_gate.py")


# --------------------------------------------------------------------------------------
# 5. Retrieval
# --------------------------------------------------------------------------------------

def check_retrieval() -> None:
    import retrieval as R

    _assert("policy documents", len(R.CORPUS), 31, "retrieval.CORPUS")
    _assert("labelled queries", len(R.QUERIES), 38, "retrieval.QUERIES")

    bm25 = R.evaluate(R.RuleIndex(R.CORPUS, use_char=False, use_dense=False))
    _assert("BM25 recall@1", round(bm25["recall@1"], 3), 0.895, "retrieval.evaluate")

    if not R.DenseIndex(R.CORPUS).__class__.__module__:
        return
    try:
        import _model_cache
        have_dense = _model_cache.available()
    except Exception:                                        # noqa: BLE001
        have_dense = False

    if have_dense:
        fused = R.evaluate(R.RuleIndex(R.CORPUS))
        _assert("three-channel recall@1", round(fused["recall@1"], 3), 0.974, "retrieval.evaluate")
        _assert("three-channel recall@3", round(fused["recall@3"], 3), 1.0, "retrieval.evaluate")
    else:
        _skip("dense channel recall", "numpy/sklearn/sentence-transformers absent",
              "build/retrieval.py")


# --------------------------------------------------------------------------------------
# 6. The front door
# --------------------------------------------------------------------------------------

def check_normalizer() -> None:
    import input_normalizer as N

    _assert("messy requests that resolve", len(N.MESSY), 10, "input_normalizer.MESSY")
    _assert("requests that must be questioned", len(N.MUST_ASK), 4, "input_normalizer.MUST_ASK")

    # to_dict nests the attribute values one level down; flatten so a check can name a slot
    # the way a reader would ("storage", "colour") rather than "attributes.storage".
    def flatten(slots: dict) -> dict:
        flat = dict(slots)
        flat.update(slots.get("attributes") or {})
        flat.update({k + "_tier": v for k, v in (slots.get("attribute_tiers") or {}).items()})
        return flat

    wrong = 0
    for text, expected in N.MESSY:
        flat = flatten(N.parse(text).to_dict())
        for key, want in expected.items():
            if flat.get(key) != want:
                wrong += 1
    _assert("slots resolved to a wrong value", wrong, 0, "input_normalizer.parse")

    leaked = sum(1 for text, _why in N.MUST_ASK if N.parse(text).ready)
    _assert("requests questioned that were resolved silently", leaked, 0, "input_normalizer.parse")


# --------------------------------------------------------------------------------------
# 7. The scripts' own self-checks
# --------------------------------------------------------------------------------------

def check_selftests() -> None:
    for script, label, expected in (("revenue_cloud_v2.py", "revenue_cloud_v2 selftests", 32),
                                    ("input_normalizer.py", "input_normalizer selftests", 33),
                                    ("scope_gate.py", "scope_gate selftests", 10)):
        code, out = _run(script)
        found = None
        for line in out.splitlines():
            if "/" in line and " passed" in line:
                found = int(line.split("/")[0].strip())
        if found is None:
            CHECKS.append(Check(label, "no summary line", script, "FAILED", False))
        else:
            _assert(label, found, expected, script)
            _assert(f"  {script} exit", code, 0, script)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def check_page() -> None:
    """The write-up itself, checked as a file.

    This script's other checks verify the artefacts. This one verifies the prose, because a
    number can be right in the code and stale on the page, and that is the failure a reader of the
    page would see. The page lives outside this repo, so the check skips when it cannot find it.
    """
    page = Path(os.environ.get("PAGE") or
                ROOT.parent / "career-preparation/salesforce-post-training/demo/index.html")
    if not page.exists():
        CHECKS.append(Check("write-up page", "not found", str(page), "SKIP"))
        return
    html = page.read_text(encoding="utf-8")

    must = [
        ("ensemble band on the page", "[0.535, 0.655]"),
        ("n-gram band on the page", "[0.695, 0.815]"),
        ("embedding band on the page", "[0.365, 0.485]"),
        ("the disagreement case, n-gram arm", "0.536"),
        ("the disagreement case, embedding arm", "0.650"),
        ("the disagreement case, mean", "0.593"),
        ("v2 selftest count on the page", "32 pipeline self-tests"),
        ("retrieval recall@1 on the page", "0.974"),
        ("retrieval MRR on the page", "0.987"),
        ("BM25 baseline on the page", "0.895"),
    ]
    for claim, needle in must:
        CHECKS.append(Check(claim, needle, page.name, "ASSERT", needle in html))

    stale = ["[0.705, 0.825]", "[0.565, 0.685]", "29 pipeline self-tests"]
    for needle in stale:
        CHECKS.append(Check(f"page does not still say {needle}", "absent", page.name,
                            "ASSERT", needle not in html))


def main() -> int:
    strict = "--strict" in sys.argv

    for name, fn in (("world", check_world),
                     ("attribute pricing", check_attribute_pricing),
                     ("dataset", check_dataset),
                     ("training runs", check_training),
                     ("scope gate", check_scope_gate),
                     ("retrieval", check_retrieval),
                     ("front door", check_normalizer),
                     ("write-up page", check_page),
                     ("script self-checks", check_selftests)):
        print(f"[{name}]")
        try:
            fn()
        except Exception as exc:                             # noqa: BLE001
            CHECKS.append(Check(name, f"crashed: {type(exc).__name__}", "-", "FAILED", False))
            print(f"  !! {type(exc).__name__}: {exc}")

    print()
    print(f"  {'claim':50s} {'measured':>22s}   source")
    print("  " + "-" * 104)
    for c in CHECKS:
        print(c.line())

    page = Path(os.environ.get("PAGE") or
                ROOT.parent / "career-preparation/salesforce-post-training/demo/index.html")
    meta_claim = "assertion count quoted on the page"
    if page.exists():
        quoted = re.search(r"(\d+) assertions", page.read_text(encoding="utf-8"))
        n_assert = len([c for c in CHECKS if c.kind == "ASSERT" and c.claim != meta_claim])
        if quoted:
            CHECKS.append(Check(meta_claim, quoted.group(1), f"== {n_assert + 1}", "ASSERT",
                                int(quoted.group(1)) == n_assert + 1))
            print(CHECKS[-1].line())

    failed = [c for c in CHECKS if not c.ok]
    skipped = [c for c in CHECKS if c.kind == "SKIP"]
    asserted = [c for c in CHECKS if c.kind == "ASSERT"]

    print()
    print(f"  {len(asserted)} asserted, {len(asserted) - len(failed)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped")
    if failed:
        print()
        print("  the write-up no longer matches the code:")
        for c in failed:
            print(f"    {c.claim} -> {c.value}")
        return 1
    if skipped and strict:
        print()
        print("  --strict: a skipped check counts as a failure")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
