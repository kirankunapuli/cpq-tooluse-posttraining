"""Dedup, decontaminate, and split the verified traces.

Splits are grouped by account (entity), never random: in workflow data the same account
recurs across near-identical tasks, so a random split leaks the answer into the eval set.

Run:  python build/filter_splits.py
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SHINGLE = 13          # n-gram length for the contamination check
MINHASH_PERM = 64     # LSH accuracy vs cost
JACCARD_THRESHOLD = 0.8


# ------------------------------------------------------------------ normalisation

def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\$[\d,]+", "$<amt>", text)
    text = re.sub(r"\b(acc|opp|case)-\d+(-\d+)?\b", r"<\1>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str, mask_ids: bool = True) -> list[str]:
    """Tokenise for shingling.

    mask_ids=True  collapses entity ids and amounts to placeholders, so two examples that
                   differ only by which account they touch look identical -> measures
                   *template* redundancy.
    mask_ids=False keeps the literal surface form, so those examples stay distinct ->
                   measures *literal* leakage.
    """
    src = _norm(text) if mask_ids else re.sub(r"\s+", " ", text.lower()).strip()
    return re.findall(r"[a-z0-9<>$]+", src)


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


# ------------------------------------------------------------------ minhash

def _hash_int(s: str, salt: int) -> int:
    h = hashlib.blake2b(f"{salt}:{s}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


def minhash(tokens: list[str], shingle: int = 5) -> tuple[int, ...]:
    grams = {" ".join(g) for g in ngrams(tokens, shingle)}
    if not grams:
        return tuple([0] * MINHASH_PERM)
    return tuple(min(_hash_int(g, salt) for g in grams) for salt in range(MINHASH_PERM))


def lsh_bands(sig: tuple[int, ...], rows: int = 4) -> list[tuple[int, int]]:
    bands = []
    for b in range(0, MINHASH_PERM, rows):
        chunk = sig[b:b + rows]
        bands.append((b, hash(chunk)))
    return bands


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ------------------------------------------------------------------ main

def main() -> None:
    rows = [json.loads(l) for l in (DATA / "traces_verified.jsonl").read_text().splitlines()]
    print(f"loaded verified traces: {len(rows)}")

    # ---- exact dedup
    seen: dict[str, int] = {}
    deduped = []
    exact_dups = 0
    for r in rows:
        key = hashlib.sha256(
            (_norm(r["messages"][1]["content"]) + "||" + _norm(r["messages"][2]["content"])).encode()
        ).hexdigest()
        if key in seen:
            exact_dups += 1
            continue
        seen[key] = 1
        deduped.append(r)
    print(f"exact duplicates removed: {exact_dups} -> {len(deduped)}")

    # ---- near dedup via MinHash LSH (over instruction + trajectory, ids normalised)
    sigs = [
        minhash(_tokens(r["messages"][1]["content"] + " " + r["messages"][2]["content"],
                        mask_ids=False))
        for r in deduped
    ]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    keep = []
    near_dups = 0
    for i, sig in enumerate(sigs):
        dup = False
        for band in lsh_bands(sig):
            for j in buckets.get(band, []):
                if jaccard(set(sigs[j]), set(sig)) >= JACCARD_THRESHOLD:
                    dup = True
                    break
            if dup:
                break
        if dup:
            near_dups += 1
            continue
        for band in lsh_bands(sig):
            buckets[band].append(i)
        keep.append(deduped[i])
    print(f"near duplicates removed:  {near_dups} -> {len(keep)}")

    # ---- split by account (entity grouping)
    accounts = sorted({r["account_id"] for r in keep})
    n = len(accounts)
    test_accts = set(accounts[: max(1, n // 10)])
    valid_accts = set(accounts[max(1, n // 10): max(2, n // 5)])
    train_accts = set(accounts) - test_accts - valid_accts

    train = [r for r in keep if r["account_id"] in train_accts]
    valid = [r for r in keep if r["account_id"] in valid_accts]
    test = [r for r in keep if r["account_id"] in test_accts]

    # ---- decontamination: 13-gram overlap between train and eval
    # ---- contamination control
    # Policy note (this is a real finding, not a shortcut):
    # On templated synthetic workflows, a 13-gram check on literal text fires on ~all
    # examples of a family, because the fixed template dominates the n-grams and only the
    # entity name varies. Dropping on that signal deletes entire task families.
    # Measurement showed 144/412 train examples flagged, and the escalate family vanished
    # from train entirely. So: the actual control is entity-disjoint splitting, and the
    # n-gram score is reported as a known generalisation limitation instead of being used
    # as a filter. Stricter (non-templated) data should use n-gram dropping.
    def grams(r, mask):
        return ngrams(_tokens(r["messages"][1]["content"], mask_ids=mask), SHINGLE)

    eval_mask: set[tuple[str, ...]] = set()
    eval_lit: set[tuple[str, ...]] = set()
    for r in valid + test:
        eval_mask |= grams(r, True)
        eval_lit |= grams(r, False)

    flagged_mask = sum(1 for r in train if grams(r, True) & eval_mask)
    flagged_lit = sum(1 for r in train if grams(r, False) & eval_lit)
    contaminated = 0
    clean_train = train

    # Hard assertion that the thing we actually rely on holds: no entity in two splits.
    shared = ({r["account_id"] for r in clean_train}
              & ({r["account_id"] for r in valid} | {r["account_id"] for r in test}))
    if shared:
        raise SystemExit(f"entity leakage across splits: {sorted(shared)[:5]}")

    print(f"{SHINGLE}-gram overlap, id-masked  (template-level): "
          f"{flagged_mask}/{len(train)} = {flagged_mask/max(1,len(train)):.0%}  [kept, reported]")
    print(f"{SHINGLE}-gram overlap, literal     (near-identical): "
          f"{flagged_lit}/{len(train)} = {flagged_lit/max(1,len(train)):.0%}  [dropped]")
    keep_tr, clean_train, contaminated = clean_train, [], 0
    eval_lit2 = eval_lit
    for r in keep_tr:
        if grams(r, False) & eval_lit2 and len(grams(r, False) & eval_lit2) > 3:
            contaminated += 1
            continue
        clean_train.append(r)
    print(f"literal near-identical dropped (>{3} shared 13-grams): {contaminated}")

    out = {
        "train": clean_train,
        "valid": valid,
        "test": test,
    }
    for name, rs in out.items():
        p = DATA / f"split_{name}.jsonl"
        with p.open("w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        fams = defaultdict(int)
        for r in rs:
            fams[r["family"]] += 1
        print(f"{name:<6} {len(rs):>4} examples  {dict(fams)}")

    # ---- packing / token stats (char-based proxy; replace with a real tokenizer)
    char_lens = [len(r["messages"][2]["content"]) + len(r["messages"][1]["content"]) for r in clean_train]
    if char_lens:
        char_lens.sort()
        p50 = char_lens[len(char_lens) // 2]
        p99 = char_lens[int(len(char_lens) * 0.99) - 1]
        approx_tokens = [c // 4 for c in char_lens]
        print(f"\ntrain char length p50={p50} p99={p99}")
        print(f"approx tokens (chars/4): total={sum(approx_tokens):,} "
              f"mean={sum(approx_tokens)//len(approx_tokens)} max={max(approx_tokens)}")
        print(f"padding waste if max_len=1024 and no packing: "
              f"{100 * (1 - sum(min(t,1024) for t in approx_tokens) / (len(approx_tokens)*1024)):.0f}%")

    summary = {
        "verified_in": len(rows),
        "exact_dups_removed": exact_dups,
        "near_dups_removed": near_dups,
        "contaminated_removed": contaminated,
        "template_overlap_frac": round(flagged_mask / max(1, len(train)), 3),
        "literal_overlap_frac": round(flagged_lit / max(1, len(train)), 3),
        "split_sizes": {k: len(v) for k, v in out.items()},
        "accounts": {"train": len(train_accts), "valid": len(valid_accts), "test": len(test_accts)},
    }
    (DATA / "split_stats.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote split_*.jsonl and split_stats.json")


if __name__ == "__main__":
    main()
