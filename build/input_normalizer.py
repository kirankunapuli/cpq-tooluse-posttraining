"""Turn what a salesperson actually types into slots the tools can accept.

The reason this file exists: every design in this repository assumed a clean request.
A real one looks like "quote 3 hundred prod phon x 512 gb titanum, 20 pct off, bellows
collage". That sentence contains five things a naive parser gets wrong:

  1. "3 hundred" is a number, and so are "1.2k", "1,200" and "5OO" with a letter O.
  2. "prod phon x" is a misspelt identifier, and a misspelt identifier is a wrong
     product if you accept it silently.
  3. "512 gb" and "512" mean the same attribute value.
  4. "titanum" is a misspelling of a value that appears in a decision table.
  5. "bellows collage" is a customer name, and the correct behaviour depends on whether
     one account matches or several do.

Resolution is done in three tiers, cheapest first, and every tier records how it
resolved so the trace can show it.

   tier 1  exact or normalised form. Free, and right almost always.
   tier 2  fuzzy match above a threshold. Used for identifiers and names, never for
           numbers. The threshold is set so that a one or two character edit passes
           and a different product does not.
   tier 3  ambiguous. More than one candidate above threshold, or none. The value is
           not guessed: it is flagged, and the caller asks through an A2UI picker.

The dangerous error here is not a refusal, it is a confident wrong resolution: "PROD-
PHON-X" resolving to PROD-PHONE-X is fine, but resolving to PROD-HUB-Z would be a
wrong quote. Every test in this file is written with that asymmetry in mind.

Run:

    python build/input_normalizer.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable

# --------------------------------------------------------------------------------------
# The catalogue this tenant actually has. In a real system this is the product table.
# --------------------------------------------------------------------------------------

CATALOGUE = {
    "PROD-PHONE-X": "phone",
    "PROD-LAPTOP-Y": "laptop",
    "PROD-HUB-Z": "hub",
    "PROD-TABLET-W": "tablet",
}

# How people actually refer to a product. "40 laptops" is a complete request, and a
# parser that only accepts identifiers rejects most real traffic.
DESCRIPTIONS = {
    "PROD-PHONE-X": ["phone", "phones", "handset", "handsets", "mobile", "mobiles",
                     "iphone", "iphones", "smartphone", "smartphones", "device"],
    "PROD-LAPTOP-Y": ["laptop", "laptops", "notebook", "notebooks", "macbook"],
    "PROD-HUB-Z": ["hub", "hubs", "dock", "docks", "dongle", "adapter"],
    "PROD-TABLET-W": ["tablet", "tablets", "ipad", "ipads", "slate"],
}

ACCOUNTS = [
    "Bellows College Pvt",
    "Trey Research GmbH",
    "Northwind Traders",
    "Contoso Ltd",
    "Fabrikam Inc",
    "Tailspin Toys",
    "Consolidated Messenger Pvt",
    "Litware Inc",
    "Proseware Inc",
    "Wide World Importers",
]

ATTRIBUTE_VALUES = {
    "storage": ["128GB", "256GB", "512GB", "1TB"],
    "memory": ["16GB", "32GB", "64GB"],
    "colour": ["black", "white", "blue", "titanium"],
    "warranty": ["1yr", "3yr"],
}

ALIASES = {
    "storage": {"128": "128GB", "256": "256GB", "512": "512GB", "1t": "1TB", "1tb": "1TB",
                "1 tb": "1TB", "one tb": "1TB", "1024gb": "1TB", "1024": "1TB"},
    "memory": {"16": "16GB", "32": "32GB", "64": "64GB"},
    "colour": {"blk": "black", "blck": "black", "bla": "black", "wht": "white",
               "whte": "white", "blu": "blue", "titan": "titanium", "titanum": "titanium",
               "tit": "titanium", "grey": "titanium", "gray": "titanium"},
    "warranty": {"1": "1yr", "1y": "1yr", "1yr": "1yr", "1 year": "1yr", "one year": "1yr",
                 "3": "3yr", "3y": "3yr", "3yr": "3yr", "3 year": "3yr", "three year": "3yr",
                 "three years": "3yr", "3 years": "3yr"},
}

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000, "k": 1000, "million": 1_000_000}

FUZZY_SKU = 0.80      # a one or two char edit in a 12 char identifier
FUZZY_NAME = 0.82     # customer names carry more noise than identifiers
FUZZY_VALUE = 0.84    # attribute values are short, so the bar is higher


@dataclass
class Resolved:
    """A resolved value plus how it was resolved, so the trace can show the reasoning."""
    value: str | None
    tier: int                    # 1 exact, 2 fuzzy, 3 ambiguous
    matched: str | None = None
    score: float = 0.0
    candidates: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def needs_question(self) -> bool:
        return self.tier == 3


# --------------------------------------------------------------------------------------
# 1. Numbers
# --------------------------------------------------------------------------------------

OCR_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"})


# Values that consume a number and must not be read as a quantity: "512 gb" is storage,
# "20 %" is a discount, "3 year" is a warranty. Without this, a quantity search finds the
# first number in the sentence, which is usually an attribute.
PROTECTED = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|percent|pct|per cent|gb|g\b|tb|t\b|gig|gigs|yr|yrs|year|years|y\b)"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\s*(?:year|years|yr|yrs|gb|tb)\b",
    re.I)


def to_number(raw: str) -> float:
    """Read a numeric literal that may use either thousands convention.

    "1,200" and "1.200" both mean twelve hundred here, because a quantity of one point
    two units would not be written that way. A decimal with one or two fractional digits
    is a real decimal; three digits after the separator is a thousands group.
    """
    r = raw.strip()
    if re.fullmatch(r"\d{1,3}(?:[,.]\d{3})+", r):
        return float(re.sub(r"[,.]", "", r))
    if re.fullmatch(r"\d+[,.]\d{1,2}", r):
        return float(r.replace(",", "."))
    return float(r.replace(",", ""))


def _spans(pattern: str, text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.I)]


def _overlaps(a: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(not (a[1] <= b[0] or a[0] >= b[1]) for b in spans)


UNIT_WORDS = r"(?:units?|pcs|pieces?|ea|each|qty|quantity|nos|numbers?)"


def parse_quantity(text: str) -> tuple[float | None, str]:
    """Read a quantity out of messy text. Returns (value, how it was read).

    Handles digits, thousands separators, decimals, k/m suffixes, digit/letter
    confusions and number words combined with a scale word.
    """
    protected = _spans(PROTECTED.pattern, text)

    def usable(span):
        return not _overlaps(span, protected)

    # 1. a number next to a unit word wins outright: "300 units", "75 pcs", "40 ea"
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*" + UNIT_WORDS, text, re.I):
        if usable((m.start(), m.end())):
            return to_number(m.group(1)), m.group(0)

    # 2. "300 x PROD-..." and "x300"
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*x\s+[A-Za-z]", text, re.I):
        if usable((m.start(), m.end())):
            return to_number(m.group(1)), f"{m.group(1)}x"

    # 3. a multiplier: 1.2k, 3k, 0.5m
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(k|m|thousand|million)\b", text, re.I):
        if usable((m.start(), m.end())):
            return to_number(m.group(1)) * SCALES.get(m.group(2).lower(), 1), m.group(0)

    # 3b. digit/letter confusion before plain digits, so "5OO" is 500 and not 5.
    #     The lookarounds stop it matching inside an identifier like PROD-PHONE-X.
    for m in re.finditer(r"(?<![A-Za-z0-9\-,.])[0-9OlI|SB]{3,}(?![A-Za-z0-9\-])", text):
        fixed = m.group(0).translate(OCR_FIX)
        if fixed.isdigit() and usable((m.start(), m.end())):
            return float(fixed), f"ocr fix {m.group(0)}"

    # 3c. digits with a scale word: "3 hundred", "2 thousand"
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s+(hundred|thousand|million)\b", text, re.I):
        if usable((m.start(), m.end())):
            return to_number(m.group(1)) * SCALES[m.group(2).lower()], m.group(0)

    # 4. plain digits, skipping anything that is really an attribute value
    for m in re.finditer(r"(\d{1,3}(?:[,.]\d{3})+|\d+(?:\.\d+)?)", text):
        if not usable((m.start(), m.end())):
            continue
        # a digit glued to letters is part of an identifier, not a quantity
        before = text[m.start() - 1] if m.start() > 0 else " "
        after = text[m.end()] if m.end() < len(text) else " "
        if before.isalpha() and before not in "xX" or (after.isalpha() and after not in "xX"):
            continue
        try:
            return to_number(m.group(0)), m.group(0)
        except ValueError:
            continue

    # 5. number words, with an optional scale: "three hundred"
    total, current, seen = 0, 0, False
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in WORD_NUMBERS:
            current += WORD_NUMBERS[w]
            seen = True
        elif w in SCALES:
            current = max(1, current) * SCALES[w]
            total += current
            current = 0
            seen = True
        elif seen:
            break
    if seen:
        return float(total + current), "word number"

    # 6. digit/letter confusion: "5OO"
    for m in re.finditer(r"(?<![A-Za-z])[0-9OlI|SB]{2,}(?![A-Za-z])", text):
        fixed = m.group(0).translate(OCR_FIX)
        if fixed.isdigit() and usable((m.start(), m.end())):
            return float(fixed), "ocr fix"

    return None, "not found"


def parse_percent(text: str) -> tuple[float | None, str]:
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct|per cent)", t)
    if m:
        return float(m.group(1)), m.group(0)
    # number words then a percent marker: "twenty five percent"
    m = re.search(r"((?:\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                  r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
                  r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)[\s-]*)+?)"
                  r"\s*(?:%|percent|pct|per cent|off|discount)", t)
    if m:
        words = re.findall(r"[a-z]+", m.group(1))
        n, cur, total = 0, 0, 0
        for w in words:
            if w in WORD_NUMBERS:
                cur += WORD_NUMBERS[w]
            elif w in SCALES:
                cur = max(1, cur) * SCALES[w]
                total += cur
                cur = 0
        val = total + cur
        if val:
            return float(val), m.group(0)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:off|discount)", t)
    if m:
        return float(m.group(1)), m.group(0)
    return None, "not found"


# --------------------------------------------------------------------------------------
# 2. Identifiers and names, fuzzy but conservative
# --------------------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def resolve_sku(text: str) -> Resolved:
    """Find the product identifier in the text.

    Exact first, then a normalised comparison that ignores hyphens and case, then a
    conservative fuzzy pass. A near miss on an identifier is a wrong product, so the
    fuzzy threshold is high and ties are reported rather than broken.
    """
    canon = {_norm(k): k for k in CATALOGUE}
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{4,}", text):
        n = _norm(token)
        if n in canon:
            return Resolved(canon[n], 1, canon[n], 1.0)

    # fuzzy over the whole string and over consecutive token windows
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", text)
    best: list[tuple[float, str]] = []
    for key in CATALOGUE:
        best.append((_ratio(text, key), key))
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + 4, len(tokens) + 1)):
                best.append((_ratio(" ".join(tokens[i:j]), key), key))
    per_key: dict[str, float] = {}
    for score, key in best:
        if score > per_key.get(key, -1.0):
            per_key[key] = score
    ranked = sorted(per_key.items(), key=lambda kv: -kv[1])
    top_key, top_score = ranked[0][0], ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score >= FUZZY_SKU and top_score - second >= 0.02:
        return Resolved(top_key, 2, top_key, round(top_score, 4))
    cands = [k for k, sc in ranked[:3] if sc >= FUZZY_SKU - 0.10]
    if len({c for c in cands}) > 1:
        return Resolved(None, 3, None, round(top_score, 4), sorted(set(cands)))

    # no identifier in the text: fall back to what people call the thing
    low = " " + text.lower() + " "
    hits = [sku for sku, words in DESCRIPTIONS.items()
            if any(re.search(r"\b" + w + r"\b", low) for w in words)]
    if len(hits) == 1:
        return Resolved(hits[0], 2, "description", 1.0)
    if len(hits) > 1:
        return Resolved(None, 3, None, round(top_score, 4), hits)

    return Resolved(None, 3, None, round(top_score, 4), [])


def _strip_company_suffix(s: str) -> str:
    return re.sub(r"\b(pvt|private|ltd|limited|inc|gmbh|llc|corp|co)\b\.?", "", s, flags=re.I).strip()


def resolve_account(text: str) -> Resolved:
    """Find the customer. Suffixes like Pvt and Ltd are ignored; ties are not broken."""
    for a in ACCOUNTS:
        if _norm(a) in _norm(text):
            return Resolved(a, 1, a, 1.0)
    for a in ACCOUNTS:
        if _norm(_strip_company_suffix(a)) in _norm(text):
            return Resolved(a, 1, a, 1.0)

    # A distinctive single token is often all a person types: "northwind", "contoso".
    # Only accepted when exactly one account owns the token, so it cannot be a guess.
    text_tokens = {_norm(t) for t in re.findall(r"[A-Za-z]{4,}", text)}
    token_hits: list[str] = []
    for a in ACCOUNTS:
        head = _strip_company_suffix(a).split()[0]
        if _norm(head) in text_tokens and len(_norm(head)) >= 5:
            token_hits.append(a)
    if len(token_hits) == 1:
        return Resolved(token_hits[0], 2, "distinctive token", 1.0)

    # Compare the request's words against each account's own words, not the whole
    # request against the whole legal name. A whole-string comparison dilutes the score
    # with the corporate suffix and the rest of the sentence: "lytware" scored 0.706
    # against "Litware Inc" and fell under the threshold, even though it is one edit from
    # the only distinctive word in that name.
    req_tokens = [_norm(t) for t in re.findall(r"[A-Za-z]{4,}", text)]
    per_account: list[tuple[float, str]] = []
    for a in ACCOUNTS:
        acc_tokens = [_norm(t) for t in
                      re.findall(r"[A-Za-z]{4,}", _strip_company_suffix(a))]
        score = max((_ratio(rt, at) for rt in req_tokens for at in acc_tokens), default=0.0)
        per_account.append((round(score, 4), a))
    scored = sorted(per_account, reverse=True)
    top_score, top_name = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if top_score >= FUZZY_NAME and top_score - second >= 0.03:
        return Resolved(top_name, 2, top_name, round(top_score, 4))
    cands = [a for s, a in scored[:3] if s >= FUZZY_NAME - 0.08]
    if len(cands) > 1:
        return Resolved(None, 3, None, round(top_score, 4), cands)
    return Resolved(None, 3, None, round(top_score, 4), [])


def resolve_attribute(kind: str, text: str) -> Resolved:
    """Read one attribute value out of the text.

    Aliases are tried before fuzzy, because "512" and "titan" are not spelling mistakes,
    they are different names for the same value, and an edit-distance comparison between
    "512" and "512GB" scores worse than between "512" and "128".
    """
    t = text.lower()
    values = ATTRIBUTE_VALUES[kind]

    for v in values:
        if re.search(r"\b" + re.escape(v.lower()) + r"\b", t):
            return Resolved(v, 1, v, 1.0)

    for alias, target in sorted(ALIASES.get(kind, {}).items(), key=lambda kv: -len(kv[0])):
        # a bare number is only an alias when it stands alone, so "512 phones" is a
        # quantity and not a storage tier
        if alias.isdigit():
            if re.search(r"\b" + alias + r"\s*(?:gb|g|gig|gigs)\b", t) or re.search(
                    r"\b" + re.escape(kind) + r"\s*[:=]?\s*" + alias + r"\b", t):
                return Resolved(target, 1, alias, 1.0)
        elif re.search(r"\b" + re.escape(alias) + r"\b", t):
            return Resolved(target, 1, alias, 1.0)

    scored = sorted(((_ratio(t, v), v) for v in values), reverse=True)
    top_score, top_val = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if top_score >= FUZZY_VALUE and top_score - second >= 0.05:
        return Resolved(top_val, 2, top_val, round(top_score, 4))

    # fuzzy per token, so "titanum" is matched against "titanium" rather than the sentence
    for tok in re.findall(r"[a-z]{3,}", t):
        cand = sorted(((_ratio(tok, v), v) for v in values), reverse=True)
        if cand and cand[0][0] >= FUZZY_VALUE and (len(cand) < 2 or cand[0][0] - cand[1][0] >= 0.05):
            return Resolved(cand[0][1], 2, tok, round(cand[0][0], 4))

    return Resolved(None, 3, None, round(top_score, 4), [])


# --------------------------------------------------------------------------------------
# 3. The whole request
# --------------------------------------------------------------------------------------

@dataclass
class Slots:
    sku: Resolved
    account: Resolved
    quantity: float | None
    quantity_how: str
    percent: float | None
    percent_how: str
    attributes: dict[str, Resolved]
    questions: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.questions

    def to_dict(self) -> dict:
        return {
            "sku": self.sku.value, "sku_tier": self.sku.tier,
            "account": self.account.value, "account_tier": self.account.tier,
            "quantity": self.quantity, "quantity_how": self.quantity_how,
            "percent": self.percent,
            "attributes": {k: v.value for k, v in self.attributes.items()},
            "attribute_tiers": {k: v.tier for k, v in self.attributes.items()},
            "needs": self.questions,
        }


ATTRIBUTE_FOR_KIND = {
    "PROD-PHONE-X": ("storage", "colour"),
    "PROD-LAPTOP-Y": ("memory", "warranty"),
    "PROD-HUB-Z": (),
    "PROD-TABLET-W": ("storage", "colour"),
}


def parse(text: str) -> Slots:
    """Parse a free-form request into slots, listing what still needs asking."""
    sku = resolve_sku(text)
    account = resolve_account(text)
    qty, qty_how = parse_quantity(text)
    pct, pct_how = parse_percent(text)

    attrs: dict[str, Resolved] = {}
    needs = ["storage", "colour"] if sku.value is None else list(ATTRIBUTE_FOR_KIND.get(sku.value, []))
    for kind in needs:
        attrs[kind] = resolve_attribute(kind, text)

    questions: list[str] = []
    if sku.needs_question:
        questions.append("product")
    if account.needs_question:
        questions.append("account")
    if qty is None:
        questions.append("quantity")
    for kind, r in attrs.items():
        if r.needs_question:
            questions.append(kind)

    return Slots(sku, account, qty, qty_how, pct, pct_how, attrs, questions)


# --------------------------------------------------------------------------------------
# 4. Tests, written around the asymmetry: a wrong resolution is worse than a question
# --------------------------------------------------------------------------------------

MESSY: list[tuple[str, dict]] = [
    # (request, expected slots that must be right)
    ("quote 3 hundred prod phon x 512 gb titanum, 20 pct off, bellows collage",
     {"sku": "PROD-PHONE-X", "account": "Bellows College Pvt", "quantity": 300,
      "percent": 20, "storage": "512GB", "colour": "titanium"}),
    ("PROD PHONE X 1.2k units for northwind, 3yr warrenty 32 gig",
     {"sku": "PROD-PHONE-X", "account": "Northwind Traders", "quantity": 1200}),
    ("need a qoute for 40 laptops trey research 32g 3 year",
     {"sku": "PROD-LAPTOP-Y", "account": "Trey Research GmbH", "quantity": 40,
      "memory": "32GB", "warranty": "3yr"}),
    ("500 prod-hub-z contoso, twenty five percent off",
     {"sku": "PROD-HUB-Z", "account": "Contoso Ltd", "quantity": 500, "percent": 25}),
    ("quote 5OO PROD-HUB-Z for fabrikam, blk",
     {"sku": "PROD-HUB-Z", "account": "Fabrikam Inc", "quantity": 500}),
    ("1,200 phones bellows college, 1tb, titan",
     {"sku": "PROD-PHONE-X", "account": "Bellows College Pvt", "quantity": 1200,
      "storage": "1TB", "colour": "titanium"}),
    ("prod tablet w please, 256gb, white, 75 units, tailspin toys",
     {"sku": "PROD-TABLET-W", "account": "Tailspin Toys", "quantity": 75,
      "storage": "256GB", "colour": "white"}),
    ("3k prod laptop y for wide world importers",
     {"sku": "PROD-LAPTOP-Y", "account": "Wide World Importers", "quantity": 3000}),
    ("40 tablets for lytware, 128 gig, blu",
     {"sku": "PROD-TABLET-W", "account": "Litware Inc", "quantity": 40,
      "storage": "128GB", "colour": "blue"}),
    ("quote 250 prod tablit w 1 tb titanum for consolodated messenger",
     {"sku": "PROD-TABLET-W", "account": "Consolidated Messenger Pvt", "quantity": 250,
      "storage": "1TB", "colour": "titanium"}),
]


# Requests that must NOT be resolved silently. The correct output is a question.
MUST_ASK: list[tuple[str, str]] = [
    ("quote 40 units for bellows college", "no product mentioned at all"),
    ("quote 500 hub z for acme rocket corp", "customer does not exist"),
    ("quote some phones for contoso", "no quantity"),
    ("quote prod phone x for contoso", "no quantity, and no size"),
]


def trace(text: str) -> dict:
    """Everything the UI needs to show one request being resolved, slot by slot."""
    sl = parse(text)

    def one(r: Resolved) -> dict:
        return {"value": r.value, "tier": r.tier, "matched": r.matched,
                "score": r.score, "candidates": r.candidates}

    return {
        "text": text,
        "sku": one(sl.sku),
        "account": one(sl.account),
        "quantity": sl.quantity,
        "quantity_how": sl.quantity_how,
        "percent": sl.percent,
        "percent_how": sl.percent_how,
        "attributes": {k: one(v) for k, v in sl.attributes.items()},
        "needs": sl.questions,
        "ready": sl.ready,
    }


def _dedupe(texts: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


TRACE_SET = _dedupe([t for t, _ in MESSY] + [t for t, _ in MUST_ASK])


def _dump_json() -> int:
    import json
    print(json.dumps([trace(t) for t in TRACE_SET], indent=1))
    return 0


def _selftest() -> int:
    checks: list[tuple[str, bool]] = []

    def ok(name, fn):
        try:
            checks.append((name, bool(fn())))
        except Exception as e:                        # noqa: BLE001
            print("ERR ", name, e)
            checks.append((name, False))

    ok("plain digits", lambda: parse_quantity("300 units")[0] == 300)
    ok("thousands separator", lambda: parse_quantity("1,200 units")[0] == 1200)
    ok("decimal thousands", lambda: parse_quantity("1.200 units")[0] == 1200)
    ok("k suffix", lambda: parse_quantity("1.2k units")[0] == 1200)
    ok("word number", lambda: parse_quantity("three units")[0] == 3)
    ok("word number with scale", lambda: parse_quantity("three hundred")[0] == 300)
    ok("compound word number", lambda: parse_quantity("twenty five")[0] == 25)
    ok("letter O for zero", lambda: parse_quantity("5OO")[0] == 500)
    ok("no number is None", lambda: parse_quantity("some units")[0] is None)

    ok("percent sign", lambda: parse_percent("20% off")[0] == 20)
    ok("percent word", lambda: parse_percent("twenty percent off")[0] == 20)
    ok("pct abbreviation", lambda: parse_percent("20 pct")[0] == 20)

    ok("exact sku", lambda: resolve_sku("PROD-PHONE-X").value == "PROD-PHONE-X")
    ok("case and space sku", lambda: resolve_sku("prod phone x").value == "PROD-PHONE-X")
    ok("typo sku", lambda: resolve_sku("prod phon x").value == "PROD-PHONE-X")
    ok("missing hyphen sku", lambda: resolve_sku("prodphonx").value == "PROD-PHONE-X")
    ok("typo in an account name still resolves",
       lambda: resolve_account("40 tablets for lytware, 128 gig").value == "Litware Inc")
    ok("typo in a two word account still resolves",
       lambda: resolve_account("for consolodated messenger").value
       == "Consolidated Messenger Pvt")
    ok("a customer that does not exist is not guessed",
       lambda: resolve_account("for acme rocket corp").tier == 3)
    ok("a wrong sku is not guessed silently",
       lambda: resolve_sku("prod washing machine").tier == 3)
    ok("the wrong sku resolves to nothing",
       lambda: resolve_sku("prod washing machine").value is None)

    ok("exact account", lambda: resolve_account("Bellows College Pvt").value == "Bellows College Pvt")
    ok("typo account", lambda: resolve_account("bellows collage").value == "Bellows College Pvt")
    ok("suffix dropped", lambda: resolve_account("trey research").value == "Trey Research GmbH")
    ok("unknown account asks", lambda: resolve_account("acme rocket corp").needs_question)

    ok("storage bare number with gb",
       lambda: resolve_attribute("storage", "512 gb").value == "512GB")
    ok("storage alias 1tb", lambda: resolve_attribute("storage", "1tb").value == "1TB")
    ok("colour typo", lambda: resolve_attribute("colour", "titanum").value == "titanium")
    ok("colour abbreviation", lambda: resolve_attribute("colour", "blk").value == "black")
    ok("warranty words", lambda: resolve_attribute("warranty", "three years").value == "3yr")
    ok("a bare 512 does not become storage when it is a quantity",
       lambda: resolve_attribute("storage", "512 units of phone").value is None)

    ok("full messy parse",
       lambda: parse("quote 3 hundred prod phon x 512 gb titanum, 20 pct off, bellows collage").sku.value
       == "PROD-PHONE-X")
    ok("parse lists what is missing",
       lambda: "quantity" in parse("quote prod phone x for bellows college").questions)

    bad = [n for n, p in checks if not p]
    for n, p in checks:
        print(("PASS  " if p else "FAIL  ") + n)
    print(f"\n{len(checks) - len(bad)}/{len(checks)} passed")

    print()
    print("messy request corpus:")
    silent_wrong = 0
    right = 0
    asked = 0
    for text, want in MESSY:
        slots = parse(text)
        got = {
            "sku": slots.sku.value, "account": slots.account.value,
            "quantity": slots.quantity, "percent": slots.percent,
        }
        for kind, r in slots.attributes.items():
            got[kind] = r.value
        misses = []
        for k, v in want.items():
            g = got.get(k)
            if g == v:
                continue
            if g is None:
                asked += 1
                misses.append(f"{k}=asked")
            else:
                silent_wrong += 1
                misses.append(f"{k}={g!r} wanted {v!r}")
        if not misses:
            right += 1
            print(f"  ok    {text[:62]}")
        else:
            print(f"  note  {text[:62]}")
            print(f"        {', '.join(misses)}")
    print(f"\n  {right}/{len(MESSY)} requests fully resolved")
    print(f"  {asked} slots deferred to a question")
    print(f"  {silent_wrong} slots resolved to the WRONG value, which is the error that matters")

    print()
    print("requests that must be questioned, not guessed:")
    leaked = 0
    for text, why in MUST_ASK:
        slots = parse(text)
        if slots.ready:
            leaked += 1
            print(f"  LEAK  resolved without asking: {text[:54]}  ({why})")
        else:
            print(f"  ok    asks for {','.join(slots.questions):28s} {text[:40]}")
    print(f"  {leaked}/{len(MUST_ASK)} resolved silently when they should have asked")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    if "--json" in sys.argv:
        raise SystemExit(_dump_json())
    raise SystemExit(_selftest())
