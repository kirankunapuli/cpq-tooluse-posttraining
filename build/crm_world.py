"""Mock CRM world + tool definitions + goal predicates.

Deterministic, stdlib-only. This is the sandbox the data pipeline verifies against:
a candidate trace only survives if replaying it against this world reaches the goal
state. That is the difference between a *plausible* trace and a *correct* one.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable

STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]

# Billing-ops constants. DISCOUNT_THRESHOLD is the whole point of the CPQ family:
# the model must know when a discount crosses the line that requires approval, and it
# must not activate a quote that needs an approval it has not requested.
PRICEBOOKS = ["USD-Standard", "EUR-Standard", "APAC-Partner"]
INVOICE_STATUSES = ["Draft", "Posted", "Paid", "Disputed"]
SUB_STATUSES = ["Active", "Suspended", "Cancelled"]
DISCOUNT_THRESHOLD = 20.0  # percent
PRODUCT_FAMILIES = ["Billing Ops", "Sales CRM", "Customer Support", "Data Platform"]
PRODUCT_NAMES = [
    ("SUB-BILL", "Subscription Billing", "Billing Ops"),
    ("CPQ-PLATFORM", "CPQ Platform", "Billing Ops"),
    ("BILL-AUTO", "Billing Automation", "Billing Ops"),
    ("CRM-SALES", "Sales CRM Enterprise", "Sales CRM"),
    ("CRM-AI", "AI Sales Assistant", "Sales CRM"),
    ("SVC-SUPPORT", "Support Desk Enterprise", "Customer Support"),
    ("DATA-INGEST", "Data Platform Ingestion", "Data Platform"),
    ("DATA-ZERO", "Data Platform Zero Copy", "Data Platform"),
]

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_accounts",
        "description": "Find accounts whose name contains the query (case-insensitive).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "list_opportunities",
        "description": "List open opportunities for an account, optionally filtered by minimum amount and stage.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "min_amount": {"type": "number"},
                "stage": {"type": "string", "enum": STAGES},
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "update_opportunity_stage",
        "description": "Move one opportunity to a new stage.",
        "parameters": {
            "type": "object",
            "properties": {
                "opportunity_id": {"type": "string"},
                "stage": {"type": "string", "enum": STAGES},
            },
            "required": ["opportunity_id", "stage"],
        },
    },
    {
        "name": "list_cases",
        "description": "List support cases for an account, optionally by status.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "status": {"type": "string", "enum": ["Open", "Escalated", "Closed"]},
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "escalate_case",
        "description": "Raise the priority of a support case.",
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "priority": {"type": "string", "enum": PRIORITIES},
            },
            "required": ["case_id", "priority"],
        },
    },
    {
        "name": "create_task",
        "description": "Create a follow-up task on an account.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "subject": {"type": "string"},
                "priority": {"type": "string", "enum": PRIORITIES},
            },
            "required": ["account_id", "subject"],
        },
    },
    # ---- billing ops: CPQ + subscription + billing ----
    {
        "name": "list_products",
        "description": "List products in the catalog, optionally filtered by family.",
        "parameters": {
            "type": "object",
            "properties": {
                "family": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "list_pricebook_entries",
        "description": "List price book entries (unit price per product) for a price book.",
        "parameters": {
            "type": "object",
            "properties": {
                "pricebook": {"type": "string", "enum": PRICEBOOKS},
                "product_id": {"type": "string"},
            },
            "required": ["pricebook"],
        },
    },
    {
        "name": "create_quote",
        "description": "Create a draft quote for an opportunity.",
        "parameters": {
            "type": "object",
            "properties": {
                "opportunity_id": {"type": "string"},
                "pricebook": {"type": "string", "enum": PRICEBOOKS},
            },
            "required": ["opportunity_id", "pricebook"],
        },
    },
    {
        "name": "add_quote_line_item",
        "description": "Add a product to a draft quote at a discount percentage.",
        "parameters": {
            "type": "object",
            "properties": {
                "quote_id": {"type": "string"},
                "pricebook_entry_id": {"type": "string"},
                "quantity": {"type": "integer"},
                "discount_pct": {"type": "number"},
            },
            "required": ["quote_id", "pricebook_entry_id", "quantity"],
        },
    },
    {
        "name": "request_quote_approval",
        "description": "Route a quote for manager approval. Required when any line discount exceeds the approval threshold.",
        "parameters": {
            "type": "object",
            "properties": {
                "quote_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["quote_id", "reason"],
        },
    },
    {
        "name": "activate_quote",
        "description": "Activate a quote so it can be ordered. Refuses if the quote needs unapproved discounts.",
        "parameters": {
            "type": "object",
            "properties": {"quote_id": {"type": "string"}},
            "required": ["quote_id"],
        },
    },
    {
        "name": "list_subscriptions",
        "description": "List subscriptions for an account.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "status": {"type": "string", "enum": SUB_STATUSES},
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "amend_subscription_quantity",
        "description": "Change the quantity on an active subscription.",
        "parameters": {
            "type": "object",
            "properties": {
                "subscription_id": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["subscription_id", "quantity"],
        },
    },
    {
        "name": "list_invoices",
        "description": "List invoices for an account, optionally by status.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "status": {"type": "string", "enum": INVOICE_STATUSES},
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "dispute_invoice",
        "description": "Open a dispute on an invoice with a stated reason.",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["invoice_id", "reason"],
        },
    },
]


@dataclass
class World:
    accounts: dict[str, dict] = field(default_factory=dict)
    opportunities: dict[str, dict] = field(default_factory=dict)
    cases: dict[str, dict] = field(default_factory=dict)
    tasks: list[dict] = field(default_factory=list)
    # Billing-ops primitives (CPQ + billing + subscription lifecycle)
    products: dict[str, dict] = field(default_factory=dict)
    pricebook_entries: dict[str, dict] = field(default_factory=dict)
    quotes: dict[str, dict] = field(default_factory=dict)
    subscriptions: dict[str, dict] = field(default_factory=dict)
    invoices: dict[str, dict] = field(default_factory=dict)
    approvals: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)

    def clone(self) -> "World":
        w = World(
            accounts=copy.deepcopy(self.accounts),
            opportunities=copy.deepcopy(self.opportunities),
            cases=copy.deepcopy(self.cases),
            tasks=copy.deepcopy(self.tasks),
            products=copy.deepcopy(self.products),
            pricebook_entries=copy.deepcopy(self.pricebook_entries),
            quotes=copy.deepcopy(self.quotes),
            subscriptions=copy.deepcopy(self.subscriptions),
            invoices=copy.deepcopy(self.invoices),
            approvals=copy.deepcopy(self.approvals),
        )
        return w


COMPANIES = [
    "Northwind", "Contoso", "Fabrikam", "Tailspin", "Adventure Works", "Litware",
    "Proseware", "Wingtip", "Lucerne", "Alpine", "Blue Yonder", "Coho", "Woodgrove",
    "Trey Research", "Fourth Coffee", "Margie Travel", "Southridge", "Humongous",
    "Consolidated Messenger", "Nod Publishers", "Graphic Design Institute", "Wide World",
    "Relecloud", "Trey", "VanArsdel", "Bellows College", "City Power", "Lakeshore",
]
REGIONS = ["EMEA", "APAC", "NA", "LATAM"]
OWNERS = ["a.okafor", "s.meyer", "r.iyer", "j.laurent", "m.chen", "p.vasquez"]
SUBJECTS = [
    "Send renewal proposal", "Schedule QBR", "Follow up on pricing", "Prepare security review",
    "Draft migration plan", "Confirm procurement timeline", "Share reference architecture",
]


def build_world(seed: int = 7, n_accounts: int = 60) -> World:
    rng = random.Random(seed)
    w = World()

    # ---- product catalog + price books (the CPQ foundation)
    for code, name, fam in PRODUCT_NAMES:
        w.products[f"PROD-{code}"] = {"id": f"PROD-{code}", "name": name, "family": fam}
    base_prices = {
        "SUB-BILL": 24_000, "CPQ-PLATFORM": 36_000, "BILL-AUTO": 18_000,
        "CRM-SALES": 1_500, "CRM-AI": 900, "SVC-SUPPORT": 1_200,
        "DATA-INGEST": 30_000, "DATA-ZERO": 12_000,
    }
    currency_mult = {"USD-Standard": 1.0, "EUR-Standard": 0.92, "APAC-Partner": 0.78}
    for code, _, _ in PRODUCT_NAMES:
        for pb, mult in currency_mult.items():
            pbe_id = f"PBE-{pb.split('-')[0]}-{code}"
            w.pricebook_entries[pbe_id] = {
                "id": pbe_id,
                "product_id": f"PROD-{code}",
                "pricebook": pb,
                "unit_price": round(base_prices[code] * mult, 2),
            }

    for i in range(n_accounts):
        aid = f"ACC-{1000 + i}"
        name = f"{rng.choice(COMPANIES)} {rng.choice(['Ltd', 'Inc', 'GmbH', 'Pvt', 'SA'])}"
        w.accounts[aid] = {
            "id": aid,
            "name": name,
            "owner": rng.choice(OWNERS),
            "region": rng.choice(REGIONS),
            "tier": rng.choice(["Enterprise", "Mid-Market", "SMB"]),
        }
        for j in range(rng.randint(1, 4)):
            oid = f"OPP-{aid[4:]}-{j}"
            w.opportunities[oid] = {
                "id": oid,
                "account_id": aid,
                "amount": rng.choice([12_000, 45_000, 90_000, 150_000, 320_000, 750_000]),
                "stage": rng.choice(STAGES[:4]),
                "open": True,
            }
        for j in range(rng.randint(0, 3)):
            cid = f"CASE-{aid[4:]}-{j}"
            w.cases[cid] = {
                "id": cid,
                "account_id": aid,
                "status": rng.choice(["Open", "Open", "Escalated", "Closed"]),
                "priority": rng.choice(PRIORITIES),
                "subject": rng.choice(["Login failure", "Billing mismatch", "API latency",
                                       "Data export error", "SSO misconfiguration"]),
            }
        # subscriptions + invoices (the billing side)
        for j in range(rng.randint(1, 3)):
            sid = f"SUB-{aid[4:]}-{j}"
            code = rng.choice([c for c, _, _ in PRODUCT_NAMES])
            qty = rng.choice([10, 25, 50, 100, 250])
            w.subscriptions[sid] = {
                "id": sid,
                "account_id": aid,
                "product_id": f"PROD-{code}",
                "quantity": qty,
                "status": rng.choice(["Active", "Active", "Active", "Suspended"]),
            }
        for j in range(rng.randint(0, 2)):
            iid = f"INV-{aid[4:]}-{j}"
            w.invoices[iid] = {
                "id": iid,
                "account_id": aid,
                "amount": rng.choice([4_500, 12_000, 24_000, 48_000]),
                "status": rng.choice(["Posted", "Posted", "Paid", "Draft"]),
                "period": rng.choice(["2026-09", "2026-10", "2026-11"]),
            }
    return w


# ---------------------------------------------------------------- tool execution


class ToolError(Exception):
    pass


def execute(world: World, call: dict) -> Any:
    """Execute one tool call, mutating `world`. Raises ToolError on invalid calls."""
    name = call.get("name")
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        raise ToolError("arguments must be an object")

    if name == "search_accounts":
        q = str(args.get("query", "")).lower()
        if not q:
            raise ToolError("query required")
        hits = [a for a in world.accounts.values() if q in a["name"].lower()]
        return [{"account_id": a["id"], "name": a["name"]} for a in hits][:10]

    if name == "list_opportunities":
        aid = args.get("account_id")
        if aid not in world.accounts:
            raise ToolError(f"unknown account_id {aid}")
        out = []
        for o in world.opportunities.values():
            if o["account_id"] != aid or not o["open"]:
                continue
            if "min_amount" in args and o["amount"] < args["min_amount"]:
                continue
            if "stage" in args and o["stage"] != args["stage"]:
                continue
            out.append(o)
        return [{"opportunity_id": o["id"], "amount": o["amount"], "stage": o["stage"]} for o in out]

    if name == "update_opportunity_stage":
        oid, stage = args.get("opportunity_id"), args.get("stage")
        if oid not in world.opportunities:
            raise ToolError(f"unknown opportunity_id {oid}")
        if stage not in STAGES:
            raise ToolError(f"invalid stage {stage}")
        world.opportunities[oid]["stage"] = stage
        return {"ok": True}

    if name == "list_cases":
        aid = args.get("account_id")
        if aid not in world.accounts:
            raise ToolError(f"unknown account_id {aid}")
        out = [
            c for c in world.cases.values()
            if c["account_id"] == aid and ("status" not in args or c["status"] == args["status"])
        ]
        return [{"case_id": c["id"], "priority": c["priority"], "subject": c["subject"]} for c in out]

    if name == "escalate_case":
        cid, prio = args.get("case_id"), args.get("priority")
        if cid not in world.cases:
            raise ToolError(f"unknown case_id {cid}")
        if prio not in PRIORITIES:
            raise ToolError(f"invalid priority {prio}")
        world.cases[cid]["priority"] = prio
        if prio in ("High", "Critical"):
            world.cases[cid]["status"] = "Escalated"
        return {"ok": True}

    if name == "create_task":
        aid = args.get("account_id")
        if aid not in world.accounts:
            raise ToolError(f"unknown account_id {aid}")
        subj = str(args.get("subject") or "").strip()
        if not subj:
            raise ToolError("subject required")
        world.tasks.append({"account_id": aid, "subject": subj, "priority": args.get("priority", "Medium")})
        return {"ok": True, "task_index": len(world.tasks) - 1}

    # ---- billing-ops tools
    if name == "list_products":
        fam = args.get("family")
        return [p for p in world.products.values() if fam is None or p["family"] == fam]

    if name == "list_pricebook_entries":
        pb = args.get("pricebook")
        if pb not in PRICEBOOKS:
            raise ToolError(f"invalid pricebook {pb}")
        out = [e for e in world.pricebook_entries.values() if e["pricebook"] == pb]
        if args.get("product_id"):
            out = [e for e in out if e["product_id"] == args["product_id"]]
        return [{"pricebook_entry_id": e["id"], "product_id": e["product_id"],
                 "unit_price": e["unit_price"]} for e in out]

    if name == "create_quote":
        oid = args.get("opportunity_id")
        if oid not in world.opportunities:
            raise ToolError(f"unknown opportunity_id {oid}")
        pb = args.get("pricebook")
        if pb not in PRICEBOOKS:
            raise ToolError(f"invalid pricebook {pb}")
        qid = f"QUO-{oid[4:]}"
        world.quotes[qid] = {
            "id": qid,
            "opportunity_id": oid,
            "account_id": world.opportunities[oid]["account_id"],
            "pricebook": pb,
            "status": "Draft",
            "line_items": [],
            "total": 0.0,
            "approval_requested": False,
            "approved": False,
        }
        return {"quote_id": qid}

    if name == "add_quote_line_item":
        qid = args.get("quote_id")
        if qid not in world.quotes:
            raise ToolError(f"unknown quote_id {qid}")
        q = world.quotes[qid]
        if q["status"] != "Draft":
            raise ToolError("quote is not in Draft status")
        pbe_id = args.get("pricebook_entry_id")
        if pbe_id not in world.pricebook_entries:
            raise ToolError(f"unknown pricebook_entry_id {pbe_id}")
        pbe = world.pricebook_entries[pbe_id]
        # A quote line must come from the quote's own price book: mixing currency
        # books is a real CPQ error, and it is silently wrong rather than loud.
        if pbe["pricebook"] != q["pricebook"]:
            raise ToolError(
                f"pricebook mismatch: entry {pbe_id} is {pbe['pricebook']}, "
                f"quote is {q['pricebook']}")
        qty = args.get("quantity")
        if not isinstance(qty, int) or qty <= 0:
            raise ToolError("quantity must be a positive integer")
        disc = float(args.get("discount_pct", 0) or 0)
        if not (0 <= disc <= 100):
            raise ToolError("discount_pct out of range")
        net = round(pbe["unit_price"] * qty * (1 - disc / 100.0), 2)
        q["line_items"].append({
            "pricebook_entry_id": pbe_id,
            "product_id": pbe["product_id"],
            "quantity": qty,
            "discount_pct": disc,
            "net": net,
        })
        q["total"] = round(sum(li["net"] for li in q["line_items"]), 2)
        return {"quote_id": qid, "line_total": net, "quote_total": q["total"]}

    if name == "request_quote_approval":
        qid = args.get("quote_id")
        if qid not in world.quotes:
            raise ToolError(f"unknown quote_id {qid}")
        if not str(args.get("reason") or "").strip():
            raise ToolError("reason required")
        world.quotes[qid]["approval_requested"] = True
        world.approvals.append({"quote_id": qid, "reason": args["reason"],
                                "state": "pending"})
        return {"ok": True}

    if name == "activate_quote":
        qid = args.get("quote_id")
        if qid not in world.quotes:
            raise ToolError(f"unknown quote_id {qid}")
        q = world.quotes[qid]
        if not q["line_items"]:
            raise ToolError("cannot activate a quote with no line items")
        needs_approval = any(li["discount_pct"] > DISCOUNT_THRESHOLD
                             for li in q["line_items"])
        if needs_approval and not q["approval_requested"]:
            raise ToolError("quote needs approval: discount exceeds threshold")
        q["status"] = "Activated"
        return {"ok": True, "total": q["total"]}

    if name == "list_subscriptions":
        aid = args.get("account_id")
        if aid not in world.accounts:
            raise ToolError(f"unknown account_id {aid}")
        out = [s for s in world.subscriptions.values() if s["account_id"] == aid]
        if args.get("status"):
            out = [s for s in out if s["status"] == args["status"]]
        return [{"subscription_id": s["id"], "product_id": s["product_id"],
                 "quantity": s["quantity"], "status": s["status"]} for s in out]

    if name == "amend_subscription_quantity":
        sid = args.get("subscription_id")
        if sid not in world.subscriptions:
            raise ToolError(f"unknown subscription_id {sid}")
        s = world.subscriptions[sid]
        if s["status"] != "Active":
            raise ToolError("only Active subscriptions can be amended")
        qty = args.get("quantity")
        if not isinstance(qty, int) or qty <= 0:
            raise ToolError("quantity must be a positive integer")
        s["quantity"] = qty
        return {"ok": True, "quantity": qty}

    if name == "list_invoices":
        aid = args.get("account_id")
        if aid not in world.accounts:
            raise ToolError(f"unknown account_id {aid}")
        out = [i for i in world.invoices.values() if i["account_id"] == aid]
        if args.get("status"):
            out = [i for i in out if i["status"] == args["status"]]
        return [{"invoice_id": i["id"], "amount": i["amount"],
                 "status": i["status"], "period": i["period"]} for i in out]

    if name == "dispute_invoice":
        iid = args.get("invoice_id")
        if iid not in world.invoices:
            raise ToolError(f"unknown invoice_id {iid}")
        if not str(args.get("reason") or "").strip():
            raise ToolError("reason required")
        world.invoices[iid]["status"] = "Disputed"
        return {"ok": True}

    raise ToolError(f"unknown tool {name!r}")


# ---------------------------------------------------------------- tasks & goals


@dataclass
class Effects:
    """Exactly what the world may look like after a correct trajectory.

    Checking goal satisfaction alone is not enough: a trace that reaches the goal *and*
    makes unrelated changes is a bad training example. Requiring the before/after diff to
    equal this spec rejects gratuitous actions, which is what a state-only predicate
    silently lets through.
    """

    opp_stage: dict[str, str] = field(default_factory=dict)
    case_priority: dict[str, str] = field(default_factory=dict)
    case_status: dict[str, str] = field(default_factory=dict)
    tasks: list[tuple[str, str]] = field(default_factory=list)
    # Billing-ops effects
    quote_status: dict[str, str] = field(default_factory=dict)
    quote_totals: dict[str, float] = field(default_factory=dict)
    quote_line_counts: dict[str, int] = field(default_factory=dict)
    approval_quotes: list[str] = field(default_factory=list)
    subscription_qty: dict[str, int] = field(default_factory=dict)
    invoice_status: dict[str, str] = field(default_factory=dict)


def check_effects(before: World, after: World, eff: Effects) -> bool:
    for oid, o in after.opportunities.items():
        if o["stage"] != eff.opp_stage.get(oid, before.opportunities[oid]["stage"]):
            return False
    for cid, c in after.cases.items():
        if c["priority"] != eff.case_priority.get(cid, before.cases[cid]["priority"]):
            return False
        if c["status"] != eff.case_status.get(cid, before.cases[cid]["status"]):
            return False
    new_tasks = after.tasks[len(before.tasks):]
    got = sorted((t["account_id"], t["subject"]) for t in new_tasks)
    if got != sorted(eff.tasks):
        return False

    # Billing ops: quotes must exist only where expected, with exact totals,
    # line counts and status; approval records must match exactly.
    for qid, q in after.quotes.items():
        allowed_status = eff.quote_status.get(qid, "Draft")
        if q["status"] != allowed_status:
            return False
        if qid in eff.quote_totals and abs(q["total"] - eff.quote_totals[qid]) > 0.01:
            return False
        if qid in eff.quote_line_counts and len(q["line_items"]) != eff.quote_line_counts[qid]:
            return False
        expected_approval = qid in eff.approval_quotes
        if q["approval_requested"] != expected_approval:
            return False
    for sid, s in after.subscriptions.items():
        if s["quantity"] != eff.subscription_qty.get(sid, before.subscriptions[sid]["quantity"]):
            return False
    for iid, inv in after.invoices.items():
        if inv["status"] != eff.invoice_status.get(iid, before.invoices[iid]["status"]):
            return False
    return True


@dataclass
class Task:
    task_id: str
    family: str
    instruction: str
    account_id: str
    gold_calls: list[dict]
    effects: Effects
    difficulty: int  # number of tool calls in the gold trace

    def check(self, before: World, after: World) -> bool:
        return check_effects(before, after, self.effects)


# Phrasing variants per family. A single template per family makes n-gram
# decontamination fire on nearly every example (only the entity name varies), which
# either destroys the dataset or hides real leakage. Diversifying surface form is the fix.
INSTRUCTION_VARIANTS = {
    "bulk_stage_update": [
        "At {name}, move open opportunities {opps} to the {stage} stage.",
        "For {name}: set these open opportunities to {stage} - {opps}.",
        "Advance {opps} at {name} to {stage}.",
        "Please move {opps} at {name} into {stage}.",
        "{name} - put {opps} into the {stage} stage.",
        "Take the open opportunities {opps} at {name} and set each to {stage}.",
    ],
    "escalate_and_followup": [
        "For {name} ({account}), escalate case {case} to Critical, then create a High priority task titled \"{subject}\".",
        "Raise case {case} at {name} ({account}) to Critical priority, and add a High priority task called \"{subject}\".",
        "Escalate {case} ({name}, {account}) to Critical and open a High priority follow-up task \"{subject}\".",
        "At {name} ({account}), bump case {case} to Critical, then log \"{subject}\" as a High priority task.",
        "Set case {case} at {name} ({account}) to Critical priority, and set up a High priority task named \"{subject}\".",
    ],
    "create_followup_task": [
        "Create a Medium priority task titled \"{subject}\" on account {account} ({name}).",
        "On {account} ({name}), add a Medium priority task called \"{subject}\".",
        "Log a Medium priority task \"{subject}\" against {account} ({name}).",
        "For account {account} ({name}), open a task named \"{subject}\" at Medium priority.",
        "Add the task \"{subject}\" to {account} ({name}) with Medium priority.",
    ],
    "quote_with_discount": [
        "At {name}, create a {pb} quote on opportunity {opp}, add {qty} x {product} using price book entry {pbe} at {disc}% discount, then activate it.",
        "{name} needs a live quote: opportunity {opp}, {pb} price book, price book entry {pbe}, {qty} units at {disc}% off. Create and activate it.",
        "Build an activated quote for {name} from opportunity {opp} in {pb}: entry {pbe}, quantity {qty}, discount {disc}%.",
        "Create a quote on {opp} for {name} using {pb} and entry {pbe}, add {qty} units at {disc}% off, and activate.",
        "Quote and activate for {name}: opportunity {opp}, {pb}, entry {pbe}, {qty} units, {disc}% discount.",
    ],
    "subscription_resize": [
        "Set subscription {sub} at {name} to a quantity of {qty}.",
        "Change the quantity on subscription {sub} ({name}) to {qty}.",
        "{name} is resizing subscription {sub} - set it to {qty} units.",
        "Amend subscription {sub} for {name} so the quantity becomes {qty}.",
        "Update subscription {sub} at {name} to quantity {qty}.",
    ],
    "invoice_dispute": [
        "Dispute invoice {inv} for {name}, reason: \"{reason}\".",
        "Open a dispute on invoice {inv} ({name}). Reason: \"{reason}\".",
        "{name} is disputing invoice {inv} - log it with reason \"{reason}\".",
        "File a dispute against {inv} for {name}, citing \"{reason}\".",
        "Raise a dispute on {inv} ({name}) because of \"{reason}\".",
    ],
}


def _phr(family: str, rng: random.Random, **kw) -> str:
    return rng.choice(INSTRUCTION_VARIANTS[family]).format(**kw)


def _escalation_status(priority: str) -> str:
    return "Escalated" if priority in ("High", "Critical") else "Open"


def make_tasks(world: World, seed: int = 11) -> list[Task]:
    """Build the task set.

    Every instruction carries the concrete identifiers the answer needs. An earlier
    version asked for things like "move every opportunity over $40k", which is
    unanswerable in one assistant turn: the ids only exist in a tool result the model
    never sees. The held-out base model scored 0.000 for that reason, not because it was
    weak. Tasks are now solvable from the prompt alone, which is what makes the accuracy
    number mean anything.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []

    for acct in world.accounts.values():
        aid, name = acct["id"], acct["name"]
        opps = [o for o in world.opportunities.values()
                if o["account_id"] == aid and o["open"]]

        # A. bulk stage update on named opportunities
        if opps:
            target = rng.choice(["Proposal", "Negotiation"])
            movable = [o for o in opps if o["stage"] != target][:3]
            if movable:
                ids = [o["id"] for o in movable]
                gold = [{"name": "update_opportunity_stage",
                         "arguments": {"opportunity_id": o["id"], "stage": target}}
                        for o in movable]
                tasks.append(Task(
                    task_id=f"T{len(tasks):04d}",
                    family="bulk_stage_update",
                    instruction=_phr("bulk_stage_update", rng, name=name,
                                     opps=", ".join(ids), stage=target),
                    account_id=aid, gold_calls=gold,
                    effects=Effects(opp_stage={o["id"]: target for o in movable}),
                    difficulty=len(gold)))

        # B. escalate a named case, then a follow-up task
        open_cases = [c for c in world.cases.values()
                      if c["account_id"] == aid and c["status"] == "Open"]
        if open_cases:
            case = rng.choice(open_cases)
            subject = rng.choice(SUBJECTS)
            gold = [
                {"name": "escalate_case",
                 "arguments": {"case_id": case["id"], "priority": "Critical"}},
                {"name": "create_task",
                 "arguments": {"account_id": aid, "subject": subject, "priority": "High"}},
            ]
            tasks.append(Task(
                task_id=f"T{len(tasks):04d}",
                family="escalate_and_followup",
                instruction=_phr("escalate_and_followup", rng, name=name,
                                 account=aid, case=case["id"], subject=subject),
                account_id=aid, gold_calls=gold,
                effects=Effects(
                    case_priority={case["id"]: "Critical"},
                    case_status={case["id"]: _escalation_status("Critical")},
                    tasks=[(aid, subject)],
                ),
                difficulty=len(gold)))

        # C. single follow-up task on a named account
        subject = rng.choice(SUBJECTS)
        gold = [{"name": "create_task",
                 "arguments": {"account_id": aid, "subject": subject,
                               "priority": "Medium"}}]
        tasks.append(Task(
            task_id=f"T{len(tasks):04d}",
            family="create_followup_task",
            instruction=_phr("create_followup_task", rng, name=name,
                             account=aid, subject=subject),
            account_id=aid, gold_calls=gold,
            effects=Effects(tasks=[(aid, subject)]),
            difficulty=len(gold)))

        # D. quote with discount, and the approval threshold
        if opps:
            opp = rng.choice(opps)
            pb = rng.choice(PRICEBOOKS)
            prod = rng.choice(PRODUCT_NAMES)
            pbe_id = f"PBE-{pb.split('-')[0]}-{prod[0]}"
            entry = world.pricebook_entries[pbe_id]
            qty = rng.choice([5, 10, 25, 50])
            above = rng.random() < 0.5
            disc = (round(rng.uniform(22, 35), 1) if above
                    else round(rng.uniform(0, 18), 1))
            qid = f"QUO-{opp['id'][4:]}"
            net = round(entry["unit_price"] * qty * (1 - disc / 100.0), 2)
            gold = [
                {"name": "create_quote",
                 "arguments": {"opportunity_id": opp["id"], "pricebook": pb}},
                {"name": "add_quote_line_item",
                 "arguments": {"quote_id": qid, "pricebook_entry_id": pbe_id,
                               "quantity": qty, "discount_pct": disc}},
            ]
            eff = Effects(
                quote_status={qid: "Activated"},
                quote_totals={qid: net},
                quote_line_counts={qid: 1},
            )
            if above:
                gold.append({"name": "request_quote_approval",
                             "arguments": {"quote_id": qid,
                                           "reason": f"discount {disc}% exceeds "
                                                     f"{DISCOUNT_THRESHOLD:.0f}% threshold"}})
                eff.approval_quotes = [qid]
            gold.append({"name": "activate_quote", "arguments": {"quote_id": qid}})
            tasks.append(Task(
                task_id=f"T{len(tasks):04d}",
                family="quote_with_discount",
                instruction=_phr("quote_with_discount", rng, name=name, opp=opp["id"],
                                 pb=pb, product=prod[1], pbe=pbe_id, qty=qty, disc=disc),
                account_id=aid, gold_calls=gold, effects=eff, difficulty=len(gold)))

        # E. subscription resize on a named subscription
        active_subs = [s for s in world.subscriptions.values()
                       if s["account_id"] == aid and s["status"] == "Active"]
        if active_subs:
            sub = rng.choice(active_subs)
            target_qty = rng.choice([5, 15, 40, 75, 120, 200])
            if target_qty != sub["quantity"]:
                gold = [{"name": "amend_subscription_quantity",
                         "arguments": {"subscription_id": sub["id"],
                                       "quantity": target_qty}}]
                tasks.append(Task(
                    task_id=f"T{len(tasks):04d}",
                    family="subscription_resize",
                    instruction=_phr("subscription_resize", rng, name=name,
                                     sub=sub["id"], qty=target_qty),
                    account_id=aid, gold_calls=gold,
                    effects=Effects(subscription_qty={sub["id"]: target_qty}),
                    difficulty=len(gold)))

        # F. invoice dispute on a named invoice
        billable = [i for i in world.invoices.values()
                    if i["account_id"] == aid and i["status"] in ("Posted", "Paid")]
        if billable:
            inv = rng.choice(billable)
            reason = rng.choice([
                "duplicate charge", "wrong seat count", "missing credit",
                "incorrect tax", "cancelled line still billed",
            ])
            gold = [{"name": "dispute_invoice",
                     "arguments": {"invoice_id": inv["id"], "reason": reason}}]
            tasks.append(Task(
                task_id=f"T{len(tasks):04d}",
                family="invoice_dispute",
                instruction=_phr("invoice_dispute", rng, name=name,
                                 inv=inv["id"], reason=reason),
                account_id=aid, gold_calls=gold,
                effects=Effects(invoice_status={inv["id"]: "Disputed"}),
                difficulty=len(gold)))

        if len(tasks) >= 1100:
            break

    return tasks


# ---------------------------------------------------------------- canonical set
# Single source of truth for the generator and the eval harness. If these drift, the
# harness silently scores against different tasks than the ones that were trained on.
WORLD_SEED = 7
N_ACCOUNTS = 520
TASK_SEED = 11


def canonical_task_set() -> tuple[World, list["Task"]]:
    world = build_world(seed=WORLD_SEED, n_accounts=N_ACCOUNTS)
    return world, make_tasks(world, seed=TASK_SEED)
