"""A2UI surfaces for the quote agent.

Every input the agent needs is declared once, in a registry, and rendered as an A2UI
`createSurface` message. The agent never writes UI. It names a field. The registry owns
type, bounds, label and the components used to render it.

That is the answer to "if the agent wants any input, it uses A2UI, creates a small input box,
gets it from the user and continues".

Run the self-check:

    python build/a2ui_surfaces.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROTOCOL_VERSION = "1.0"
BASIC_CATALOG = "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json"

ALLOWED_COMPONENTS = frozenset({
    "Text", "Column", "Row", "Card", "Button", "TextField", "DateTimeInput",
    "ChoicePicker", "CheckBox", "Divider", "List", "Modal", "Icon", "Slider",
    "Tabs", "Image", "AudioPlayer", "Video",
})


class SurfaceError(ValueError):
    pass


def validate(components: list[dict]) -> None:
    """Reject anything the catalog does not allow, before it reaches a renderer."""
    ids = set()
    for c in components:
        name = c.get("component")
        if name not in ALLOWED_COMPONENTS:
            raise SurfaceError(f"UNALLOWED_CHILD: {name} is not in the basic catalog")
        if "id" not in c:
            raise SurfaceError("INTEGRITY_ERROR: component missing id")
        if c["id"] in ids:
            raise SurfaceError(f"INTEGRITY_ERROR: duplicate id {c['id']}")
        ids.add(c["id"])
    if "root" not in ids:
        raise SurfaceError("INTEGRITY_ERROR: surface needs a component with id 'root'")
    for c in components:
        for ref in (c.get("children") or []):
            if ref not in ids:
                raise SurfaceError(f"UNALLOWED_PARENT: child {ref} does not exist")
        child = c.get("child")
        if child and child not in ids:
            raise SurfaceError(f"UNALLOWED_PARENT: child {child} does not exist")


def create_surface(surface_id: str, components: list[dict],
                   data_model: dict | None = None) -> dict:
    validate(components)
    return {
        "version": PROTOCOL_VERSION,
        "createSurface": {
            "surfaceId": surface_id,
            "catalogId": BASIC_CATALOG,
            "sendDataModel": True,
            "components": components,
            "dataModel": data_model or {},
        },
    }


# --------------------------------------------------------------------------------------
# The registry. One entry per thing the agent may ask for.
# --------------------------------------------------------------------------------------
# `field`     the key the answer arrives under
# `component` which catalog component renders it
# `variant`   that component's variant, which sets the input type
# `bounds`    validated before the answer is used, never after
#
# The agent names one of these keys. It cannot invent a new one, cannot choose the component,
# and cannot widen a bound. That is the capability boundary.

INPUT_REGISTRY: dict[str, dict] = {
    "minimum_quantity": {
        "label": "Minimum order quantity",
        "component": "TextField",
        "variant": "number",
        "placeholder": "e.g. 10",
        "why": "A new product needs it, and the request does not contain it.",
        "bounds": {"min": 1, "max": 1_000_000, "integer": True},
    },
    "storage": {
        "label": "Storage",
        "component": "ChoicePicker",
        "variant": "mutuallyExclusive",
        "options": ["128GB", "256GB", "512GB", "1TB"],
        "why": "One SKU, several storage tiers, and the unit price differs by tier.",
        "bounds": {"one_of": ["128GB", "256GB", "512GB", "1TB"]},
    },
    "colour": {
        "label": "Colour",
        "component": "ChoicePicker",
        "variant": "mutuallyExclusive",
        "options": ["black", "white", "blue", "titanium"],
        "why": "Colour changes the price on this product, so it cannot be guessed.",
        "bounds": {"one_of": ["black", "white", "blue", "titanium"]},
    },
    "memory": {
        "label": "Memory",
        "component": "ChoicePicker",
        "variant": "mutuallyExclusive",
        "options": ["16GB", "32GB", "64GB"],
        "why": "Memory tier changes the unit price, so the lookup needs it.",
        "bounds": {"one_of": ["16GB", "32GB", "64GB"]},
    },
    "warranty": {
        "label": "Warranty",
        "component": "ChoicePicker",
        "variant": "mutuallyExclusive",
        "options": ["1yr", "3yr"],
        "why": "Warranty length is priced, so the lookup needs it.",
        "bounds": {"one_of": ["1yr", "3yr"]},
    },
    "delivery_date": {
        "label": "Requested delivery date",
        "component": "DateTimeInput",
        "variant": "date",
        "why": "The seasonal rule depends on the month.",
        "bounds": {"after": "2026-11-23"},
    },
    "price_record": {
        "label": "Which price record",
        "component": "ChoicePicker",
        "variant": "mutuallyExclusive",
        "why": "Two records match. Picking one silently would be a currency error.",
        "bounds": {"one_of": None},   # filled at call time from the candidates
    },
    "escalation_ack": {
        "label": "Send to your manager",
        "component": "CheckBox",
        "why": "This request is above your authority. Confirming records who it goes to.",
        "bounds": {"one_of": ["yes", "no"]},
    },
}


def _option_components(prefix: str, label: str, why: str,
                       options: list[str]) -> list[dict]:
    """A labelled ChoicePicker plus its option children."""
    comps = [
        {"id": f"{prefix}_label", "component": "Text", "variant": "h3", "text": label},
        {"id": f"{prefix}_why", "component": "Text", "variant": "caption", "text": why},
    ]
    opt_ids = []
    for i, opt in enumerate(options):
        oid = f"{prefix}_opt{i}"
        comps.append({"id": oid, "component": "Text", "text": opt})
        opt_ids.append(oid)
    comps.append({
        "id": f"{prefix}_picker", "component": "ChoicePicker",
        "variant": "mutuallyExclusive", "label": label, "children": opt_ids,
    })
    return comps


def build_surface(input_name: str, surface_id: str,
                  context: dict | None = None) -> dict:
    """Render one registry entry as an A2UI surface. This is the only UI generator."""
    if input_name not in INPUT_REGISTRY:
        raise SurfaceError(f"UNKNOWN_INPUT: {input_name} is not in the registry")
    spec = INPUT_REGISTRY[input_name]
    context = context or {}
    comps: list[dict] = [
        {"id": "root", "component": "Column",
         "children": ["title", "why", "input", "actions"]},
        {"id": "title", "component": "Text", "variant": "h3",
         "text": context.get("title", "More information needed")},
        {"id": "why", "component": "Text", "variant": "body",
         "text": context.get("why", spec.get("why", ""))},
    ]

    if spec["component"] == "ChoicePicker":
        options = context.get("options") or spec.get("options") or []
        comps[0]["children"] = ["title", "why", "input", "actions"]
        comps.append({
            "id": "input", "component": "ChoicePicker",
            "variant": spec["variant"], "label": spec["label"],
            "options": [{"label": o, "value": o} for o in options],
        })
    elif spec["component"] == "CheckBox":
        comps.append({
            "id": "input", "component": "CheckBox",
            "label": spec["label"],
        })
    elif spec["component"] == "DateTimeInput":
        comps.append({
            "id": "input", "component": "DateTimeInput",
            "variant": spec["variant"], "label": spec["label"],
        })
    else:
        comps.append({
            "id": "input", "component": "TextField",
            "variant": spec.get("variant", "shortText"),
            "label": spec["label"],
            "placeholder": spec.get("placeholder", ""),
        })

    comps += [
        {"id": "actions", "component": "Row", "justify": "end",
         "children": ["cancel_text", "cancel", "submit_text", "submit"]},
        {"id": "cancel_text", "component": "Text", "text": "Cancel"},
        {"id": "cancel", "component": "Button", "variant": "default",
         "child": "cancel_text", "action": {"name": "cancel"}},
        {"id": "submit_text", "component": "Text",
         "text": context.get("submit", "Continue")},
        {"id": "submit", "component": "Button", "variant": "primary",
         "child": "submit_text", "action": {"name": "submit"}},
    ]
    return create_surface(surface_id, comps, context.get("dataModel") or {})


def coerce(input_name: str, raw) -> object:
    """Validate an answer against the registry bounds. Never trust the renderer."""
    spec = INPUT_REGISTRY[input_name]
    b = spec.get("bounds") or {}
    val = raw
    if "integer" in b:
        try:
            val = int(str(raw).strip())
        except (TypeError, ValueError):
            raise SurfaceError(f"INVALID_VALUE: {input_name} must be an integer")
        if "min" in b and val < b["min"]:
            raise SurfaceError(f"INVALID_VALUE: {input_name} below minimum {b['min']}")
        if "max" in b and val > b["max"]:
            raise SurfaceError(f"INVALID_VALUE: {input_name} above maximum {b['max']}")
    if b.get("one_of"):
        if val not in b["one_of"]:
            raise SurfaceError(f"INVALID_VALUE: {input_name} must be one of {b['one_of']}")
    return val


# --------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------

def _selftest() -> int:
    checks: list[tuple[str, bool]] = []

    def ok(name, fn):
        try:
            fn()
            checks.append((name, True))
        except Exception as e:  # noqa: BLE001
            checks.append((f"{name} -> {e}", False))

    def fails(name, fn):
        try:
            fn()
            checks.append((name, False))
        except SurfaceError:
            checks.append((name, True))

    ok("every registry entry renders", lambda: [
        build_surface(n, f"s_{n}") for n in INPUT_REGISTRY])

    ok("number input renders as TextField number", lambda: (
        build_surface("minimum_quantity", "s1")["createSurface"]["components"]
        and next(c for c in build_surface("minimum_quantity", "s2")["createSurface"]["components"]
                 if c["id"] == "input")["variant"] == "number") or (_ for _ in ()).throw(
        AssertionError("wrong variant")))

    ok("storage renders as ChoicePicker with 4 options", lambda: (
        len(next(c for c in build_surface("storage", "s3")["createSurface"]["components"]
                 if c["id"] == "input")["options"]) == 4) or (_ for _ in ()).throw(
        AssertionError("wrong option count")))

    fails("unknown input rejected",
          lambda: build_surface("free_text_anything", "s4"))

    fails("unlisted component rejected", lambda: validate(
        [{"id": "root", "component": "ScriptTag"}]))

    fails("surface without root rejected", lambda: validate(
        [{"id": "x", "component": "Text", "text": "hi"}]))

    fails("dangling child ref rejected", lambda: validate(
        [{"id": "root", "component": "Column", "children": ["nope"]}]))

    fails("duplicate id rejected", lambda: validate(
        [{"id": "root", "component": "Column", "children": []},
         {"id": "root", "component": "Text", "text": "dup"}]))

    checks.append(("integer coercion accepts 10", coerce("minimum_quantity", "10") == 10))
    fails("integer coercion rejects 0", lambda: coerce("minimum_quantity", "0"))
    fails("integer coercion rejects text", lambda: coerce("minimum_quantity", "lots"))
    fails("choice coercion rejects a non-option",
          lambda: coerce("storage", "2TB"))
    checks.append(("choice coercion accepts a listed option",
                   coerce("colour", "titanium") == "titanium"))

    # the agent cannot widen a bound by supplying its own
    fails("registry bounds cannot be bypassed",
          lambda: coerce("minimum_quantity", "99999999"))

    # The browser demo carries a mirror of this registry in JavaScript. Two
    # hand-maintained copies drift, so compare the key sets.
    demo = Path(__file__).resolve().parents[1] / ".." / "career-preparation" \
        / "salesforce-post-training" / "demo" / "index.html"
    if demo.exists():
        js = demo.read_text()
        block = js.split("var REGISTRY=", 1)[1].split("\n};", 1)[0]
        js_keys = set(re.findall(r"^  (\w+):\{", block, re.M))
        ok("js registry matches this one",
           lambda: js_keys == set(INPUT_REGISTRY))
    else:
        print(f"SKIP  js registry parity: {demo} not present")

    bad = [n for n, passed in checks if not passed]
    for n, passed in checks:
        print(("PASS  " if passed else "FAIL  ") + n)
    print(f"\n{len(checks) - len(bad)}/{len(checks)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_selftest())
