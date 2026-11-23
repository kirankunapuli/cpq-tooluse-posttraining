"""Pipeline tests. Stdlib only: `python tests/test_pipeline.py` (also pytest-compatible).

These lock in the failure modes that were actually hit while building this, so they
cannot silently come back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "eval"))

from crm_world import (  # noqa: E402
    Effects,
    ToolError,
    build_world,
    check_effects,
    execute,
    make_tasks,
)
from filter_splits import _norm, _tokens, ngrams  # noqa: E402
from harness import parse_calls, score_one, wilson  # noqa: E402


def test_unknown_id_raises():
    w = build_world(seed=1, n_accounts=5)
    try:
        execute(w, {"name": "update_opportunity_stage",
                    "arguments": {"opportunity_id": "OPP-9999-0", "stage": "Proposal"}})
    except ToolError as e:
        assert "unknown opportunity_id" in str(e)
    else:
        raise AssertionError("expected ToolError")


def test_gold_trajectory_satisfies_effects():
    w = build_world(seed=1, n_accounts=5)
    for task in make_tasks(w, seed=3)[:20]:
        after = w.clone()
        for c in task.gold_calls:
            execute(after, c)
        assert task.check(w, after), f"{task.task_id} gold should satisfy its own effects"


def test_extra_side_effect_is_rejected():
    """Regression: the state-only check accepted gratuitous actions, so traces with
    unprompted create_task calls were written into the *training* set."""
    w = build_world(seed=1, n_accounts=8)
    task = next(t for t in make_tasks(w, seed=3) if t.family == "bulk_stage_update")
    after = w.clone()
    for c in task.gold_calls:
        execute(after, c)
    assert task.check(w, after)
    execute(after, {"name": "create_task",
                    "arguments": {"account_id": task.account_id,
                                  "subject": "unprompted", "priority": "Low"}})
    assert not task.check(w, after), "an extra task must invalidate the trace"


def test_missing_update_is_rejected():
    w = build_world(seed=1, n_accounts=8)
    task = next(t for t in make_tasks(w, seed=3) if t.family == "bulk_stage_update")
    after = w.clone()
    updates = [c for c in task.gold_calls if c["name"] == "update_opportunity_stage"]
    for c in task.gold_calls:
        if c is updates[-1]:
            continue
        execute(after, c)
    assert not task.check(w, after)


def test_effects_rejects_unrelated_case_change():
    w = build_world(seed=1, n_accounts=8)
    task = next(t for t in make_tasks(w, seed=3) if t.family == "escalate_and_followup")
    after = w.clone()
    for c in task.gold_calls:
        execute(after, c)
    assert task.check(w, after)
    other = next(c for c in after.cases.values() if c["account_id"] != task.account_id)
    other["priority"] = "Critical"
    assert not check_effects(w, after, task.effects)


def test_parse_calls_handles_fenced_json():
    raw = 'Sure!\n```json\n[{"name": "list_cases", "arguments": {"account_id": "ACC-1000"}}]\n```'
    calls, err = parse_calls({"task_id": "T1", "raw": raw})
    assert err is None and calls and calls[0]["name"] == "list_cases"


def test_parse_calls_rejects_prose():
    calls, err = parse_calls({"task_id": "T1", "raw": "I think so!"})
    assert calls is None and err == "parse_error"


def test_score_one_distinguishes_invalid_id_from_goal_miss():
    w = build_world(seed=1, n_accounts=8)
    task = next(t for t in make_tasks(w, seed=3) if t.family == "escalate_and_followup")
    ok, cat = score_one(w, task, [{"name": "escalate_case",
                                   "arguments": {"case_id": "CASE-0000-0",
                                                 "priority": "Critical"}}])
    assert not ok and cat == "invalid_id"

    unrelated = [{"name": "list_cases",
                  "arguments": {"account_id": task.account_id, "status": "Open"}}]
    ok, cat = score_one(w, task, unrelated)
    assert not ok and cat == "goal_not_satisfied"


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson(60, 100)
    assert lo < 0.60 < hi
    lo0, hi0 = wilson(0, 50)
    assert lo0 == 0.0 and hi0 > 0.0
    lo1, hi1 = wilson(50, 50)
    assert lo1 < 1.0 and hi1 == 1.0


def test_ngram_masking_distinguishes_template_from_literal():
    # Same account, different amount: identical template, different text.
    a = "Move every open opportunity over $40,000 at Acme Ltd to the Proposal stage."
    b = "Move every open opportunity over $90,000 at Acme Ltd to the Proposal stage."
    masked = bool(ngrams(_tokens(a), 13) & ngrams(_tokens(b), 13))
    literal = bool(ngrams(_tokens(a, mask_ids=False), 13)
                   & ngrams(_tokens(b, mask_ids=False), 13))
    assert masked, "id-masked 13-grams should collide: masked fields are collapsed"
    assert not literal, "literal 13-grams must not collide: the amount differs"
    assert "$<amt>" in _norm(a)

    # Truly identical text is literal leakage at any n, which is what must be dropped.
    assert ngrams(_tokens(a, mask_ids=False), 13) & ngrams(_tokens(a, mask_ids=False), 13)


def test_normalisation_is_idempotent():
    s = "Move OPP-1234-0 at ACC-5678 over $1,000"
    assert _norm(_norm(s)) == _norm(s)


def test_effects_spec_is_json_serialisable_shape():
    e = Effects(opp_stage={"O": "Proposal"}, tasks=[("ACC-1", "do")])
    json.dumps({"opp": e.opp_stage, "tasks": e.tasks})


def test_activation_refused_without_approval_above_threshold():
    """The CPQ policy invariant: a quote with a discount above the threshold cannot be
    activated until approval is requested. This is the 'human approval boundary' the JD
    asks about, enforced in the environment rather than in a prompt."""
    from crm_world import DISCOUNT_THRESHOLD

    w = build_world(seed=2, n_accounts=3)
    opp = next(iter(w.opportunities))
    q = execute(w, {"name": "create_quote",
                    "arguments": {"opportunity_id": opp, "pricebook": "USD-Standard"}})
    qid = q["quote_id"]
    execute(w, {"name": "add_quote_line_item",
                "arguments": {"quote_id": qid,
                              "pricebook_entry_id": "PBE-USD-CPQ-PLATFORM",
                              "quantity": 5,
                              "discount_pct": DISCOUNT_THRESHOLD + 10}})
    try:
        execute(w, {"name": "activate_quote", "arguments": {"quote_id": qid}})
    except ToolError as e:
        assert "needs approval" in str(e)
    else:
        raise AssertionError("activation must be refused before approval is requested")

    execute(w, {"name": "request_quote_approval",
                "arguments": {"quote_id": qid, "reason": "strategic deal"}})
    execute(w, {"name": "activate_quote", "arguments": {"quote_id": qid}})
    assert w.quotes[qid]["status"] == "Activated"


def test_activation_allowed_below_threshold():
    from crm_world import DISCOUNT_THRESHOLD

    w = build_world(seed=2, n_accounts=3)
    opp = next(iter(w.opportunities))
    qid = execute(w, {"name": "create_quote",
                      "arguments": {"opportunity_id": opp,
                                    "pricebook": "USD-Standard"}})["quote_id"]
    execute(w, {"name": "add_quote_line_item",
                "arguments": {"quote_id": qid,
                              "pricebook_entry_id": "PBE-USD-CPQ-PLATFORM",
                              "quantity": 2,
                              "discount_pct": max(0.0, DISCOUNT_THRESHOLD - 5)}})
    execute(w, {"name": "activate_quote", "arguments": {"quote_id": qid}})
    assert w.quotes[qid]["status"] == "Activated"


def test_pricebook_mismatch_refused():
    """A line from the wrong price book is a silent currency error in real CPQ; here it
    is loud, which is the point of putting the check in the tool layer."""
    w = build_world(seed=2, n_accounts=3)
    opp = next(iter(w.opportunities))
    qid = execute(w, {"name": "create_quote",
                      "arguments": {"opportunity_id": opp,
                                    "pricebook": "USD-Standard"}})["quote_id"]
    try:
        execute(w, {"name": "add_quote_line_item",
                    "arguments": {"quote_id": qid,
                                  "pricebook_entry_id": "PBE-EUR-CPQ-PLATFORM",
                                  "quantity": 5}})
    except ToolError as e:
        assert "pricebook mismatch" in str(e)
    else:
        raise AssertionError("cross-pricebook line item must be refused")


def test_quote_total_arithmetic():
    w = build_world(seed=2, n_accounts=3)
    opp = next(iter(w.opportunities))
    qid = execute(w, {"name": "create_quote",
                      "arguments": {"opportunity_id": opp,
                                    "pricebook": "USD-Standard"}})["quote_id"]
    unit = w.pricebook_entries["PBE-USD-CPQ-PLATFORM"]["unit_price"]
    execute(w, {"name": "add_quote_line_item",
                "arguments": {"quote_id": qid,
                              "pricebook_entry_id": "PBE-USD-CPQ-PLATFORM",
                              "quantity": 10, "discount_pct": 10}})
    expected = round(unit * 10 * 0.9, 2)
    assert abs(w.quotes[qid]["total"] - expected) < 0.01


def test_amendment_refused_on_inactive_subscription():
    w = build_world(seed=2, n_accounts=12)
    suspended = next((s for s in w.subscriptions.values() if s["status"] != "Active"), None)
    if suspended is None:
        return  # seed produced no inactive subscription; nothing to assert
    try:
        execute(w, {"name": "amend_subscription_quantity",
                    "arguments": {"subscription_id": suspended["id"], "quantity": 99}})
    except ToolError as e:
        assert "Active" in str(e)
    else:
        raise AssertionError("only Active subscriptions may be amended")


def test_preference_pairs_are_clean():
    """Every DPO pair must have a correct chosen side and a genuinely wrong rejected side.

    Rejections are derived from the verifier's own output, so this test guards the
    assumption that makes the derivation legitimate: the two sides really do differ in
    correctness, not just in text.
    """
    from crm_world import canonical_task_set

    pref = ROOT / "data" / "pref_valid.jsonl"
    if not pref.exists():
        return  # pipeline not run yet
    world, tasks = canonical_task_set()
    tbi = {t.task_id: t for t in tasks}
    checked = 0
    for line in pref.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        task = tbi[r["task_id"]]
        assert r["chosen"] != r["rejected"]

        after = world.clone()
        for c in json.loads(r["chosen"]):
            execute(after, c)
        assert task.check(world, after), "chosen side must satisfy the task"

        after2 = world.clone()
        raised = False
        try:
            for c in json.loads(r["rejected"]):
                execute(after2, c)
        except ToolError:
            raised = True
        if not raised:
            assert not task.check(world, after2), "rejected side must not satisfy the task"
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# Training-runner regression tests. These need no torch and no GPU: the parts that
# failed in Colab were the shape handling around generate(), so the fakes model exactly
# that API surface.
# ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, values, shape):
        self.values = values
        self.shape = shape

    def to(self, _device):
        return self


class _FakeBatch(dict):
    """Mimics a transformers BatchEncoding: dict-like, has .get, no .shape."""


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, _msgs, add_generation_prompt=True,
                            return_tensors=None, return_dict=False):
        ids = _FakeTensor([[7, 8, 9]], (1, 3))
        if return_dict:
            return _FakeBatch(input_ids=ids,
                              attention_mask=_FakeTensor([[1, 1, 1]], (1, 3)))
        return _FakeBatch(input_ids=ids, attention_mask=None)

    def decode(self, ids, skip_special_tokens=True):
        return "decoded(%d)" % len(ids)


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.seen = None

    def eval(self):
        return self

    def generate(self, input_ids=None, **_kw):
        # The original bug: generate received the BatchEncoding and failed on .shape.
        assert hasattr(input_ids, "shape"), "generate needs a tensor with .shape"
        self.seen = input_ids
        return [[0, 0, 0, 11, 12]]


def test_split_batch_normalises_encoding_and_tensor():
    from train_eval_colab import split_batch

    ids, mask = _FakeTensor([[1]], (1, 1)), _FakeTensor([[1]], (1, 1))
    got_ids, got_mask = split_batch(_FakeBatch(input_ids=ids, attention_mask=mask))
    assert got_ids is ids and got_mask is mask

    plain = _FakeTensor([[1]], (1, 1))
    got_ids, got_mask = split_batch(plain)
    assert got_ids is plain and got_mask is None


def test_generate_all_survives_batch_encoding():
    """Regression: Colab failed with AttributeError because apply_chat_template returned a
    BatchEncoding and generate() tried to read inputs_tensor.shape."""
    import sys
    import types

    if "torch" not in sys.modules:
        stub = types.ModuleType("torch")

        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        stub.no_grad = _NoGrad
        sys.modules["torch"] = stub

    from train_eval_colab import generate_all

    rows = [{"task_id": "T1", "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]}]
    model = _FakeModel()
    out = generate_all(model, _FakeTok(), rows, "\nRespond with JSON.", 16)
    assert out["T1"] == "decoded(2)", out["T1"]
    assert model.seen is not None


class _OldStyleTok(_FakeTok):
    """Older transformers: apply_chat_template rejects return_dict, returns a tensor."""

    def apply_chat_template(self, _msgs, add_generation_prompt=True, return_tensors=None):
        return _FakeTensor([[7, 8, 9]], (1, 3))


def test_generate_all_handles_old_style_tokenizer():
    import sys
    import types

    if "torch" not in sys.modules:
        stub = types.ModuleType("torch")

        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        stub.no_grad = _NoGrad
        sys.modules["torch"] = stub

    from train_eval_colab import generate_all

    rows = [{"task_id": "T2", "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]}]
    out = generate_all(_FakeModel(), _OldStyleTok(), rows, "", 16)
    assert out["T2"] == "decoded(2)", out["T2"]


class _ThinkingTok(_FakeTok):
    """Hybrid reasoning model: rejects calls that do not pass enable_thinking."""
    calls = []

    def apply_chat_template(self, _msgs, add_generation_prompt=True,
                            return_tensors=None, return_dict=False, enable_thinking=None):
        type(self).calls.append(enable_thinking)
        if enable_thinking is True:
            raise AssertionError("we must never enable reasoning traces for this task")
        return _FakeBatch(input_ids=_FakeTensor([[7, 8, 9]], (1, 3)),
                          attention_mask=_FakeTensor([[1, 1, 1]], (1, 3)))


class _JinjaStrictTok(_FakeTok):
    """Template that blows up on the unknown enable_thinking variable."""

    def apply_chat_template(self, _msgs, add_generation_prompt=True,
                            return_tensors=None, return_dict=False, **kw):
        if "enable_thinking" in kw:
            err = type("UndefinedError", (Exception,), {})("enable_thinking undefined")
            raise err
        return _FakeBatch(input_ids=_FakeTensor([[7, 8, 9]], (1, 3)),
                          attention_mask=_FakeTensor([[1, 1, 1]], (1, 3)))


def test_encode_chat_disables_thinking_and_survives_strict_templates():
    from train_eval_colab import encode_chat

    _ThinkingTok.calls.clear()
    enc = encode_chat(_ThinkingTok(), [{"role": "user", "content": "u"}])
    assert _FakeTensor is not None
    assert enc is not None and _ThinkingTok.calls and _ThinkingTok.calls[0] is False

    enc2 = encode_chat(_JinjaStrictTok(), [{"role": "user", "content": "u"}])
    assert enc2 is not None


def test_autocast_dtype_respects_turing_hardware():
    """Regression: bf16 on a T4 killed the Colab run. is_bf16_supported() is not a usable
    probe there because it reports True for emulated bf16, so gate on compute capability."""
    from train_eval_colab import _autocast_dtype

    class _T:
        bfloat16, float16, float32 = "bf16", "fp16", "fp32"

    def torch_for(capability, bf16_probe=True):
        class Cuda:
            @staticmethod
            def get_device_capability(_i=0):
                return capability

            @staticmethod
            def is_bf16_supported():
                # Turing emulates bf16, so real torch returns True here. Do not trust it.
                return bf16_probe

        class Torch:
            bfloat16, float16, float32 = _T.bfloat16, _T.float16, _T.float32
            cuda = Cuda()
        return Torch

    # T4, compute 7.5, and the probe lies about bf16.
    assert _autocast_dtype(torch_for((7, 5)), "cuda") == "fp16"
    # A100 and H100, compute 8.0 and 9.0.
    assert _autocast_dtype(torch_for((8, 0)), "cuda") == "bf16"
    assert _autocast_dtype(torch_for((9, 0)), "cuda") == "bf16"
    # Older torch without the capability probe must not crash.
    class NoProbe:
        bfloat16, float16, float32 = _T.bfloat16, _T.float16, _T.float32

        class cuda:
            @staticmethod
            def get_device_capability(_i=0):
                raise RuntimeError("not available")

    assert _autocast_dtype(NoProbe, "cuda") == "fp32"
    assert _autocast_dtype(_T, "mps") == "bf16"
    assert _autocast_dtype(_T, "cpu") == "fp32"


def test_sft_config_refuses_to_silently_truncate():
    """Regression: if a TRL build does not accept max_length under any name and it is
    dropped, the default sequence length truncates every example and the run is wasted."""
    from train_eval_colab import make_sft_config

    class _Native:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    cfg = make_sft_config(_Native, max_length=2048, warmup_ratio=0.05, _total_steps=10,
                          seed=13)
    assert cfg.max_length == 2048

    class _Old(_Native):
        def __init__(self, *, max_seq_length=None, seed=None):
            self.max_seq_length = max_seq_length
            self.seed = seed

    cfg = make_sft_config(_Old, max_length=2048, seed=13)
    assert cfg.max_seq_length == 2048

    class _Unrelated(_Native):
        def __init__(self, *, seed=None):
            self.seed = seed

    try:
        make_sft_config(_Unrelated, max_length=2048, seed=13)
    except RuntimeError as e:
        assert "sequence length" in str(e)
    else:
        raise AssertionError("expected RuntimeError instead of a truncated run")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
