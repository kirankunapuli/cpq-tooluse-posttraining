# Training data you can check by running it

Most synthetic tool-call data is accepted because it looks right. The JSON parses, the tool
names are real, the ids look like ids. None of that means the call did the right thing.

This repo generates tool-call data where acceptance is mechanical. Replay the model's calls
against a sandbox, and keep the example only if the resulting change matches exactly what the
task asked for. Everything else is rejected, and the rejections turn out to be useful too.

A 1.5B model trained on that data went from **0 correct out of 110** held-out tasks to
**100 out of 110**.

Dataset: [kirankunapuli/cpq-tooluse-verified](https://huggingface.co/datasets/kirankunapuli/cpq-tooluse-verified)

## Where this sits in post-training

Post-training is everything you do to a base model after pre-training. The methods below are the
usual ladder, and this repo touches three of the four. What separates them is the kind of feedback
each one needs, and that is where most of the engineering effort actually goes.

| Stage | What it is | How it works | What this project did | Result |
|---|---|---|---|---|
| **SFT**<br>supervised fine-tuning | The model copies correct answers | Loss lands on the answer tokens only, over traces the verifier accepted | 872 verified traces, QLoRA rank 16 on Qwen2.5-1.5B | 0/110 to **100/110** |
| **DPO**<br>direct preference optimisation | The model learns from pairs: same prompt, one better answer, one worse | Raise the probability of the chosen answer, lower the rejected one. No reward model needed | 2,180 pairs, produced free by the verifier's rejections | 102/110 to **68/110**, it regressed |
| **GRPO**<br>group relative policy optimisation | The model learns from its own attempts | Sample a group of answers per prompt, score each, use the group average as the baseline instead of a value network | Not run. Needs a harder task mix to produce any score spread | unrun |
| **RLVR**<br>RL with verifiable rewards | Reinforcement learning where the reward is a program, not a person | Reward 1 when the replay matches the declared effects, 0 otherwise. Nothing to label | The verifier already is that reward function | it is what makes GRPO possible here |

The ladder is meant to go SFT, then preference optimisation, then RL. SFT teaches a behaviour, DPO
shifts it toward what you prefer, GRPO explores beyond what your data covers. Running them in order
is normal. Assuming each one helps is not, which is why the DPO row above reports a regression
rather than a win.

## The idea

```
a task
  |
the model writes tool calls
  |
replay them on a copy of the world
  |
does the change match what the task declared?
  |
 yes -> keep it as training data
  |
  no -> reject it, label the defect, keep it as a counterexample
```

The world is a small sandbox holding accounts, opportunities, cases, price books, quotes,
subscriptions and invoices. Sixteen tools, and every tool changes state. Replay costs nothing
and leaves no trace, so the check is free and the rejected rows come with a label.

## One task, start to finish

Everything in this section is real output from the sandbox. Nothing here is an illustration.

### The task

```
Build an activated quote for Consolidated Messenger Pvt from opportunity OPP-1002-1
in APAC-Partner: entry PBE-APAC-CPQ-PLATFORM, quantity 25, discount 25.0%.
```

Two details matter. The discount is 25%, and this business has a rule: above 20%, a quote
cannot be activated until approval is requested. The task also names one specific price book
entry, so a different entry is wrong even if it still produces a valid quote.

### Step 1. The world before

```
quotes       none
approvals    none
```

### Step 2. The four calls that solve it

```
1  create_quote              OPP-1002-1, APAC-Partner
2  add_quote_line_item       PBE-APAC-CPQ-PLATFORM x 25, 25.0% off
3  request_quote_approval    "discount 25.0% exceeds 20% threshold"
4  activate_quote            QUO-1002-1
```

Call 3 is the interesting one. Without it, call 4 has nothing to succeed against.

### Step 3. Replay them on a copy

```
call 1   ok      quote QUO-1002-1 created, status Draft
call 2   ok      1 line added, net 25 x 28,080 x 0.75 = 526,500.00
call 3   ok      approval request filed, status pending
call 4   ok      status Activated
```

### Step 4. Compare before and after

```
the task declared                 the trace produced          same?
quote QUO-1002-1 Activated        quote QUO-1002-1 Activated  yes
total 526,500.00                  total 526,500.00            yes
1 line item                       1 line item                 yes
approval on QUO-1002-1            approval on QUO-1002-1      yes
no new tasks                      no new tasks                yes

verdict: accepted
```

### Step 5. The same task with one call removed

The generator also emits a wrong version. This one is identical except call 3 is gone.

```
1  create_quote         ok
2  add_quote_line_item  ok
3  activate_quote       RAISED: quote needs approval: discount exceeds threshold

verdict: rejected. defect: approval_threshold_violated
state left behind: status Draft, approval false, no approval record
```

The trace is well formed. It uses the right opportunity, the right price book entry, the right
quantity, the right discount. The only thing wrong is a call that is not there.

### Step 6. Why the rejection is the useful part

The approval rule lives in the tool, not in the prompt:

```python
if any(line.discount_pct > 20) and not quote.approval_requested:
    raise ToolError("quote needs approval: discount exceeds threshold")
```

A prompt can ask for that behaviour. A tool can guarantee it. On this dataset the rule alone
rejected **51 traces** that were otherwise perfect. With the rule in a prompt instead, all 51
become training data.

The same idea catches a wrong price book. Putting a EUR entry on a USD quote raises:

```
pricebook mismatch: entry PBE-EUR-DATA-ZERO is EUR-Standard, quote is USD-Standard
```

### Step 7. Why the comparison is equal, not contains

Suppose a trace solves the task and also creates a task nobody asked for.

```
declared effects     tasks: 0
after the trace      tasks: 1
verdict              rejected
```

A check that only asks "was the goal met" answers yes here. That is exactly the bug that put
185 traces with unrequested side effects into my training data before I caught it.

### Step 8. The rejection becomes a training pair

The verified trace and its broken twin answer the same prompt. That is already a preference
pair, with no human labelling:

```
prompt      the task above
chosen      the 4 calls
rejected    the 3 calls
defect      approval_threshold_violated
```

1,100 verified traces and 2,180 pairs like this came out of the same generator.

## Results

Same 110 held-out tasks, scored by executing the predicted calls and comparing the end state.
Not text similarity.

| | base | after fine-tuning |
|---|---:|---:|
| correct | 0 / 110 | **100 / 110** |
| 95% CI | 0.000 to 0.034 | 0.841 to 0.950 |
| parse failures | 39 | 0 |
| invented identifier | 62 | 0 |
| bad argument | 9 | 7 |
| goal not reached | 0 | 3 |

Paired delta +0.909, 95% CI [+0.855, +0.955].

### Why 100 out of 110 is not the number to quote

The base model failed to parse 39 times. So maybe the fine-tune only taught it to emit JSON.

Here is the check that removes that doubt. Take only the 71 tasks where the base model
**already** produced valid JSON. There was no formatting left to improve, so any gain there
is skill.

| On those 71 tasks | base | after fine-tuning |
|---|---:|---:|
| correct | 0 | **62** |

That is +0.873 against a headline of +0.909. Nearly all of the gain is capability, not format.

### What the base model actually gets wrong

It does not refuse. It rarely fails to produce JSON. It writes a perfectly formed tool call
carrying an identifier that does not exist, 62 times out of 110. It invents a deal id instead
of looking one up.

That is a grounding failure, and no amount of format training fixes it. A person reviewing
outputs would miss almost all of it, because the only wrong thing is an id nobody checks by eye.

## The DPO arm, which made it worse

I also ran DPO on the 2,180 preference pairs, on top of the fine-tuned model.

| | before DPO | after DPO |
|---|---:|---:|
| correct | 102 / 110 | **68 / 110** |
| invented identifier | 0 | 13 |
| parse failure | 0 | 2 |
| bad argument | 8 | 18 |

Paired delta -0.309, 95% CI [-0.391, -0.227]. One epoch, beta 0.1, learning rate 5e-5.

The error breakdown says what happened. Before DPO there were no invented identifiers and no
parse failures. Afterwards both came back. DPO undid the grounding habit that fine-tuning had
taught.

Learning rate 5e-5 is almost certainly too high for DPO on an already converged model. That is
a diagnosis I have not tested, so this is not evidence that DPO cannot work here. It is evidence
that DPO on top of an already converged model can damage it.

GRPO is unrun. I assumed there was no point, because at 93 percent accuracy nearly every group of
sampled answers would be entirely correct. That arithmetic was too pessimistic. With 8 generations
the chance all 8 are right is 0.909 to the 8th, about 0.47, so roughly half of all groups still
contain a failure and carry a real advantage signal. GRPO is viable here. It is thin, not absent,
and it is the next thing to run.

## The data

| | |
|---|---:|
| candidate traces generated | 3,300 |
| passed verification | 1,100 (33.3%) |
| train / valid / test | 872 / 108 / 110 |
| customer accounts per split | 180 / 23 / 22 |
| preference pairs | 2,180 |
| customers appearing in two splits | 0 |

A 33 percent yield is the useful part. Two out of three generated traces being wrong but
plausible is exactly the ratio that ruins a dataset built on plausibility. A high yield would
have meant the injected mistakes were not realistic.

Ten kinds of mistake were injected to test the checker itself: invented identifier, wrong
customer, wrong price book, wrong quantity, wrong stage, missing step, partial escalation,
unrequested extra action, wrong subscription quantity, and a discounted quote activated with no
approval. All ten caught. Zero escaped.

Of the 2,180 rejected rows, 406 were refused outright by a tool. **1,774 ran with no error at
all and ended in the wrong state.** Those 1,774 would pass any similarity-based quality check.

Setup: Qwen2.5-1.5B-Instruct in 4 bit, LoRA rank 16 alpha 32 on all linear layers, 3 epochs,
learning rate 2e-4, effective batch 16, max length 2048, prompt masked so loss falls only on
the answer. 872 examples, 4.97M tokens, one A100.

## What this does not measure

- The split is by customer, so no account appears twice, but **14.7% of training rows share a
  long word sequence with a test row** once ids are masked. So this measures unseen customers
  on familiar workflow shapes. It is not proof of handling something genuinely new.
- Training loss reached **0.01**, which is closer to memorising than learning. A harder test
  set would score below 100.
- The checker is **blind to wasted work**. Change every lookup argument to nonsense and the end
  state is unchanged, so the score stays at 110 out of 110. Checking the call sequence instead
  would catch it, but would also reject legitimate alternative solutions.
- 110 test tasks means about **plus or minus 5 points**. Nothing finer is claimed.
- The generator is a scripted planner, not a language model. Deliberate: with a planner the
  ground truth is known, so the checker is the only thing under test. Swapping a model in is
  one function.

## Three bugs worth knowing about

Each one survived a casual read of my own code. Each is now a test.

| Bug | What it looked like | The fix |
|---|---|---|
| Mutations silently did nothing | 91% yield, which was fiction. Real figure 33%. | Apply the mutation or fail loudly |
| Goal check instead of state check | 185 traces with unrequested actions counted as correct | Compare the whole before and after change |
| Standard decontamination | Deleted an entire task family, because one instruction template per family made every example look alike | Write varied phrasings instead of filtering |

## Run it

The data side needs no GPU and no API key. It rebuilds in about 20 seconds.

```
python build/generate.py                # candidates -> verified + rejected
python build/filter_splits.py           # dedup, decontaminate, split by account
python build/make_preference_pairs.py   # rejections -> preference pairs
python eval/harness.py --selftest       # check the grader before trusting it
python -m pytest tests/                 # 24 tests
```

Train and score a model:

```
pip install -r requirements.txt
python build/train_qlora.py inspect                                  # token and masking budget
python build/train_qlora.py train --out runs/lora-v1                 # needs CUDA
python build/train_qlora.py predict --out runs/lora-v1 --predictions preds.jsonl
python eval/harness.py --predictions preds.jsonl --split test
```

Or open `colab_train_eval.ipynb` on a free T4. It scores the base model, trains an adapter,
scores the same tasks again, and reports a paired bootstrap interval on the difference.

To continue past fine-tuning with DPO or GRPO:

```
python build/train_align.py --method dpo --sft-adapter <hub id or local path>
```

Both alignment stages need compute capability 8.0 or higher, so an A10G or A100. On a T4 the
TRL log-probability path refuses to run, which is a hardware limit and not a bug in this repo.

## Layout

```
build/
  crm_world.py              sandbox: objects, tools, effects, task generation
  generate.py               candidate generation and the verifier
  filter_splits.py          dedup, decontamination, account-disjoint splits
  make_preference_pairs.py  rejections -> preference pairs
  train_qlora.py            fine-tuning: inspect, train, predict
  train_align.py            DPO and GRPO from the fine-tuned checkpoint
eval/
  harness.py                executes predictions, Wilson intervals, paired bootstrap
tests/
  test_pipeline.py          24 tests
colab_train_eval.ipynb      train and eval on a free T4
train_eval_colab.py         the runner behind the notebook
```

## License

Apache 2.0. Everything here is synthetic. There is no proprietary or customer data anywhere
in this repository.
