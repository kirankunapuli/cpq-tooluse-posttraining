# Training data you can check by running it

Most synthetic tool-call data is accepted because it looks right. The JSON parses, the tool
names are real, the ids look like ids. None of that means the call did the right thing.

This repo generates tool-call data where acceptance is mechanical. Replay the model's calls
against a sandbox, and keep the example only if the resulting change matches exactly what the
task asked for. Everything else is rejected, and the rejections turn out to be useful too.

A 1.5B model trained on that data went from **0 correct out of 110** held-out tasks to
**100 out of 110**.

If you want the short version of why this exists and what the results mean, read
[STORY.md](STORY.md). It is nine steps and one honest table.

Dataset: [kirankunapuli/cpq-tooluse-verified](https://huggingface.co/datasets/kirankunapuli/cpq-tooluse-verified)

## Where this sits in post-training

Post-training is everything you do to a base model after pre-training. The methods below are the
usual ladder, and this repo touches three of the four. What separates them is the kind of feedback
each one needs, and that is where most of the engineering effort actually goes.

| Stage | What it is | How it works | What this project did | Result |
|---|---|---|---|---|
| **SFT**<br>supervised fine-tuning | The model copies correct answers | Loss lands on the answer tokens only, over traces the verifier accepted | 872 verified traces, QLoRA rank 16 on Qwen2.5-1.5B | 0/110 to **100/110** |
| **DPO**<br>direct preference optimisation | The model learns from pairs: same prompt, one better answer, one worse | Raise the probability of the chosen answer, lower the rejected one. No reward model needed | 2,180 pairs, produced free by the verifier's rejections | 102/110 to **68/110** at lr 5e-5; **back to 102/110** at lr 5e-6 |
| **GRPO**<br>group relative policy optimisation | The model learns from its own attempts | Sample a group of answers per prompt, score each, use the group average as the baseline instead of a value network | 200 prompts x 8 generations, binary verifiable reward | **102/110**, no change |
| **RLVR**<br>RL with verifiable rewards | Reinforcement learning where the reward is a program, not a person | Reward 1 when the replay matches the declared effects, 0 otherwise. Nothing to label | The verifier already is that reward function | it is what makes GRPO possible here |

The ladder is meant to go SFT, then preference optimisation, then RL. SFT teaches a behaviour, DPO
shifts it toward what you prefer, GRPO explores beyond what your data covers. Running them in order
is normal. Assuming each one helps is not. Here SFT did all the work, and both alignment stages
left the score exactly where they found it: DPO after its learning rate was corrected, GRPO because
almost no sampled group carried a signal.

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

Learning rate 5e-5 is almost certainly too high for DPO on an already converged model. That was
a diagnosis I had not tested, so at that point it was not evidence that DPO cannot work here.

I have since tested it. Re-running the same 1,744 pairs, one epoch, beta 0.1, at learning rate
5e-6:

| | before DPO | after DPO |
|---|---:|---:|
| correct | 102 / 110 | **102 / 110** |

Paired delta 0.000, 95% CI [0.000, 0.000]. The learning rate was the cause of the regression,
and lowering it removed the damage. It did not produce a gain.

The internal numbers say why. DPO optimised its own objective almost to completion — the margin
between chosen and rejected grew 190x, and the fraction of pairs it ordered correctly went from
50% to 99.4%. Held-out accuracy did not move. The probability it assigns to the correct answers
stayed flat at about -0.2 per token, meaning it never lost confidence in them: the failure mode
of the 5e-5 run never engaged. Reported honestly, DPO here avoided harm without buying anything,
because the teaching stage had already saturated this task set.

## GRPO, which also changed nothing

GRPO is now run. The reward is the verifier itself: 1.0 when the completion replays to the
declared end state, 0.0 otherwise. 200 training prompts, 8 generations each, learning rate 1e-6.

Result at checkpoint 15: 102/110 before, 102/110 after. Paired delta 0.000, CI [0.000, 0.000].

The reason is measurable, and it corrects the estimate I made above. I argued that at 0.909
per-sample accuracy the chance all 8 samples are correct is about 0.47, so roughly half of
all groups would still contain a failure and carry advantage signal. The trainer reports the
true figure:

| | estimated | measured |
|---|---:|---:|
| groups with zero reward variance | ~47% | **88.75%** |

The binomial estimate assumes one accuracy shared by every task. In reality difficulty is
uneven: most training tasks are solved on every attempt, while a minority are failed on most.
Under that spread, unanimous groups are far more common than the mean accuracy suggests.

So the conclusion is not that GRPO cannot work here, but that **mean accuracy is the wrong
statistic for predicting whether GRPO has anything to learn from. The per-task success-rate
distribution is the right one.** With 88.75% of groups unanimous, only about one group in nine
carried a gradient, and at KL pinned near 1e-6 it changed no answer.

The honest summary of all three stages: teaching by example took the model from 0 to 102 of 110,
and both alignment stages left that number untouched. The bottleneck is the task set, not the
learning rate, and not the choice of algorithm.

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

## The three components that are not the adapter

The adapter is the smallest part of this system, so three supporting components ship with
their own measurements, plus one measurement of a component that does not ship.

**The scope gate** (`build/scope_gate.py`) stops the model answering anything that is not
this domain. Three layers: fixed patterns for injection and authority claims, two
classifiers whose probabilities are averaged, and a reject-option band that asks a question
instead of guessing. One classifier reads hashed character 3 to 5-grams and word features,
the other reads `BAAI/bge-small-en-v1.5` embeddings; both are logistic regression.
Evaluated on 240 utterances drawn from **eight template families held out of training**:

| classifier | in scope acted on | out of scope refused | leaked | probes | reject band |
|---|---:|---:|---:|---:|---|
| character n-grams only | 120 / 120 | 120 / 120 | 0 | 11 / 12 | [0.695, 0.815] |
| embeddings only | 118 / 120 | 119 / 120 | 0 | 10 / 12 | [0.365, 0.485] |
| **mean of the two** (shipped) | **120 / 120** | **120 / 120** | **0** | **11 / 12** | **[0.535, 0.655]** |

The mean ties the n-gram channel on accuracy, so accuracy is not the reason to ship it. The
two bands are the same width, so neither is safer by construction. What the mean buys shows up
only when the arms disagree. On the probe "draft me an email to the customer explaining the
price rise", the n-gram arm scores 0.536 and refuses, the embedding arm scores 0.650 and acts,
and the mean of 0.593 falls inside the shipped band, so the gate asks one question instead of
picking a side. The encoder is already resident for retrieval, so that second opinion costs one
forward pass. Embeddings alone is the weakest arm and the only one that loses a probe.

Masking entities before classification, so an unseen customer name becomes a
placeholder, moved in-scope recall from 108 to 120. Without it a real customer whose
name the model had not seen could have their own quote refused.

**The rule index** (`build/retrieval.py`) is the one channel that uses text search. BM25
over 31 policy documents, a character n-gram channel for typos and morphology, a dense
channel over bge-small embeddings, all fused by reciprocal rank, then an optional
cross-encoder. 38 labelled queries:

| arm | R@1 | R@3 | MRR |
|---|---:|---:|---:|
| BM25 only | 0.895 | 0.974 | 0.941 |
| BM25 + char n-grams, RRF | 0.947 | 1.000 | 0.974 |
| BM25 + dense, RRF | 0.921 | 1.000 | 0.961 |
| **all three, RRF** (shipped) | **0.974** | **1.000** | **0.987** |
| all three + coverage reranker | 0.895 | 1.000 | 0.943 |
| all three + cross-encoder reranker | 0.974 | 1.000 | 0.987 |

The coverage reranker loses three queries and is wrong on all three, because term coverage
cannot cross a synonym: "which model made this quote" has to reach the document about
adapter version pinning, and no lexical overlap gets it there. The dense channel is what
fixes that class of query. The cross-encoder ties fusion and changes exactly two answers:
it fixes "can I approve my own quote" and breaks "how do I know which model made this
quote", where it prefers a passage about order lines to the version pinning document. Same
score, one traded for the other, so both rerankers stay in the code, out of the shipped
path, with the number next to them.

faiss is used above 2000 documents and skipped below it. At 32 documents the exhaustive
inner product is a 32x384 matrix multiply, and loading faiss next to torch aborts the
process with OMP #15 because both link libomp.

**The messy-input parser** (`build/input_normalizer.py`) is the front door. Every other
file here assumed a clean request; this one starts from
`quote 3 hundred prod phon x 512 gb titanum, 20 pct off, bellows collage`. Resolution runs
in three tiers, cheapest first, and each tier records how it resolved so the trace can show
the reasoning:

| tier | what it does | why it is not a guess |
|---|---|---|
| 1, exact | normalised form or a known alias. `512` becomes `512GB`, `1.2k` becomes 1200 | nothing is inferred, the alias table is what is matched |
| 2, fuzzy | edit distance above a threshold. `prod phon x` becomes `PROD-PHONE-X`, `lytware` becomes `Litware Inc` | identifiers and names only, never numbers |
| 3, ambiguous | two candidates above threshold, or none. Nothing is chosen | the slot is flagged and the caller asks |

| check | result |
|---|---:|
| unit checks in the file | 33 / 33 |
| messy requests that fully resolve | 10 / 10 |
| slots resolved to a wrong value | 0 |
| requests that must be questioned, resolved silently | 0 / 4 |

The dangerous error is not a refusal, it is a confident wrong resolution:
`PROD-PHON-X` resolving to PROD-PHONE-X is fine, resolving to PROD-HUB-Z would price the
wrong product in silence. Two real bugs came out of writing it. Number search read
attribute values as quantities, so `512 gb` became a request for 512 units, until spans
already claimed by a capacity, a percentage or an identifier were protected. And account
matching compared the whole sentence against the whole legal name, diluting `lytware` to
0.706 against `Litware Inc`; it compares words to words now.

## A decision model, measured where it helps and where it does not

[Laya](https://huggingface.co/convaiinnovations/laya) is a non-autoregressive System 1 decision
model: a ModernBERT encoder plus a decision head, `pip install laya`, no text generated. It answers
typed questions with calibrated probabilities. `noul` gives a calibrated P(true) for a yes or no
statement; `choice` gives a distribution over options with a conformal prediction set. It is
trained with RL against a strictly proper scoring rule, so an honest probability is the only way it
scores.

`build/laya_gate.py` measures it on the gate, on the same 240 held-out utterances and the same 12
probes as the ensemble, with the regex layer in front of every arm:

| arm | in scope | out of scope | leaked | free-text probes |
|---|---:|---:|---:|---|
| Laya zero-shot, one `noul` question | 120 / 120 | 120 / 120 | 0 | 10 / 12, 1 leak |
| Laya, head fitted on confirmed answers, temperatures calibrated | 120 / 120 | 120 / 120 | 0 | 11 / 12, 1 leak |
| **character n-grams plus embeddings** (shipped) | **120 / 120** | **120 / 120** | **0** | **11 / 12, 0 leaks** |

All three tie on the structural half. They separate on the 12 probes, which nobody trained on.
Fitting a head on 720 confirmed answers took **897 seconds on CPU** with no gradient step and no
GPU, and 1,392 seconds on an earlier run while the machine was busy. Both are slower than the model
card's 34 seconds for 300 answers on 8 threads, and both are jobs that fit in a release window
rather than a training budget.

`build/laya_resolve.py` measures it on the front door, where the question is which product and
which customer a sloppy sentence means. There it loses:

| | Laya `choice` | the parser |
|---|---:|---:|
| product named correctly, 8 resolvable requests | 4 / 8 | 8 / 8 |
| customer named correctly | 4 / 8 | 8 / 8 |
| named wrongly | 0 | 0 |
| ms per slot on CPU | 282 | under 1 |

`prod phon x`, `lytware` and `5OO` are character-level noise. An encoder normalises that away;
an explicit edit distance on the identifier keeps it. It is safe rather than accurate, because it
declines instead of naming a wrong one, and a string problem stays a string problem.

So the decision model belongs at the gate, and at the reviewer where the question is a calibrated
yes or no about a plan. It does not belong where the question is which of four identifiers an OCR
error was reaching for. The gate ships with both paths: Laya when the model is present, the
ensemble when it is not, since the fallback answers in 0.1 ms and needs no download.

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

## Bugs worth knowing about

Each one survived a casual read of my own code. Each is now a test.

| Bug | What it looked like | The fix |
|---|---|---|
| Mutations silently did nothing | 91% yield, which was fiction. Real figure 33%. | Apply the mutation or fail loudly |
| Goal check instead of state check | 185 traces with unrequested actions counted as correct | Compare the whole before and after change |
| Standard decontamination | Deleted an entire task family, because one instruction template per family made every example look alike | Write varied phrasings instead of filtering |
| A number is a number until you know what it is attached to | `512 gb` read as a request for 512 units | Protect spans already claimed by a capacity, a percentage or an identifier, and refuse to match a quantity inside one |
| Comparing a sentence to a legal name | `lytware` scored 0.706 against `Litware Inc`, so a one character typo fell under the threshold | Compare words to words; a corporate suffix is not evidence about which company someone meant |

## Run it

The data side needs no GPU and no API key. It rebuilds in about 20 seconds.

```
python build/generate.py                # candidates -> verified + rejected
python build/filter_splits.py           # dedup, decontaminate, split by account
python build/make_preference_pairs.py   # rejections -> preference pairs
python eval/harness.py --selftest       # check the grader before trusting it
python -m pytest tests/                 # 36 tests
```

Two scripts exist to keep the claims honest. `build/verify_all.py` reproduces every number in the
write-up against the artefact that produces it, including the write-up file itself, and exits
non-zero when a quoted value drifts. `build/demo_pipeline.py` pushes one typing-mistake request
through the gate, the parser, the price lookup, the planner and the two checks in one run:

```
python build/verify_all.py                       # 76 assertions, no GPU
python build/demo_pipeline.py                    # one request, all five stages
python build/demo_pipeline.py --break price      # the same plan, one stale number
python build/demo_pipeline.py --scope            # the gate refusing, zero tool calls
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
  scope_gate.py             three layer domain gate, two classifiers, calibrated band
  retrieval.py              three channel retrieval, RRF, optional cross-encoder
  input_normalizer.py       messy request -> slots in three tiers, or a question
  demo_pipeline.py          one request through all five stages, plus defect variants
  verify_all.py             every quoted number, asserted against the artefact
  _model_cache.py           the two small encoders, loaded once
eval/
  harness.py                executes predictions, Wilson intervals, paired bootstrap
tests/
  test_pipeline.py          36 tests, runs on a bare interpreter
colab_train_eval.ipynb      train and eval on a free T4
train_eval_colab.py         the runner behind the notebook
STORY.md                    how this started, and what the results mean in plain words
```

## License

Apache 2.0. Everything here is synthetic. There is no proprietary or customer data anywhere
in this repository.
