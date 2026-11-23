# How this started

This began as interview preparation, not as a research project.

1. I opened the job description. Lots of words I knew, one phrase I did not.
2. Then the recruiter call: the work is for **Revenue Cloud**, so the model is for one product family.
3. So I read what Revenue Cloud is: options, price, quote, order, billing, invoice.
4. Then I hit **CPQ** — Configure, Price, Quote. Once I saw that, the job description made sense.
5. But I had nothing safe to practise on. So I built a small fake world.
6. Then the 16 tools, then the six kinds of job — each with the correct answer written down beforehand.
7. That is the trick: I can check an answer by **running** it and comparing what should have changed with what did.
8. Then I learned each method on data I understood: teaching by example with LoRA, then DPO, then GRPO.
9. The first DPO run made it worse — learning rate too high — so I lowered it and ran it again, and it came back.

---

## What "it came back" actually means

I want to be precise about step 9, because the difference matters.

| run | score |
|---|---:|
| base model | 0 / 110 |
| teaching by example (LoRA) | 102 / 110 |
| DPO, learning rate 5e-5 | **68 / 110** — it went backwards |
| DPO, learning rate 5e-6 | **102 / 110** — back to where it started |
| GRPO, binary reward | **102 / 110** — no change |

So the first DPO run damaged the model, and lowering the learning rate stopped the damage.
But it did not make the model better than it already was. It got back to 102 and stayed there.

That is the honest result, and I think it is the more interesting one.

## Why it went backwards, and why fixing it did nothing

DPO was optimising its own objective very well and that was the problem. Over one epoch:

| measured | start | end |
|---|---:|---:|
| how well it ordered the pairs it was trained on | 50% | 99.4% |
| gap between the good and bad answer it preferred | 0.02 | 3.28 |

It learned the preference data almost perfectly and the held-out score did not move at all.
The teaching stage had already taught the behaviour, so there was nothing left for the
preference stage to add — only things it could break.

Lowering the learning rate made it break nothing. It still added nothing.

## Why GRPO did nothing either

GRPO learns from its own attempts, so it needs the model to get some attempts wrong. On 8
attempts per task, I measured how often all 8 came out the same:

| | predicted before running | measured |
|---|---:|---:|
| groups with nothing to learn from | about 47% | **88.75%** |

The prediction assumed every task is equally hard. It is not. Most tasks the model solves
every single time, and a few it fails most times. When difficulty is spread like that,
almost every group is unanimous and gives no signal.

So the real finding is not "GRPO does not work". It is that **you cannot predict whether
GRPO has anything to learn from by looking at average accuracy. You have to look at how
success is distributed across tasks.**

## What I would do next

The benchmark is used up. The model gets 102 of 110 and the 8 misses are all the same
single weakness — it occasionally invents an identifier. Three things follow:

1. **Make the tasks harder, not the settings.** More steps per task, or train only on the
   families it actually misses.
2. **Build a harder test set.** A perfect score on a benchmark you built yourself is a
   ceiling of the benchmark, not of the model.
3. If I ever use a softer reward, say so out loud — it is a different experiment and
   easier to game.

## The parts I would say out loud before anyone asks

- The checker is blind to wasted work. Nonsense lookup arguments leave the end state
  unchanged, so the score stays 110/110.
- 14.7% of training rows share a long word sequence with a test row once ids are masked.
  This measures new customers on familiar workflows, not genuinely new behaviour.
- 110 tasks means about ±5 points. Nothing finer is claimed.
- GRPO is one seed and stopped at 80% of its first epoch, because it was slow.

---

*I started this because I read a job description and did not know one word in it. What
followed was a world small enough to check, tools that refuse to let a mistake pass, and
three ways of teaching a model on top of it. Some of it worked and some of it went
backwards. All of it is me learning, and I can show you every step.*
