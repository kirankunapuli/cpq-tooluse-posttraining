# Results

Two alignment runs, both scored by the same harness on the same 110 held-out tasks, both
comparing against their own starting point so the delta is paired.

| file | run | before | after | delta | 95% CI |
|---|---|---|---:|---:|---:|---|
| `dpo_lr5e6.json` | DPO, 1,744 pairs, 1 epoch, beta 0.1, lr 5e-6 | 102/110 | 102/110 | 0.000 | [0.000, 0.000] |
| `grpo_ckpt15.json` | GRPO, 200 prompts x 8 generations, binary reward, lr 1e-6, checkpoint 15 | 102/110 | 102/110 | 0.000 | [0.000, 0.000] |

The starting point in both files is 102/110 with 8 `invalid_argument` errors, which reproduces
the published merged-bf16 SFT score. A delta of exactly 0.000 with a CI of [0.000, 0.000] means
both runs marked the same 102 tasks correct; it is not a rounding artifact.

## Reproduce

```
python build/train_align.py --method dpo \
  --sft-adapter kirankunapuli/cpq-sft-qwen1.5b \
  --lr 5e-6 --epochs 1 --beta 0.1 --out runs/dpo_lr5e6.json

python build/train_align.py --method grpo \
  --sft-adapter kirankunapuli/cpq-sft-qwen1.5b \
  --num-generations 8 --max-completion-length 256 \
  --limit-train 200 --lr 1e-6 --save-steps 5 --out runs/grpo_200.json
```

The DPO run takes about 45 minutes on an RTX 3090. The GRPO run takes 1 to 3 hours depending
on how the allocator behaves; `--resume` continues from the newest checkpoint.

## Reading the GRPO file

`grpo_ckpt15.json` records the trainer's own diagnostics for the two logged points. The number
that matters is `frac_reward_zero_std`: 0.8875 at step 10 and 0.875 at step 20, against a
binomial estimate of about 0.47. That gap is the finding.
