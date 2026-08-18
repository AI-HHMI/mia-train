# Can a 3.4%-trainable fine-tune match a full one on NISB?

Two arms, four stages, one difference: whether the encoder is adapted through rank-16 low-rank
deltas or trained outright. Everything else -- data, crops, augmentation, batch, schedule, decoder
sequence, seed -- is identical, and the configs are generated from shared blocks so that identity is
mechanical rather than careful. The only keys that differ between `1a` and `2a`, and between `1b` and
`2b`, are `experiment_name` and the six under `[lora]`.

**The experiment has not run. This document is a pre-registration and contains no results.**

## Arms

| arm | stage a (50k) | stage b (50k) | trainable |
|---|---|---|---|
| **1 LoRA** | interpolating head, encoder adapted at rank 16 | sub-pixel head, adapter continues | **10.6M of 312.9M (3.4%)** |
| **2 full** | interpolating head, all params | sub-pixel head, all params | 306.6M (100%) |

```
1a_lora_interp.toml   ->  1b_lora_subpixel.toml
2a_full_interp.toml   ->  2b_full_subpixel.toml
```

Both start from the released DINOv3 **ViT-L/16** (`dinov3_vitl16_pretrain_lvd1689m`), inflated to a
16³ single-channel stem. `experiment_name` is `lvf__` + the file stem, so run directories are
`/nrs/scicompsoft/orhane/mia-train-runs/lvf__1a_lora_interp_<timestamp>/`.

Arm 1's trainable budget: adapters 6.29M (96 of them), stem 4.20M, LayerNorm + LayerScale 0.15M,
cls/storage/mask tokens 6.1K, the RoPE depth gate 1 scalar.

## Why this is worth 4 runs

It is the gate on a larger plan. The intended follow-up is *SSL with LoRA* -- adapting the encoder on
unlabelled in-domain data before fine-tuning, on the hypothesis that full-model SSL erases the
pretrained prior. That plan is only interpretable if a LoRA fine-tune is competitive to begin with:
if arm 1 lands far below arm 2, then a rank-16 adapter is capacity-limited for dense affinity
prediction on this task, and no SSL stage upstream of it can be read as helping or hurting. This
experiment costs no SSL compute and answers that first.

It is also a useful result on its own. If arm 1 matches arm 2, LoRA becomes the cheap default for
every sweep this project still owes -- decoder variants, `depth_scale` initialisation, `train_100`,
augmentation -- at 3.4% of the optimizer state.

## The caveat that decides how to read a null result

**Both arms run at the same learning rate (3e-4 peak), and that is a known handicap for arm 1.**
Standard LoRA practice uses a higher rate than a full fine-tune, because `B` starts at zero and the
delta is scaled by `alpha/rank`. Matching the rate is what keeps this a single-variable comparison --
raising it for arm 1 alone would mean the arms differ in two ways and neither result could be
attributed.

So the readings are asymmetric:

- **arm 1 ≈ arm 2, or better** -- conclusive. LoRA is competitive here, at a rate that was not even
  tuned for it. Proceed to the SSL arms.
- **arm 1 well below arm 2** -- *not* conclusive, and must not be reported as "LoRA does not work on
  this task". The designed follow-up is a rate sweep for arm 1 alone (1e-3 is the usual ~3x, and
  rank 32 doubles the subspace at 12.6M adapter params), which is 1-2 more stage-a runs and can
  reuse arm 2 as its control.

Nothing else about the arms is asymmetric, so a null result costs one cheap follow-up rather than the
plan.

## Choices worth defending

**`targets = ["attn_qkv", "attn_proj", "mlp"]`, not attention alone.** LoRA's canonical recipe adapts
attention only, and that is the wrong call here. Weight-space drift measured over an existing 100k
supervised fine-tune of this exact architecture: `mlp.fc1` reaches cosine **0.667** to its initial
value while `attn.qkv` reaches **0.714** -- this task rewrites the FFN slightly *more* than it
rewrites attention. An attention-only adapter would constrain the half that moves less. Costs
6.29M against 2.36M for qkv+proj.

**The stem trains in full, in both arms.** `patch_embed` is an inflated RGB kernel now reading
single-channel EM: there is no pretrained answer in it to preserve, and a rank-`r` correction to a
wrong stem is not the tool for the job. Throttling it cost ~11x nERL in `banis_parity`. 4.20M params,
and the largest non-adapter share of arm 1's budget.

**`rope_embed.depth_scale` is trainable unconditionally.** Superposition RoPE computes
`angles_spatial + depth_scale * angles_depth`, and `depth_scale` is one scalar starting at zero. At
zero the positional encoding cannot distinguish one z-slice from another. The model declares it
through `BaseModel.lora_required_trainable()`, so no `[lora]` switch can freeze it.

**`[trainer].zero_weight_decay_on_lora` is left at its default (true).** The adapter's factors are
both 2-D and would otherwise take the full `weight_decay = 0.05`. Decaying `lora_b` -- which starts
at zero -- shrinks the *delta*, which is a pull back toward the pretrained weights: an L2-SP penalty
at a strength this config never states. Arm 1 tests a rank constraint, not a rank constraint plus an
accidental anchor.

**Stage b carries no `[init].skip`, and this is a correction.** The equivalent stage in
`new_ssl_recipe` and `init_comparison` carries `skip = ["rope_embed."]`, inherited from the 2D→3D
load in stage a where it is necessary (the 2D rotary tables have a different length). At 3D→3D every
rope tensor has the same shape, so the skip transfers nothing and *discards* something:
`rope_embed.depth_scale` is a learned parameter, and skipping it resets the depth gate to zero,
throwing away whatever stage a learned about the z axis. Verified 2026-08-18 -- with the skip, stage
b starts at `depth_scale = 0.0000` where stage a ended at `0.3700`. **The prior experiments' stage-2
and stage-3 configs have this bug**; their encoders re-learned the gate from scratch at each head
change. Omitted here entirely rather than narrowed to the two buffers, so a genuine disagreement in
the rope tensors raises instead of being silently tolerated.

**Two phases rather than one.** Three single-phase attempts at the sub-pixel head from an encoder
that had never seen EM all collapsed onto the trivial "everything is connected" predictor. The
interpolating head has never had that problem (0.3691 nERL from these released weights in
`banis_parity`), and the best result in the repo -- 0.4941 -- came from the sub-pixel head
*warm-started* from an interpolation-trained encoder. Both arms run that sequence identically, so it
cancels.

**50k + 50k as one 100k plan, split at the head change.** Stage b's peak `1.545e-4` is exactly where
a single 100k linear schedule (peak 3e-4, warmup 2000, `min_lr_ratio = 0.01`) sits at step 50000, and
its `min_lr_ratio = 0.0194` lands it on that plan's 3e-6 endpoint. The same arithmetic reproduces
`new_ssl_recipe`'s published 1.53e-4 / 0.0196 for its 100k + 100k split. Inherited caveat: stage a
declares `max_steps = 50000` and therefore anneals all the way to 3e-6 before stopping, so the
handoff is 3e-6 → 1.545e-4 rather than genuinely continuous; the 500-step warmup softens it.

## Comparability with earlier experiments

The arms are compared **against each other**, and that comparison is internal and clean. Reading
`lvf__` numbers next to `sslrec__` or `init__` ones needs one adjustment stated:

- **Schedule length.** This is 50k + 50k. `new_ssl_recipe`'s finetunes are 100k + 100k and
  `init_comparison`'s are 150k + 100k. Arm 2 here is otherwise the same recipe as
  `new_ssl_recipe`'s arm 1, so its curve should track that run's over the first 50k steps up to the
  schedule difference -- which makes it a cheap consistency check that nothing has drifted.
- **The depth-gate correction above** applies only here, so arm 2 is not quite a replica of
  `new_ssl_recipe`'s arm 1 at the stage boundary.

## Running it

```bash
# both arms, twin-submitted to gpu_h100 and gpu_h200 -- whichever frees first runs each stage
bash experiments/lora_vs_fullft/submit.sh

bash experiments/lora_vs_fullft/submit.sh 1          # the LoRA arm only
bash experiments/lora_vs_fullft/submit.sh --smoke 1  # 20 steps on one GPU, validates the chain
QUEUE=gpu_h200 bash experiments/lora_vs_fullft/submit.sh   # force one generation
```

Each stage is submitted twice, once per GPU generation; `claim.sh` gives the pair an atomic lock and
the loser exits 42, releasing its node in seconds. Both twins share an `experiment_name` and pass
`--resume`, so either can continue the same run directory. Stage b resolves its predecessor's
checkpoint inside the job, after the `done()` dependency fires.

Smoke first if anything in the stack has changed. It exercises the one thing unique to arm 1 that a
unit test cannot: the adapter surviving a DCP save and reload across the stage boundary.

```bash
bash experiments/lora_vs_fullft/tensorboard.sh          # both arms, stage-paired legend
bash experiments/lora_vs_fullft/tensorboard.sh 6008 --smoke
```

## Scoring

`banis_parity/score_checkpoint.sh` takes `--run` and `--tag`, so it needs no copy here:

```bash
bash experiments/banis_parity/score_checkpoint.sh 50000 \
  --run /nrs/scicompsoft/orhane/mia-train-runs/lvf__1b_lora_subpixel_<timestamp> \
  --tag lvf1b
```

**`banis/mia_predict.py` must be the version from 2026-08-18 or later.** Earlier versions rebuild the
model from `resolved_config.json`'s `[model]` section alone, which produces a *plain* encoder -- and
DCP loads into a state dict, skipping keys the template does not ask for, silently. Arm 1 would then
be scored with its adapters dropped, i.e. as the un-adapted released checkpoint, landing near the
baseline it started from. The current version applies `[lora]` on rebuild and additionally refuses to
load any checkpoint holding `model.*` tensors it has nowhere to put.

nERL on the val cube (seed100) is the selection metric; the test cube (seed101) is scored once, at
the end, and appears in no config here.
