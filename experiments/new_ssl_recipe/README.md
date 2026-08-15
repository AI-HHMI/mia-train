# Does a two-phase SSL recipe beat plain joint SSL, and does either beat no SSL at all?

`init_comparison` asked whether in-domain SimMIM helps at all, by putting an SSL arm next to a
directly-finetuned one. This asks a narrower question that the earlier design cannot answer: **how**
the SSL stage is run. A published domain-adaptation recipe (FINO, arXiv:2606.05107) claims the SSL
stage should start with the backbone **frozen**, so the parts that begin wrong — the patch embedding
and the pretext head — settle before they are allowed to push gradients into a pretrained encoder.
Every SSL run in this repo so far has trained everything jointly from step 0.

So: three arms, all starting from the released DINOv3 **ViT-B/16** checkpoint, differing only in
what happens between that checkpoint and the finetune.

**Nothing has run yet. This document is a pre-registration — it contains no results.**

## Arms

| arm | stage 1 | stage 2 | stage 3 | total |
|---|---|---|---|---|
| 1 control | — | interpolating, **50k** | sub-pixel, **50k** | 100k |
| 2 two-phase | SimMIM **100k**, backbone frozen for the first 10k | interpolating, **50k** | sub-pixel, **50k** | 200k |
| 3 joint | SimMIM **100k**, no freeze | interpolating, **50k** | sub-pixel, **50k** | 200k |

Configs, in submission order:

```
1a_dinov3_interp.toml       1b_dinov3_subpixel.toml
2a_ssl_twophase.toml        2b_ssl_twophase_interp.toml    2c_ssl_twophase_subpixel.toml
3a_ssl_joint.toml           3b_ssl_joint_interp.toml       3c_ssl_joint_subpixel.toml
```

`experiment_name` is `sslrec__` + the file stem, so run directories are
`/nrs/scicompsoft/orhane/mia-train-runs/sslrec__2a_ssl_twophase_<timestamp>/`.

The two finetune stages are the sequence `init_comparison` established: the **interpolating** head
first, then the **sub-pixel** head warm-started from that encoder. It is used here as machinery, not
as a question — all three arms run it identically, so it cancels.

### What each comparison isolates

- **arm 2 vs arm 3** isolates the **frozen warm-up**. The two SSL configs are identical apart from
  `freeze_backbone_steps = 10000` / `unfreeze_warmup_steps = 1000` in 2a, which 3a omits entirely
  (0 is the default and disables the mechanism). Same data, same masking, same 100k steps, same
  learning-rate curve outside the ramp.
- **arm 3 vs arm 1** isolates **SSL itself** — 100k steps of joint SimMIM against nothing at all,
  with the same released checkpoint on both sides and the same 50k + 50k finetune after it.
- **arm 2 vs arm 1** is the recipe end to end, and is only interesting decomposed into the two
  above.

Every arm starts from the same weights, so `pos_embed_rope_type = "superposition"` is used
throughout. `init_comparison` mixed `vanilla` into its cold arms because a from-scratch model has no
2D channel layout to preserve; here there are no cold arms, and the RoPE variant is not a variable.

## ViT-B, not ViT-L

`init_comparison` runs ViT-L/16 (303.2M). This runs **ViT-B/16 (85.7M)**: `embed_dim = 768`,
`depth = 12`, `num_heads = 12`. Roughly a third of the cost per step, which is the point — this
experiment asks a recipe question that needs three arms and eight stages, and it is worth more to
get an answer at all than to get it at the largest size.

**The consequence is that these numbers are not directly comparable to `init_comparison`'s.**
A `sslrec__` nERL and an `init__` nERL differ in model size, in step budget (50k + 50k against
150k + 100k), and in the SSL stage's starting point (`init_comparison` pretrains a cold encoder;
every arm here pretrains a DINOv3-initialised one). Read this experiment against its own control,
arm 1, and nothing else.

## Provenance of the two-phase recipe, and what was dropped

FINO (arXiv:2606.05107) is a metadata-guided self-supervised adaptation of vision foundation models
to a new domain. Its schedule is two-phase: a **short first phase with the backbone frozen**, which
trains only the input stem and the objective's heads, followed by joint training with the backbone
released behind a **1000-iteration linear learning-rate warmup** so it does not take a full-size
step on its first one. That is exactly the shape `[trainer].freeze_backbone_steps` and
`unfreeze_warmup_steps` implement here, and arm 2 is that schedule and nothing else.

**What did not carry over, stated plainly.** FINO's objective is DINO + iBOT with metadata-guidance
modules and a SIGReg term; none of that exists in this repo. The pretext task here is SimMIM, whose
entire head is one `nn.Linear(768, 4096)` — a single layer predicting a 16³ patch. So the thing
being warmed up is **the SimMIM head and the patch embedding**, not a stack of projection heads, and
the "protect the backbone from a randomly initialised head" argument is correspondingly weaker: a
3.1M-parameter linear layer is a much smaller perturbation than DINO/iBOT's prototype heads. Whether
the freeze still buys anything under a much lighter head is precisely what arm 2 vs arm 3 measures.
Borrowing a schedule from a paper whose objective is absent is a transplant, and it is being tested
as one.

**10k of 100k is a chosen round 10%, not a tuned value.** The source describes the phase as short
and does not report a sweep over its length, so there is nothing to copy. If arm 2 and arm 3 differ,
the length becomes worth tuning; if they do not, the honest conclusion is about 10%, not about all
possible warm-up lengths.

## Settings that matter

**The frozen phase is not cheap.** Only the model's parameters outside its input stem are frozen —
the patch embedding and the algorithm's head keep training — and the stem sits below every block, so
backward still traverses the whole stack to reach it. What the freeze saves is weight gradients and
optimizer state, not the pass. Budget arm 2's first 10k steps at close to full price.

**The unfreeze ramp applies to every parameter group**, not just the newly released ones. The head
has been training for 10k steps by then, and briefly slowing it too is a smaller distortion than
running two learning-rate schedules at once. The ramp is a multiplier on top of the ordinary decay,
so the decay curve is the same function of `step` whether or not a freeze is configured.

**The stage-2 learning rate is derived, not copied.** The two finetune stages are the two halves of
a single 100k linear plan (peak 3.0e-4, warmup 2000, `min_lr_ratio = 0.01`), split at step 50000:

- `lr = 1.5453e-4` is what that plan reaches at step 50000
- `min_lr_ratio = 0.0194` lands stage 2 on the same 3.0e-6 endpoint the plan would have reached
- `warmup_steps = 500` softens the step up, since stage 1 has already annealed to its own floor —
  the handoff is 3.0e-6 → 1.5453e-4, not genuinely continuous

The same formula reproduces `init_comparison`'s published **1.228e-4 / 0.0244** for its 150k + 100k
split, which is the check that these two numbers are arithmetic and not taste. Restarting stage 2 at
the full 3.0e-4 would be ~2x higher than anything the encoder saw in stage 1.

**No `[augment]` section anywhere.** This matches `init_comparison` arms 1–3 and deliberately not
its arm 4: augmentation is a separate variable and is held out. The training data config
(`nisb_base_256_aug.yaml`, byte-identical to `init_comparison`'s) still carries miao's
`aug_rot: "inplane"`, which is dataset-level and applies to all eight stages equally.

**Validation differs by stage type, on purpose.** The SSL stages are self-supervised and have no
labelled cube to score, so 2a and 3a validate SimMIM reconstruction on the same five training cubes
(`samples_per_epoch = 32`) — a training-progress readout, not a comparison number. The six finetune
stages validate on the held-out NISB base val cube (seed100). An SSL stage's val loss and a finetune
stage's val loss are different quantities; do not put them on one axis.

## Running it

Two prerequisites that `submit.sh` handles and a hand-run command does not.

**Arm 2 needs `[trainer].freeze_backbone_steps`,** which is what makes its first phase a frozen
warm-up. Confirm it is present before submitting — on a checkout without it, `2a` does not run
badly, it refuses to parse (`[trainer] has unknown key(s) ['freeze_backbone_steps',
'unfreeze_warmup_steps']`), which is the safe direction but only if someone reads the error.
Arms 1 and 3 are unaffected.

**The data config needs the pinned `miao`.** `nisb_base_256_aug.yaml` sets `aug_rot: "inplane"` per
volume, which the default `miao` rejects outright (`aug_rot Extra inputs are not permitted`), so
resolving `[data]` outside a submitted job needs the same pin `submit.sh` exports:

```bash
export PYTHONPATH=/nrs/scicompsoft/orhane/mia-train-scratch/miao-pinned/8d41638/src${PYTHONPATH:+:$PYTHONPATH}
```

`submit.sh` pins it deliberately rather than tracking `miao`'s main branch: a stray checkout there
once killed four live runs mid-epoch.

```bash
bash experiments/new_ssl_recipe/submit.sh              # every arm, every stage
bash experiments/new_ssl_recipe/submit.sh 2 3          # just the two SSL arms
bash experiments/new_ssl_recipe/submit.sh --smoke      # 20 steps of all eight stages, one GPU
QUEUE=gpu_h200 bash experiments/new_ssl_recipe/submit.sh 1   # override the queue choice
```

Each arm submits as a chain, every stage held on `done()` of the one before it. A stage cannot name
its predecessor's checkpoint at submission time — the run directory does not exist yet — so the
dependent configs carry a `PREV_CHECKPOINT` placeholder that the job resolves for itself, taking its
own arm's predecessor's newest run and its **numerically highest** checkpoint. Every stage is
submitted twice, to `gpu_h100` and `gpu_h200`, and the twins mutually exclude through
`init_comparison`'s `claim.sh` lock; the loser exits 42.

`--smoke` runs every stage for 20 steps on one GPU under `--output-root .../new_ssl_recipe/smoke`.
Two things it does that `init_comparison`'s does not: it rewrites `freeze_backbone_steps` to 2 (the
trainer rejects a freeze that outlasts the run, so arm 2's 10000 against `max_steps = 20` would
abort at startup), and it resolves `PREV_CHECKPOINT` from the smoke chain's own predecessor rather
than from a stand-in encoder — there is no trained ViT-B of this shape to borrow, every earlier
experiment here being ViT-L.

There is no `tensorboard.sh` in this directory yet; `experiments/init_comparison/tensorboard.sh` is
the pattern to copy. `boundary_accuracy` is the panel to read on the finetune stages — pooled
`affinity_accuracy` sits near the target's ~83% positive rate whatever the model does.

## Scoring

There is no wrapper here either. Call the shared script directly, naming the run and a tag so arms
cannot overwrite each other's affinities:

```bash
RUN=$(ls -dt /nrs/scicompsoft/orhane/mia-train-runs/sslrec__2c_ssl_twophase_subpixel_*/ | head -1)
bash experiments/banis_parity/score_checkpoint.sh 50000 --run "${RUN%/}" --tag sslrec_2c
```

After the `experiments/` reorganisation (`fb4afe7`), that script's dead NISB paths were updated: its
default `--cube` moved off the old `legacy/` layout and now reads the reorganised cubes under
`/groups/miaai/miaai/lmd-v0.0.1/dev/nisb` — specifically `train_100/val/seed100.zarr` — so no
`--cube` override is needed. Note that this is the `train_100` val cube while `[val_data]` here
trains against the `base` val cube; that split is `init_comparison`'s and `banis_parity`'s
convention, kept so scores stay comparable across experiments, but it does mean the val loss and the
nERL are measured on different volumes.

The SSL stages (2a, 3a) cannot be scored — they have no affinity head — so score **both finetune
stages of every arm**: the interpolating stage is the comparable-to-`banis_parity` number in shape
if not in scale, and the sub-pixel stage is what the second head adds.

## What each outcome would mean

- **2 > 3 > 1** → the recipe works and the frozen warm-up is a real part of it. The freeze is then
  worth tuning (length, and whether the patch embedding alone suffices), and worth trying on the
  finetune stages too, where the decoder is also randomly initialised.
- **2 ≈ 3 > 1** → SSL pays for itself but the freeze does not. The most likely reading is the one
  flagged above: SimMIM's single linear head perturbs the backbone too little for there to be
  anything to protect it from, and FINO's phase 1 earns its keep against DINO/iBOT heads
  specifically. That is a result about this transplant, not a refutation of the source.
- **3 ≈ 1 and 2 ≈ 1** → 100k steps of SimMIM on five cubes adds nothing on top of DINOv3, and the
  SSL budget should go to supervised steps instead. Note this is a *stronger* null than
  `init_comparison`'s arm 3 could produce, since there SimMIM had a cold encoder to improve on.
- **2 > 1 but 3 ≈ 1** → the SSL stage is only useful when it starts frozen, i.e. joint SimMIM was
  damaging the pretrained features early and the warm-up is what prevents it. The diagnostic is the
  SSL stages' own loss curves over the first few thousand steps.
- **3 < 1** → 100k steps of joint SimMIM actively degrades DINOv3's features. Check 2 in that case:
  if 2 ≈ 1, the freeze bought back exactly the damage, which is the sharpest possible version of the
  first result.

## What would make this inconclusive

- **`samples_per_epoch = 32` for validation.** 32 crops is a small sample; differences under ~2 sd
  between arms are sampling noise, not signal. Validation loss is for spotting divergence, not for
  ranking arms — and it has already, in `subpixel_decoder`, failed to identify the best checkpoint
  (the lowest val loss scored worse nERL than a later step). Rank on nERL from
  `score_checkpoint.sh`.
- **Comparing at unmatched training.** Arm 1's encoder has seen 50k or 100k supervised steps at
  scoring time; arms 2 and 3 have seen the same plus 100k SSL steps. That asymmetry is the
  experiment, but it means every arm must be scored at the *same finetune step* — 1b at 50k against
  2c at 50k, never 1b at 50k against 2c at whatever its newest checkpoint happens to be. The single
  most misleading thing in `subpixel_decoder` was scoring against a checkpoint that had since been
  overtaken: a 12% apparent win was a 3% loss once the control was scored at matched age.
- **Quoting old numbers instead of re-scoring.** Do not compare a `sslrec__` result to a figure
  written down in another README. Different model size, different budget, and possibly a different
  scoring cube revision. If a reference number is needed, re-score the reference checkpoint with the
  same script on the same day.
- **A confounded arm 2.** The freeze changes the effective learning-rate trajectory as well as which
  parameters move: for its first 11k steps arm 2 has a smaller trainable set and then a ramped rate.
  If 2 and 3 differ, "the freeze helped" and "arm 2 effectively had a gentler early schedule" are
  not separated by this design. A fourth arm with the ramp but no freeze would separate them, and is
  worth adding only if the gap is real.
- **One seed per arm.** `seed = 0` everywhere. A gap of the size these arms are likely to produce
  could be a seed draw; treat a single small gap as a reason to repeat the arm, not as an answer.
