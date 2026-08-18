# Does a two-phase SSL recipe beat plain joint SSL, and does either beat no SSL at all?

`init_comparison` asked whether in-domain SimMIM helps at all, by putting an SSL arm next to a
directly-finetuned one. This asks a narrower question that the earlier design cannot answer: **how**
the SSL stage is run. A published domain-adaptation recipe (FINO, arXiv:2606.05107) claims the SSL
stage should start with the backbone **frozen**, so the parts that begin wrong — the patch embedding
and the pretext head — settle before they are allowed to push gradients into a pretrained encoder.
Every SSL run in this repo so far has trained everything jointly from step 0.

So: three arms, all starting from the released DINOv3 **ViT-L/16** checkpoint, differing only in
what happens between that checkpoint and the finetune.

**The experiment has not run. This document is a pre-registration and contains no results for it.**
The one thing that has run is an earlier ViT-B configuration of it, which was aborted; that is a
result about ViT-B and it is recorded below.

## Arms

| arm | stage 1 | stage 2 | stage 3 | total |
|---|---|---|---|---|
| 1 control | — | interpolating, **100k** | sub-pixel, **100k** | 200k |
| 2 two-phase | SimMIM **200k**, backbone frozen for the first 20k | interpolating, **100k** | sub-pixel, **100k** | 400k |
| 3 joint | SimMIM **200k**, no freeze | interpolating, **100k** | sub-pixel, **100k** | 400k |

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
  `freeze_backbone_steps = 20000` / `unfreeze_warmup_steps = 1000` in 2a, which 3a omits entirely
  (0 is the default and disables the mechanism). Same data, same masking, same 200k steps, same
  learning-rate curve outside the ramp.
- **arm 3 vs arm 1** isolates **SSL itself** — 200k steps of joint SimMIM against nothing at all,
  with the same released checkpoint on both sides and the same 100k + 100k finetune after it.
- **arm 2 vs arm 1** is the recipe end to end, and is only interesting decomposed into the two
  above.

Every arm starts from the same weights, so `pos_embed_rope_type = "superposition"` is used
throughout. `init_comparison` mixed `vanilla` into its cold arms because a from-scratch model has no
2D channel layout to preserve; here there are no cold arms, and the RoPE variant is not a variable.

## ViT-L/16, and what is and is not comparable to `init_comparison`

All eight stages are DINOv3 **ViT-L/16 — 303.2M as released, 306.6M as built here**, the difference
being the stem, which becomes a 16³ single-channel kernel: `embed_dim = 1024`, `depth = 24`,
`num_heads = 16`. That is `init_comparison`'s architecture exactly. It was not the original plan —
see "A first attempt at ViT-B" below for the measurement that forced it.

**So the model is no longer a reason these numbers cannot be read against `init_comparison`'s.**
Same architecture, same 256³ crops, same global batch 8, same decoder sequence. Two things still
differ, and both are worth naming before anyone puts a `sslrec__` nERL next to an `init__` one:

- **Schedule length.** The finetune here is 100k + 100k; `init_comparison`'s is 150k + 100k. Its
  encoders have seen 50k more supervised steps by the end of stage 1, and the derived stage-2
  learning rates differ accordingly (1.228e-4 there, 1.53e-4 here).
- **Where the SSL stage starts.** *Every* arm here starts from the released DINOv3 checkpoint,
  including 2a and 3a, so the SSL stages adapt a pretrained encoder. `init_comparison`'s SSL arm
  pretrains a cold one, and its arm 1 is from scratch throughout. There is no from-scratch arm
  here at all.

The control that answers this experiment's own question is arm 1, which is inside it. A
cross-experiment comparison answers a different question — about schedule length and about SSL on a
pretrained versus a cold encoder — and should be stated as such rather than as a rerun.

## A first attempt at ViT-B, and why it was abandoned

The experiment was first configured with DINOv3 **ViT-B/16 (85.7M)** — `embed_dim = 768`,
`depth = 12`, `num_heads = 12` — for the obvious reason: three arms and eight stages at roughly a
third of the cost per step, on the argument that it is worth more to get an answer at all than to
get it at the largest size. It was launched on 2026-08-15 and aborted. **The ViT-B initialisation
was buying approximately nothing.**

Arm 1 (pretrained ViT-B, interpolating head) against both extremes of `init_comparison`, on val
`boundary_accuracy` at matched steps:

| step | arm 1 here: ViT-B, **pretrained** | `init_comparison` arm 1: ViT-L, **from scratch** | `init_comparison` arm 2: ViT-L, **pretrained** |
|---|---|---|---|
| 5000 | 0.2444 | 0.2407 | 0.7294 |
| 10000 | 0.3212 | 0.3734 | 0.8063 |
| 20000 | 0.4063 | 0.4048 | 0.8063 |
| 30000 | 0.4915 | 0.5000 | 0.9009 |

A *pretrained* ViT-B tracks a *from-scratch* ViT-L almost exactly over 30k steps, while a pretrained
ViT-L is already at 0.73 by step 5000 — a level the ViT-B run does not reach by 30k. Same data, same
target, same head, same recipe.

**This was not a loading bug.** What was checked, and what each check rules out:

- the resolved-config diff against `init_comparison`'s arm 2 showed only the intended differences
- `long_range` and `split_disconnected` are left at their defaults here and set explicitly there,
  and the defaults match those values byte for byte — so both runs train an identical target
- `load_pretrained` reported `kept_initial=[]`, `unused=[]`, `mismatched=[]` for **both** ViT-B and
  ViT-L: nothing stayed at its random initialisation, nothing in the checkpoint went unused, nothing
  was shape-mismatched and silently skipped
- `init_weights()` runs inside `__init__`, i.e. *before* the load, so there is no path by which the
  loaded weights are overwritten afterwards
- the run's own step-5000 checkpoint still matched the released ViT-B weights at cosine similarity
  0.97–0.999 per tensor, which shows the initialisation survived into training rather than being
  correct only at step 0

**The learning rate is not the explanation either.** The gap is already ~3x at step 5000, and at
that point the 50k schedule this arm ran and `init_comparison`'s 150k schedule are at nearly the
same rate — 2.81e-4 against 2.94e-4. A 5% difference in rate does not turn 0.73 into 0.24.

**What was not established: why.** The run was stopped, not diagnosed, and the mechanism is
unknown. Two hypotheses were written down and *neither was tested*:

- DINOv3's ViT-B is **distilled from ViT-7B**, while its ViT-L is closer to directly trained. A
  distilled student may carry less that survives a 2D→3D inflation plus a large domain shift. This
  is a plausible story, not a measurement.
- **ViT-B may want a different learning rate.** 3.0e-4 was tuned for ViT-L and nothing was retuned
  when the width and depth were cut. That the *early* gap is not a schedule artefact (above) does
  not mean 3.0e-4 is the right peak rate for this model.

So the honest scope of this result is: the released DINOv3 ViT-B/16 checkpoint, inflated to 3D and
finetuned with ViT-L's recipe, transferred no better than a from-scratch ViT-L on this task. It is
not a claim about ViT-B in general.

**The metric is noisy.** Validation uses `samples_per_epoch = 32`, so individual points move by
several points between evaluations — the repeated 0.8063 in the table is that, not a plateau. The
conclusion rests on the trend across four checkpoints and on the from-scratch comparison, not on any
single value, and it is a 3x gap rather than a marginal one.

## Provenance of the two-phase recipe, and what was dropped

FINO (arXiv:2606.05107) is a metadata-guided self-supervised adaptation of vision foundation models
to a new domain. Its schedule is two-phase: a **short first phase with the backbone frozen**, which
trains only the input stem and the objective's heads, followed by joint training with the backbone
released behind a **1000-iteration linear learning-rate warmup** so it does not take a full-size
step on its first one. That is exactly the shape `[trainer].freeze_backbone_steps` and
`unfreeze_warmup_steps` implement here, and arm 2 is that schedule and nothing else.

**What did not carry over, stated plainly.** FINO's objective is DINO + iBOT with metadata-guidance
modules and a SIGReg term; none of that exists in this repo. The pretext task here is SimMIM, whose
entire head is one `nn.Linear(1024, 4096)` — a single layer predicting a 16³ patch. So the thing
being warmed up is **the SimMIM head and the patch embedding**, not a stack of projection heads, and
the "protect the backbone from a randomly initialised head" argument is correspondingly weaker: a
4.2M-parameter linear layer is a much smaller perturbation than DINO/iBOT's prototype heads. Whether
the freeze still buys anything under a much lighter head is precisely what arm 2 vs arm 3 measures.
Borrowing a schedule from a paper whose objective is absent is a transplant, and it is being tested
as one.

**20k of 200k is a chosen round 10%, not a tuned value.** The source describes the phase as short
and does not report a sweep over its length, so there is nothing to copy. If arm 2 and arm 3 differ,
the length becomes worth tuning; if they do not, the honest conclusion is about 10%, not about all
possible warm-up lengths.

## Settings that matter

**The frozen phase is not cheap.** Only the model's parameters outside its input stem are frozen —
the patch embedding and the algorithm's head keep training — and the stem sits below every block, so
backward still traverses the whole stack to reach it. What the freeze saves is weight gradients and
optimizer state, not the pass. Budget arm 2's first 20k steps at close to full price.

**The unfreeze ramp applies to every parameter group**, not just the newly released ones. The head
has been training for 20k steps by then, and briefly slowing it too is a smaller distortion than
running two learning-rate schedules at once. The ramp is a multiplier on top of the ordinary decay,
so the decay curve is the same function of `step` whether or not a freeze is configured.

**The stage-2 learning rate is derived, not copied.** The two finetune stages are the two halves of
a single **200k** linear plan (peak 3.0e-4, warmup 2000, `min_lr_ratio = 0.01`), split at step
100000. Stage 2 continues the tail of that one plan rather than restarting a schedule of its own:

- `lr = 1.53e-4` is what that plan reaches at step 100000
- `min_lr_ratio = 0.0196` lands stage 2 on the same 3.0e-6 endpoint the plan would have reached
- `warmup_steps = 500` softens the step up, since stage 1 has already annealed to its own floor —
  the handoff is 3.0e-6 → 1.53e-4, not genuinely continuous

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
trainer rejects a freeze that outlasts the run, so arm 2's 20000 against `max_steps = 20` would
abort at startup), and it resolves `PREV_CHECKPOINT` from the smoke chain's own predecessor rather
than from a stand-in encoder from another experiment. The chain's `done()` dependency is what makes
that safe: the predecessor has written its step-20 checkpoint before the next stage starts.

`tensorboard.sh` serves every arm at once, rebuilding its symlink tree on each launch so stages that
start later appear without a restart:

```bash
bash experiments/new_ssl_recipe/tensorboard.sh          # all arms, port 6006 (probes past a busy one)
bash experiments/new_ssl_recipe/tensorboard.sh 6007 --smoke   # the 20-step runs instead
```

`boundary_accuracy` is the panel to read on the finetune stages — pooled `affinity_accuracy` sits
near the target's ~83% positive rate whatever the model does, so it looks healthy for a model that
has learned nothing. On the SSL stages `loss` compares only against the other SSL arm, and `val`
there is a progress readout on the training cubes, so a train/val gap is not generalisation. A kink
in 2a's loss at step 20k is the freeze boundary, not an instability.

## Scoring

There is no wrapper here either. Call the shared script directly, naming the run and a tag so arms
cannot overwrite each other's affinities:

```bash
RUN=$(ls -dt /nrs/scicompsoft/orhane/mia-train-runs/sslrec__2c_ssl_twophase_subpixel_*/ | head -1)
bash experiments/banis_parity/score_checkpoint.sh 100000 --run "${RUN%/}" --tag sslrec_2c
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
- **3 ≈ 1 and 2 ≈ 1** → 200k steps of SimMIM on five cubes adds nothing on top of DINOv3, and the
  SSL budget should go to supervised steps instead. Note this is a *stronger* null than
  `init_comparison`'s arm 3 could produce, since there SimMIM had a cold encoder to improve on.
- **2 > 1 but 3 ≈ 1** → the SSL stage is only useful when it starts frozen, i.e. joint SimMIM was
  damaging the pretrained features early and the warm-up is what prevents it. The diagnostic is the
  SSL stages' own loss curves over the first few thousand steps.
- **3 < 1** → 200k steps of joint SimMIM actively degrades DINOv3's features. Check 2 in that case:
  if 2 ≈ 1, the freeze bought back exactly the damage, which is the sharpest possible version of the
  first result.

## What would make this inconclusive

- **`samples_per_epoch = 32` for validation.** 32 crops is a small sample; differences under ~2 sd
  between arms are sampling noise, not signal. Validation loss is for spotting divergence, not for
  ranking arms — and it has already, in `subpixel_decoder`, failed to identify the best checkpoint
  (the lowest val loss scored worse nERL than a later step). Rank on nERL from
  `score_checkpoint.sh`.
- **Comparing at unmatched training.** Arm 1's encoder has seen 100k or 200k supervised steps at
  scoring time; arms 2 and 3 have seen the same plus 200k SSL steps. That asymmetry is the
  experiment, but it means every arm must be scored at the *same finetune step* — 1b at 100k against
  2c at 100k, never 1b at 100k against 2c at whatever its newest checkpoint happens to be. The
  single most misleading thing in `subpixel_decoder` was scoring against a checkpoint that had since
  been overtaken: a 12% apparent win was a 3% loss once the control was scored at matched age.
- **Quoting old numbers instead of re-scoring.** Do not compare a `sslrec__` result to a figure
  written down in another README. Different budget, possibly a different scoring cube revision, and
  — for anything older than `init_comparison` — a different model. If a reference number is needed,
  re-score the reference checkpoint with the same script on the same day.
- **A confounded arm 2.** The freeze changes the effective learning-rate trajectory as well as which
  parameters move: for its first 21k steps arm 2 has a smaller trainable set and then a ramped rate.
  If 2 and 3 differ, "the freeze helped" and "arm 2 effectively had a gentler early schedule" are
  not separated by this design. A fourth arm with the ramp but no freeze would separate them, and is
  worth adding only if the gap is real.
- **One seed per arm.** `seed = 0` everywhere. A gap of the size these arms are likely to produce
  could be a seed draw; treat a single small gap as a reason to repeat the arm, not as an answer.
