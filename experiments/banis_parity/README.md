# Can a DINOv3 ViT reach the published BANIS baselines on NISB `base`?

`simmim_vs_direct` asked whether SSL pretraining helps. It does — but it also produced the first
real benchmark numbers, and they were far off the published baselines on the *same* 5 training
cubes:

| | nERL | VOI sum |
|---|---|---|
| best published baseline (`base`) | **59.6%** | 1.91 |
| BANIS-S (`base`) | **24.4%** | 3.46 |
| our arm B (Meta DINOv3 → fine-tune) | **2.6%** | 7.12 |
| our arm C (from scratch) | 0.08% | 8.65 |
| ground-truth affinities (pipeline control) | 100.0% | 0.00 |

The control matters: pushing ground-truth affinities through the identical scoring path returns
nERL exactly 1.0000 and VOI 0.0000, so the 2.6% is a model result, not an instrumentation
artifact. This experiment asks whether the gap is the *recipe* or the *architecture*.

## What changed, and why

Three things, all pointed at by measurements rather than intuition. See `finetune_256_long.toml`.

**1. The encoder's stem was frozen.** `layerwise_lr_decay = 0.9` compounds with
`patch_embed_lr_mult = 0.2` over 24 blocks, so `patch_embed` ran at 0.9²⁵ × 0.2 = 0.0144 of the
base LR — an effective **1.4e-06**, against BANIS' 1e-3 for every parameter. That is **696×**
slower, for **167×** fewer steps. The patch embedding is DINOv3's natural-image RGB kernel averaged
and spread over depth; it is the one layer that *must* adapt to grayscale EM texture, and it never
moved. Now both knobs are 1.0 and the base LR is 3e-4, so `patch_embed` trains at 3e-4 — a 209×
increase.

**2. The compute went into crop size instead of steps.** Measured, same algorithm and GPU:

| crop | tokens | s/step | activation memory |
|---|---|---|---|
| 256³ | 4,096 | **0.25** | 22 GiB |
| 512³ | 32,768 | 6.14 | 112 GiB (needs checkpointing) |

512³ costs **24× the time per step for 8× the voxels**, because attention is quadratic in tokens.
And the model demonstrably was not using the extra context: across an isolated ground-truth
boundary its predicted affinity dipped by 0.04, and even across a **24-voxel-wide** boundary — one
and a half patches, trivially resolvable at 16× upsampling — only by 0.084. BANIS uses 256³. At
256³ activation checkpointing is unnecessary too, which is a further ~30%.

**3. No augmentation.** BANIS runs slice-drop/shift, intensity, noise and affine. We ran none, on
5 cubes. Now `aug_rot = "inplane"`.

Why `"inplane"` and not the full group: miao's 48-transform `aug_rot` requires isotropic voxels,
and ours are 9×9×20 nm. An x↔z swap is shape-valid but semantically wrong — a neurite is
physically isotropic (~285–300 nm on every axis) yet spans 32 voxels in x and only 15 in z, so
swapping those axes produces an object shape that never occurs, and relabels a 20 nm neighbour
relationship as a 9 nm one. `"inplane"` draws from the 16 transforms that never exchange z with x
or y (8 flips × the x↔y swap), which *is* a genuine symmetry since x and y share a voxel size.
Implemented in miao on `feature/aug-rot-anisotropic` at the time; it lives in this repo now
([`src/data/augment.py`](../../src/data/augment.py)), since that branch never merged.

The three are changed **together**, deliberately. At 300k steps on 5 cubes, augmentation is not a
confound to isolate — it is what makes the longer run a fair test rather than an overfitting
demonstration.

## Everything held fixed

Same 5 training cubes, same val cube, same `affinity_seg` objective, same offsets (3× +1, 3× +10),
same ViT-L/16 architecture, same Meta initialisation, same cosine schedule, same global batch 8,
same evaluation. So this is head-to-head with the published `base` numbers.

## A dependency this run has that the run record does not capture

In-plane rotation is `[augment] rotate = "inplane"` in the .toml, implemented in
[`src/data/augment.py`](../../src/data/augment.py). It used to be miao's per-volume
`aug_rot`, which needed commit 8d41638 on `feature/aug-rot-anisotropic`; no miao pin is
required now.
miao is installed **editable**, so a run reads whatever branch is checked out in `~/projects/miao`
— and mia-train's provenance records its own commit, not miao's. Two consequences:

- Switching miao branches during the ~32 h run changes what the dataloader workers do.
- A `--resume` after a wall-time limit will fail config validation on `master`, where
  `aug_rot` is still a bool and `"inplane"` is not a valid value.

Merging the branch (or pinning miao non-editably) removes the hazard.

## Running it

```bash
bash experiments/banis_parity/submit.sh      # idempotent: resubmit to continue
bash experiments/banis_parity/tensorboard.sh # then open localhost:6006
```

`submit.sh` passes `--resume` and `bsub -r`, because 300k steps will not fit one wall-time window.

## Scoring it

`val/affinity_accuracy` is a poor guide here — it pools six channels against an ~83%-positive
target, so a field that predicts 0.7 everywhere scores well and is useless to connected components.
Use the real metrics:

```bash
bash experiments/banis_parity/score_checkpoint.sh 160000
```

That submits three chained jobs — affinities over the whole val cube (1 H100, ~35 min), then
nERL/VOI and a predicted-vs-ground-truth figure in parallel behind it. The two halves run in
different environments so that mia-train never acquires `funlib.evaluate`/`numba` and BANIS is
never imported next to a GPU; `--dry-run` prints the `bsub` lines without submitting.

```bash
score_checkpoint.sh 160000 --reuse                # re-score existing affinities, no GPU
score_checkpoint.sh 160000 --reuse --no-score \
    --origin "512 512 300"                        # just another figure, ~5 s
score_checkpoint.sh 200000 --logits "5 6 7"       # narrower threshold sweep
```

The underlying two stages are `~/projects/banis/mia_predict.py` and `mia_score.py`, callable
directly if a one-off needs an option the wrapper does not pass through.

## What happened

The gap was mostly the recipe. nERL on the full val cube, sweeping the threshold on val only:

| step | nERL | best VOI | vs previous |
|---|---|---|---|
| 20k | 0.1293 | | |
| 50k | 0.2329 | | +80% |
| 110k | 0.2815 | | +21% |
| 160k | **0.3045** (logit +6) | 3.702 (logit +5) | +8.2% |

For reference, the arm this replaced scored **0.0259**, and the published BANIS-S baseline on the
same `base` condition is 0.244. So the recipe fix — unfrozen stem, `layerwise_lr_decay = 1.0`,
256³ instead of 512³, and long training — was worth ~11× on its own, and the ViT is now above the
small baseline but still well short of the best published 0.596.

Two follow-ups were then measured rather than assumed, and both corrected an earlier guess:

- **Mutex watershed over the long-range channels is much worse here, not better** — 0.0141 against
  thresholded connected components' 0.4192 on an identical 512³ block, producing 173k segments. The
  argument for it was that the long-range channels separate better on average (+0.40 to +0.46 vs
  +0.35), but MWS uses repulsion as a *hard* constraint, and one false repulsive edge forbids a
  merge permanently. Average separation is not per-edge reliability. Kept as a diagnostic in
  `banis/mia_score_mws.py` (with a 6-case self-test), not wired into the routine path.
- **The trilinear head limits localization, not contrast.** Boundary contrast improved ~4× over the
  earlier arm at every ground-truth boundary thickness, and the output now spans [0.006, 0.999]
  rather than [0.044, 0.978] — so the "the head cannot express a boundary" reading was wrong. But
  the figures show predicted boundaries as 8–16 voxel bands where ground truth is 1–2 voxels, which
  is what one trilinear upsample from a 64-channel bottleneck would give, and it matches the
  measured 10:1 split-to-merge error ratio: wide cuts erode thin processes into fragments. A
  sub-pixel decoder (token → 16³ learned readout, as the MAE/SimMIM decoders already do) is
  therefore still the leading architectural change, on sharpness grounds.
