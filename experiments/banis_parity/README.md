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
Added to miao on `feature/aug-rot-anisotropic`.

The three are changed **together**, deliberately. At 300k steps on 5 cubes, augmentation is not a
confound to isolate — it is what makes the longer run a fair test rather than an overfitting
demonstration.

## Everything held fixed

Same 5 training cubes, same val cube, same `affinity_seg` objective, same offsets (3× +1, 3× +10),
same ViT-L/16 architecture, same Meta initialisation, same cosine schedule, same global batch 8,
same evaluation. So this is head-to-head with the published `base` numbers.

## A dependency this run has that the run record does not capture

`aug_rot = "inplane"` needs miao commit **8d41638** on branch `feature/aug-rot-anisotropic`.
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
# 1. affinities over the whole val cube (mia-train's env, 1 GPU)
python ~/projects/banis/mia_predict.py <run_dir> \
    --cube /groups/miaai/miaai/lmd-v0.0.1/nisb/train_100/val/seed100 \
    --out <out>/aff.zarr
# 2. instances + nERL/VOI (banisvenv, CPU, ~200 GB)
~/banisvenv/bin/python ~/projects/banis/mia_score.py <out>/aff.zarr \
    --skeleton .../val/seed100/skeleton.pkl --out <out>/scores.json
```

## What the result will mean

- **nERL climbs toward 24%** → the gap was the recipe, and the ViT is competitive.
- **nERL stays near 3%** → the recipe was not the binding constraint and the architecture is. Then
  the fix is a sub-pixel decoder (token → 16³ learned readout, as our MAE/SimMIM decoders already
  do) rather than one trilinear upsample from a 64-channel bottleneck, and/or mutex watershed over
  the long-range channels we currently discard at inference.

Either way it is decidable, which the previous round was not.
