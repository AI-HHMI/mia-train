# Does pretraining help on NISB, and do EM augmentations?

`simmim_vs_direct` asked the first of these with a ViT-L, a trilinear decoder and a recipe we now
know was wrong in several ways. This asks it again with everything learned since:

- the **sub-pixel decoder**, which beat the trilinear head by 39% nERL at matched encoder age and
  is also ~6x cheaper per step (see [`../subpixel_decoder/`](../subpixel_decoder/))
- the **sub-pixel head left open at initialisation** (`decoder_zero_init_output = false`) --
  see "A false start" below
- **uniform learning rate** -- no layerwise decay, no patch-embedding multiplier. The original
  experiment gave the patch embedding an effective LR of 1.4e-6, 696x below the baseline's, which
  is most of why its numbers were an order of magnitude off
- **linear decay** to `min_lr_ratio = 0.01`, so a run that stops early has still annealed

## Arms

Two phases per arm: the **interpolating head first**, then the **sub-pixel head** warm-started from
that encoder. All arms are DINOv3 ViT-L/16 at 256³, global batch 8, linear decay to
`min_lr_ratio = 0.01`, uniform learning rate.

| arm | phase 1 | phase 2 | RoPE |
|---|---|---|---|
| 1 | random init, interpolating, 150k | sub-pixel, 100k | vanilla (axial) |
| 2 | DINOv3 LVD-1689M, interpolating, 150k | sub-pixel, 100k | superposition |
| 3 | SimMIM 100k → interpolating 150k | sub-pixel, 100k | vanilla (axial) |
| 4 | DINOv3, interpolating, 150k, **augmented** | sub-pixel, 100k, augmented | superposition |

RoPE differs by design: a released 2D checkpoint says nothing about a third axis, and superposition
keeps the 2D channel layout intact while adding a gated depth angle, so the pretrained weights
transfer. A from-scratch model gets the axial form, where depth owns its own third of the channels.

## Why two phases

Three single-phase attempts at the sub-pixel head from a cold encoder all failed the same way:
training affinity accuracy tracked the batch's positive rate to three decimal places -- the trivial
"everything is connected" predictor -- with a sharp dip around step 400 and no recovery. The
apparent flailing between 0.60 and 0.85 was the per-batch label statistics, not optimisation noise.

The interpolating head has never had this problem: from these same released weights it reached
**0.3691 nERL** (`banis_parity`). And the best result on this task, **0.4941 nERL**, came from the
sub-pixel head *warm-started* from an interpolation-trained encoder (`subpixel_decoder`). So this
experiment reproduces that sequence deliberately rather than hoping the head trains cold.

It also puts `decoder_zero_init_output = true` back in its element. Zeroing the output convolution
makes the head emit a constant, which protects an encoder that already solves the task from a random
head's meaningless gradients -- correct for a phase-2 warm start, and a handicap for a cold one,
where it also zeroes the gradient reaching the encoder.

**Measured along the way, and not adopted:** torch initialises `ConvTranspose3d` from
`fan_in = out_channels * kernel_volume`, which is right for an overlapping kernel. With
`kernel == stride` each output voxel is fed by exactly one input position, so the true fan-in is
`hidden` -- 256x smaller at patch 16, leaving the weights 16x too small, the signal through the
expansion attenuated 27x, and the gradient reaching the encoder 20x weaker than intended.
Correcting it did **not** fix cold-start training, so it was reverted to keep the head as it was
when it produced 0.4941. It remains a real latent issue worth revisiting if the head is ever
trained cold again.

## Augmentation (arm 4)

BANIS trains with augmentations we had never used. Four are now implemented in
[`src/data/augment.py`](../../src/data/augment.py), configured by an `[augment]` section, at
BANIS' own default magnitudes -- both codebases present images in [0, 1], so the numbers transfer
unchanged:

| | value | what it models |
|---|---|---|
| `drop_slice_prob` | 0.05 | a lost section: the image goes blank, **the labels do not** |
| `shift_slice_prob` / `shift_magnitude` | 0.05 / 10 | imperfect section alignment |
| `intensity` (`mul`/`add`) | 0.1 / 0.1 | acquisition brightness and contrast drift |
| `noise_scale` | 0.5 | sensor noise -- severe, up to half the dynamic range, on half of samples |

Each is gated by a coin flip before its own parameters apply, so a per-slice probability of 0.05
reaches roughly 2.5% of sections. That double gate is copied from the reference so its numbers can
be used as written.

**One deliberate divergence.** The reference shifts the image and leaves the labels in place,
desynchronising them by up to 10 voxels. Here both move together. Training against targets that no
longer describe the image is a strange thing to ask for, and boundary localisation -- the thing the
sub-pixel decoder just bought -- is the first thing it would cost.

Affine warping is the fifth augmentation and is **not** implemented: the reference gets it from
MONAI, which this repo will not depend on, so it needs a torch-native `affine_grid`/`grid_sample`
with nearest-neighbour label interpolation. Worth adding only if the other four move the number.

Augmentation is attached to the **training** dataset alone; the engine never wraps `[val_data]`, so
no config key can quietly change what a validation number means.

## Running it

```bash
bash experiments/init_comparison/submit.sh              # every arm, both phases
bash experiments/init_comparison/submit.sh 2 4          # just those arms
bash experiments/init_comparison/submit.sh --smoke      # 20 steps of all nine stages
QUEUE=gpu_h200 bash experiments/init_comparison/submit.sh 3   # override the queue choice
bash experiments/init_comparison/tensorboard.sh
```

Each arm submits as a chain, every stage held on `done()` of the one before it. A stage cannot name
its predecessor's checkpoint at submission time -- the run directory does not exist yet -- so the
dependent configs carry a `PREV_CHECKPOINT` placeholder that the job resolves for itself, taking the
predecessor's newest run and its **numerically highest** checkpoint. That is what makes a whole
chain safe to submit in one command.

`--smoke` runs every stage for 20 steps on one GPU. Phase-2 stages have no predecessor then, so
their placeholder is pointed at an existing trained ViT-L encoder; `skip = ["rope_embed."]` is what
lets that work regardless of which RoPE variant the stand-in used, since those buffers are derived
rather than learned.

## Scoring

```bash
bash experiments/init_comparison/score_checkpoint.sh 2b_dinov3_subpixel 100000
```

A wrapper over [`../banis_parity/score_checkpoint.sh`](../banis_parity/score_checkpoint.sh); the
stage name picks the run directory and tags artifacts so stages cannot overwrite each other.

Score **both phases** of each arm: phase 1 is the interpolating-head result, directly comparable to
`banis_parity`, and phase 2 is what the sub-pixel head adds on top.

**Compare at matched training, and re-score the reference.** The single most misleading thing in the
`subpixel_decoder` experiment was scoring against a checkpoint that had since been overtaken -- a
12% apparent win was really a 3% loss once the control was scored at matched encoder age. Note too
that the best validation loss did not identify the best checkpoint there (85k had the lowest val
loss, 100k scored higher nERL); with `samples_per_epoch = 32`, val differences under ~2 sd are
sampling noise.

## What the result will mean

- **2 > 1** → natural-image pretraining transfers to EM, as the earlier ViT-L experiment suggested
  (0.0215 vs 0.0008 there, though at a scale where everything was broken).
- **3 > 2** → in-domain SSL beats out-of-domain pretraining, and the SSL budget is worth spending.
- **3 ≈ 1** → SimMIM on five cubes is not enough signal, and the SSL stage is not paying for itself.
- **4 vs 2** isolates the augmentations alone. If they help, the gap is the price we have been
  paying for not having them; if they hurt, the likeliest cause is `noise_scale = 0.5` on a task
  whose remaining errors are 8:1 splits, since noise pushes toward predicting boundaries.
