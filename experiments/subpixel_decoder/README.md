# Is the trilinear head what limits boundary sharpness?

`banis_parity` closed most of the gap to the published NISB baselines by fixing the training
recipe, reaching **0.3045 nERL** at 160k steps against BANIS-S's 0.244. What it did not fix is
*where* the model puts a boundary. Predicted affinities show boundaries as 8–16 voxel bands where
ground truth is 1–2 voxels, and the error profile is lopsided in the way that predicts: **4,055
splits against 184 mergers**, a 10:1 ratio. Wide cuts erode thin processes into fragments.

This arm changes one thing — how patch tokens become voxels — and holds everything else at the
control's value.

## What is different

| | control (`banis_parity`) | this arm |
|---|---|---|
| decoder | `interpolate` | `subpixel` |
| encoder init | released DINOv3 ViT-L/16 | **the control's own encoder at step 200k** |
| max_steps | 300k | 100k |
| everything else | | identical |

The interpolating head produces all sub-token detail from one `F.interpolate` followed by a
`Conv3d(64→64, k=3)` at voxel resolution. Its 3-voxel receptive field acts on an already-smooth
field, so structure finer than the 16-voxel token spacing has almost no way to appear.

The sub-pixel head gives each token a learned readout of its own 16³ block
(`ConvTranspose3d` with `kernel_size == stride`, which is exactly a per-token linear map), then
refines at voxel resolution with two narrow convolutions.

## Why this is not a more expensive model

Counterintuitively it is a **cheaper** one, because cost at voxel resolution is driven by
positions, not width. Measured param counts; FLOPs and activations are analytic, batch 1 at 256³:

| | params | forward MAC | activations |
|---|---|---|---|
| interpolating head | 176,646 | 1.86 T | ~13.3 GB (fp32) |
| **sub-pixel head** | **17,053,590** | **0.30 T** | ~5.8 GB fp32 / ~2.9 GB bf16 |
| *encoder, for scale* | *306 M* | *~2.0 T* | |

The wide arithmetic moves onto the 16³ token grid, where there are 4096 positions rather than
16.78M. A secondary consequence: the fp32 constraint on the interpolating head goes away with it,
since a `kernel == stride` transposed convolution accumulates over channels only, not over eight
spatial neighbours.

## Running it

```bash
bash experiments/subpixel_decoder/submit.sh --smoke   # 20 steps, 1 GPU, ~70 s
bash experiments/subpixel_decoder/submit.sh           # idempotent: resubmit to continue
bash experiments/subpixel_decoder/tensorboard.sh      # this arm beside the control
```

Scoring reuses the control's tooling, which takes `--run`/`--tag` for exactly this:

```bash
bash experiments/banis_parity/score_checkpoint.sh 20000 \
    --run /nrs/scicompsoft/orhane/mia-train-runs/subpixel_decoder__subpixel_256_<stamp> --tag sp
```

## Reading it

**`boundary_accuracy` is the metric**, not `affinity_accuracy`. The target is ~83% positive, so
predicting "same object" everywhere scores 0.83 pooled while cutting nothing — the smoke run
demonstrated this precisely, reporting `affinity_accuracy = 0.718` and `boundary_accuracy = 0.0`
for a head that was still at its zero initialisation.

The head starts from the trivial constant predictor by design: `out` is zero-initialised, so the
first step's loss is exactly `ln 2 = 0.693` and no random gradient reaches an encoder that already
solves the task. The cost is that the head is quiet for the first step and, under warmup, a little
longer.

## What the result will mean

- **`boundary_accuracy` climbs and nERL beats 0.3045** → the head was the constraint, and the
  decoder is worth keeping on efficiency grounds alone.
- **Sharper boundaries but nERL flat or worse** → look for seams. Each token's block is decoded
  independently, and a systematic discontinuity every 16 voxels would cut objects at block faces,
  causing exactly the fragmentation this is meant to fix. The figures will show it as a grid. The
  escalation is an overlapping kernel (`kernel_size = 2 × patch`, the support trilinear
  interpolation itself uses), at 8× the expansion parameters.
- **Nothing moves** → capacity was not the binding constraint, and the next suspect is the loss:
  the width-1 boundary voxels that sharpness depends on are ~0.2% of it, so the head may simply
  not be paid enough to use what it now has.

A caveat carried over from the control: two earlier mechanism hypotheses in this line of work
(mutex watershed, and the head being unable to express contrast) were refuted by measurement. This
one is stated the same way and should be held to the same standard.
