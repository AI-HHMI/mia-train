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

Scoring is a wrapper over the control's tooling, which takes `--run`/`--tag` for exactly this, so
the two experiments score through one code path:

```bash
bash experiments/subpixel_decoder/score_checkpoint.sh 15000          # predict + score + figure
bash experiments/subpixel_decoder/score_checkpoint.sh 15000 --reuse  # re-score, no GPU
```

Artifacts are tagged `sp_*` against the control's `bp_*`, so they share the eval directory without
colliding.

## Reading it

**`boundary_accuracy` is the metric**, not `affinity_accuracy`. The target is ~83% positive, so
predicting "same object" everywhere scores 0.83 pooled while cutting nothing — the smoke run
demonstrated this precisely, reporting `affinity_accuracy = 0.718` and `boundary_accuracy = 0.0`
for a head that was still at its zero initialisation.

The head starts from the trivial constant predictor by design: `out` is zero-initialised, so the
first step's loss is exactly `ln 2 = 0.693` and no random gradient reaches an encoder that already
solves the task. The cost is that the head is quiet for the first step and, under warmup, a little
longer.

## What happened

Scored on the full val cube, sweeping the threshold on val only. **Both arms are compared at
matched encoder age**, since this one warm-starts from the control at 200k -- its step 50k is an
encoder that has seen 250k steps in total. Getting this wrong is easy and flattering: an early
comparison against the control's *stale* 160k checkpoint showed the sub-pixel head 12% ahead when
the step-matched control was actually ahead of it.

| | nERL | VOI | splits | mergers |
|---|---|---|---|---|
| control @ 220k | 0.3519 | 3.519 | 4,397 | 524 |
| control @ 250k | 0.3691 | 3.189 | 3,770 | 543 |
| sub-pixel @ 15k (enc 215k) | 0.3425 | 3.216 | 4,855 | 594 |
| **sub-pixel @ 50k (enc 250k)** | **0.4482** | **2.831** | **3,691** | **486** |

At matched encoder age the sub-pixel head is **+21.4% on nERL** and better on VOI and on both
error counts -- none of which was true at 15k, where it was behind on nERL and worse on both
errors. The slopes are the clearer signal: the control gained 4.9% over its last 30k steps, this
arm 30.9% over its last 35k. At 0.4482 it is 84% above BANIS-S (0.244) and roughly three quarters
of the way to the best published baseline (0.596).

**A confound to keep in view.** `max_steps` differs (100k vs 300k), so the cosine schedules are at
different points: at these checkpoints this arm runs at 1.69e-4 against the control's 4.83e-5, a
factor of 3.5. Part of the control's flatness is schedule rather than saturation, and removing the
confound properly needs a matched-schedule run. It seems unlikely to account for a 21% gap that
also shows up in both error counts, but it has not been ruled out.

**The seam risk did not materialise.** Measured as the ratio of mean absolute gradient across
block faces to the interior, the sub-pixel field is 1.097 / 1.011 / 1.001 on x / y / z, and its
total variation across the 16-voxel cycle (1.11-1.21) is *smaller* than the interpolating head's
(1.75-1.93) -- trilinear upsampling has its own periodic signature at grid nodes. The overlapping
kernel held in reserve has not been needed.

## Open questions

- **The schedule confound above.** A matched-schedule control (same `max_steps`) is the clean way
  to settle how much of the gap is the decoder and how much is learning rate.
- **Where it tops out.** Both arms are still training; the sub-pixel arm has 50k steps left and
  the control 25k. The slopes favour this arm heavily, but neither has converged.
- **Whether the loss is now the binding constraint.** This arm's training loss is ~40% below the
  control's (0.090 against 0.147), a much larger gap than the 21% in nERL, so the loss is buying
  less segmentation quality per unit than it used to. The width-1 boundary voxels that sharpness
  depends on are ~0.2% of the loss, which remains the obvious next suspect.
- **`boundary_accuracy` has no control baseline.** The metric postdates the control's launch, and
  a running job does not pick up new code, so the 0.92-0.94 this arm reports cannot be compared
  against the interpolating head. Any future control run gets it for free.

Two earlier mechanism hypotheses in this line of work -- mutex watershed, and the head being
unable to express boundary *contrast* -- were refuted by measurement. This one survived, but only
once the control was scored at matched encoder age; against a stale control it first appeared to
win for the wrong reason, and then to lose.
