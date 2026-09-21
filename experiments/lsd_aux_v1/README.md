# lsd_aux_v1: do local shape descriptors help the affinity head?

Sheridan et al. (2023) add an auxiliary target to affinity prediction: ten *local shape
descriptors* per voxel -- the Gaussian-weighted size, centre-of-mass offset and coordinate
covariance of the object around it -- and report that predicting them beside the affinities
sharpens the affinities, with the largest gains on the largest volumes. Their networks were U-Nets
with a limited field of view; the argument in the paper is that the descriptors force the network
to use all of it. Whether a ViT with global attention gains the same way is the question here.

The implementation is `[algorithm].lsd_sigma` on `affinity_seg` (`src/algorithms/affinity/lsd.py`,
the paper's MTLSD form: one head, shared up to the last 1x1 convolution). Prediction and scoring
are unchanged, the artifact carries the six affinity channels only, so every number below is an
affinity-segmentation number.

## Arms

| arm | config | differs from the control by |
|---|---|---|
| control | `experiments/subpixel_decoder/subpixel_256.toml`, already trained (`subpixel_decoder__subpixel_256_20260811_143947`, checkpoints to 95k) and scored | -- |
| `mtlsd_s80` | `mtlsd_s80.toml` | `lsd_sigma = 80.0` nm, the paper's FIB-25 window: (8.9, 8.9, 4.0) voxels on 9x9x20 nm |
| `mtlsd_s80_bgzero` | `mtlsd_s80_bgzero.toml` | as above, plus `lsd_background = "zero"`: background voxels are supervised towards the all-zero descriptor instead of being left out of the loss |

Both arms also set `defer_image_ops = true`, which moves resampling and augmentation onto the
device (the standing rule for new configs) without changing what they compute. Everything else --
encoder, warm start from the banis_parity encoder at 200k, sub-pixel head, data, augmentation,
schedule, global batch 8 -- is the control's, and the two configs are generated from the control's
file so they cannot drift from it:

    diff experiments/subpixel_decoder/subpixel_256.toml experiments/lsd_aux_v1/mtlsd_s80.toml

The second arm exists because of where the control's errors are: predicted boundaries are 8-16
voxels wide against a 1-2 voxel truth and splits outnumber mergers 10:1. Under the reference's
default the descriptor loss says nothing at a membrane; under `"zero"` every membrane carries a
target that changes abruptly across it.

## Running and scoring

    bash experiments/lsd_aux_v1/submit.sh mtlsd_s80 --smoke     # 20 steps, 1 GPU
    bash experiments/lsd_aux_v1/submit.sh mtlsd_s80             # 8 GPUs, resumable
    bash experiments/lsd_aux_v1/score_checkpoint.sh 50000 mtlsd_s80

Runs, job logs and evaluations live under `/nrs/scicompsoft/orhane/mia-train-experiments/lsd_aux_v1/`.
Scoring is the control's path (`banis_parity/score_checkpoint.sh`: predict on one GPU, BANIS
scoring on the val cube, threshold swept on val only), with the arm name as the artifact tag.

**Compare at matched steps**, as `subpixel_decoder/README.md` does: both arms warm-start the same
encoder, so step *N* of an arm is the control's step *N*. The control was trained on `gpu_h100`
and `submit.sh` defaults to that queue; `QUEUE=gpu_b300` is a legitimate choice when the H100 queue
is deep, but keep every arm's *prediction* on one architecture -- predictions from the same
checkpoint differ between H100 and B300 (instance counts, pq), and that difference is not small
next to the effects being measured.

## What to read

- `loss_lsd` and the four `lsd_mse_*` curves (offset, variance, pearson, size), which say whether
  the head is learning the descriptors at all and which statistic it finds hard.
- `boundary_accuracy` against the control's at the same step: the affinity metric this target is
  supposed to move.
- nERL, VOI, splits and mergers at matched steps, from the scoring script.

## Cost of the target

Built on the device, per sample, after the connected-components split (B300, `sigma = 80 nm`;
`probes/lsd_target_cost/` under the experiment's /nrs directory, `results.txt`):

| crop | objects | `lsd_downsample = 2` (default) | `lsd_downsample = 1` |
|---|---|---|---|
| 128^3 | 32-35 | 4-6 ms | 21-30 ms |
| 256^3 | 49-107 | 38-75 ms | 290-630 ms |

Peak extra device memory is 0.9 GB at 128^3 and about 2.5 GB at 256^3. The first implementation
used `conv3d` with 1-D kernels and was 14x slower (`results_conv3d.txt`); each separable pass is now
a banded matrix product.

## Not in this round

- A sigma sweep (the paper's window is a FIB-25 value; 60 or 120 nm are the obvious neighbours).
- Writing the predicted descriptors out for viewing, and any post-processing that reads them.
- The auto-context variants (ACLSD, ACRLSD), which need a two-network cascade.
