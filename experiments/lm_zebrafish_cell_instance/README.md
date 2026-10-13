# An affinity model for lm_zebrafish_cell_instance

mia-evals' `lm_zebrafish_cell_instance` task is cell instance segmentation in one frame of a light-sheet
zebrafish time series (Betzig lab, crop-001: 48 frames of 152 × 508 × 1466 voxels, z, y, x, at
200 × 108 × 108 nm). Frame 10 is reported (293 proofread cells). Frames 1 and 25 (317 and 296 cells)
fit the task's size filter, and here they are also the training data. No frame is held out for
validation (2026-10-12).

The experiment trains one recipe on those two frames at two windows, with and without the
serial-section EM augmentations:

| arm | window (x, y, z voxels) | physical extent | tokens | EM augmentations |
|---|---|---|---|---|
| `w496_em` | 496 × 496 × 144 | 53.6 × 53.6 × 28.8 µm | 8,649 | dropped + shifted sections, p = 0.05 each |
| `w496_noem` | 496 × 496 × 144 | | | off |
| `w288_em` | 288 × 288 × 144 | 31.1 × 31.1 × 28.8 µm | 2,916 | dropped + shifted sections, p = 0.05 each |
| `w288_noem` | 288 × 288 × 144 | | | off |

What bounds the window: each axis is a multiple of the 16-voxel patch and smaller than the frame,
because miao's `bounding_box` must exceed the window. z = 144 is therefore the most a 152-slice
frame allows, and 496 the most its 508 rows allow. x = y because `rotate = "inplane"` swaps them,
and `rotate = "none"` would drop the flips too. 512³ cannot fit: a whole frame (113 M voxels) is
smaller than one 512³ window (134 M). The median cell's bounding box is 100 × 93 × 165 voxels
(z, y, x; frame 10), the 95th percentile 152 × 210 × 342.

## One recipe

The configs start from `large_inputs_nisb__w512_5cube`'s, which is gary_comparison arm 5a:
- the released DINOv3 ViT-L/16 (LVD-1689M), inflated to 3D, with global attention in every block;
- `affinity_seg` with the UNETR decoder, attention scores capped at 1e5;
- batch 1 per rank × 8 ranks, AdamW at 3e-4 with linear decay, and its augmentation.

Every arm starts from the released weights, not from that run's checkpoint, which was trained on NISB EM.
Changed at the user's request (2026-10-12):
- the data and the window;
- no `[val_data]`;
- 100k steps, with a 5k-step warmup and a checkpoint every 10k.

One further change follows from the window. The head decodes undivided (`decode_chunks = 1`, as in
large_inputs_nisb's 256³ arms), because a 496 × 496 × 144 crop has a quarter of 512³'s voxels.

`make_configs.py` generates the four configs and `data/train.yaml`, so the arms differ only in the
window and two augmentation probabilities. `diff <(grep -v '^#' w288_noem.toml) <(grep -v '^#' w496_em.toml)`
shows exactly those lines.

## Data

`data/train.yaml` is mia-evals' fit split (`configs/lm_zebrafish_cell_instance/data/fit.yaml` at
mia-evals 3e6cd5a), rewritten in x, y, z order. `affinity_seg` defines its targets in that order, and
miao reads `resolutions` and `bounding_box` in `output_axes` order, so both are rewritten too.

Frame 1's box starts past the strip that its label's re-registration left unlabelled (z = 0, y < 5,
x < 5). About a fifth of each frame, chiefly the lower left, was left unannotated on purpose and is
stored as 0. The models learn it as background, which is also how the scorer treats it in frame 10.

## miao

The store holds the frames as t, c, z, y, x. Pinning one frame per volume entry (`fixed_axes`) needs
miao 0.4.2. myvenv has 0.4.1, and other experiments' jobs still run on it. So every job puts a
read-only export of `~/projects/miao` at 6b0533a (miao 0.4.2) first on `PYTHONPATH`:
`jobs/miao-6b0533a/src` in this experiment's /nrs home.

It is an export rather than the checkout, because the checkout's branch changes and a running job's
loader workers re-import miao at every epoch boundary. Each job prints the miao it imported and
stops if it is not the export. Prediction needs the same `PYTHONPATH`, because `predict.py` reads
the frame through miao.

## Running it

    bash experiments/lm_zebrafish_cell_instance/submit.sh w496_em w496_noem w288_em w288_noem
    bash experiments/lm_zebrafish_cell_instance/submit.sh --dry-run <arm> ...    # print the bsub lines
    bash experiments/lm_zebrafish_cell_instance/tensorboard.sh                   # the four arms

Each arm gets one full B300 node, and each real run waits on a 20-step, one-GPU smoke run. Outputs
go to `/nrs/scicompsoft/orhane/mia-train-experiments/lm_zebrafish_cell_instance/` (`runs/`, `jobs/`,
smoke runs under `*/smoke/`).

## Scoring

Arms are scored on request only, through mia-evals' `lm_zebrafish_cell_instance` task, route
`mws_size_filter`: mutex watershed, then a size filter fitted on frames 1 and 25 and applied to
frame 10. The size filter is therefore fitted on the frames the models trained on, as it was for the
task's Cellpose-SAM and LSD rows.

Each arm predicts whole frames with `predict.py --cover-box` at its own window: 20 tiles per frame at
496, 60 at 288. The data configs are copies of the task's `data/test.yaml` and `data/fit.yaml` with
`patch_size` set to the window in their z, y, x order ([144, 496, 496] or [144, 288, 288]), and
`predict.py` checks that against `img_size`.

There is one run per arm, and no curve in TensorBoard ranks the arms (`tensorboard.sh` says why).
