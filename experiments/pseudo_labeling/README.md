# pseudo_labeling

Bootstrap from five labelled NISB cubes to a hundred unlabelled ones, NoisyStudent-style: label
`train_100` with the best model we have, train a student on real + pseudo labels with noise, then
repeat with the student as the new teacher.

`train_100` is synthetic and therefore fully labelled, and we ignore that. Its ground truth is
never trained on. It is used only as a diagnostic oracle — the thing a real unlabelled dataset
could not give us — to measure how good the pseudo-labels actually were, per round.

| arm | initialisation each round |
|---|---|
| 1 `reset` | the interpolation-trained encoder at 200k, every round |
| 2 `warm` | the previous round's student, i.e. the model that produced its labels |

The arms are the same file with one section changed (`diff 1a_reset_r1.toml 2a_warm_r1.toml`), so
they isolate the initialisation policy and nothing else.

## Why this needed almost no new training code

The whole loop is the existing `affinity_seg` algorithm with a different data config. Four facts
compose:

- **`label_key` is per volume** (`miao/config.py`, on `VolumeConfig`, not `MiaoConfig`), so one
  config mixes `base` cubes reading real ground truth with `train_100` blocks reading pseudo-labels,
  `weight` setting the ratio. That is what `make_round_config.py` emits.
- **`affinity_seg` already turns instance labels into affinity targets**, so materialised
  pseudo-labels train through the unchanged algorithm.
- **Abstention is free.** `affinities_from_labels` masks `labels != ignore_index`, so filtering a
  pseudo-label means writing `-1` into the label array. No loss code changes, and "no filtering"
  is the same path with the thresholds opened up.
- **The noise NoisyStudent needs already exists**: image-only augmentations in `[augment]`, plus
  `drop_path_rate` on the model. Both are config knobs.

The new code is `banis/mia_pseudolabel.py` (`predict` / `build` / `calibrate`) plus
`make_round_config.py` here.

## The traps

**Uncertain must be `-1`, never `0`.** Targets are `(a == b) & (a > 0)` and only `!= ignore_index`
is masked, so `0` is *background* — a confident, trained-on assertion. Writing `0` for "the teacher
was unsure" teaches the student that ambiguity means background. This also forces a signed label
dtype: `uint16` would store `-1` as 65535, and the int64 cast in `_prepare_labels` would turn every
abstention into a real instance id.

**A connected-components label of `0` is not background either.** It means no affinity edge
survived the threshold, which happens at real membrane *and* wherever the teacher was unsure.
`filter_labels` only calls such a voxel background if its foreground score independently agrees,
and abstains otherwise.

**Pick the threshold for precision, not for nERL.** A merger corrupts every voxel pair spanning two
fused neurons; a split corrupts one seam. On a 512³ block of seed100 the best-nERL threshold
(logit +5) had 10 mergers, while logit +7 had 4 at the cost of 3 extra splits. The teacher's
benchmark score is not the objective — the student's is.

**Inference patch size must match training.** RoPE normalises coordinates by the *runtime* grid
extent, so a 512³ input maps the same physical distance to half the phase delta a 256³ input does.
`mia_predict.py`'s own `--patch` default is 512 and contradicts this; `mia_pseudolabel.py` defaults
to 256, matching `score_checkpoint.sh` and the recipe that produced 0.4941.

## Why connected components rather than mutex watershed

MWS is the better story — it uses the long-range channels as repulsive edges and needs no threshold
— and it loses badly here. Measured twice, on a 512³ block of seed100:

| | nERL | mergers |
|---|---|---|
| MWS (`repulsive_stride=4`) | 0.0243 | 242 |
| CC (logit +6) | 0.5587 | 6 |

Diagnosed rather than assumed: the default stride discards 64× of the repulsive edges, and
repulsion is the only thing separating objects. On a 256³ block MWS recovers monotonically —
0.152 (stride 4) → 0.342 (2) → 0.380 (1) — but CC still wins 0.786 to 0.380, so CC it is.

`build` reproduces BANIS' `compute_connected_component_segmentation` via
`cc3d.color_connectivity_graph` rather than importing it, because `banis/inference.py` pulls in
numba, dask, distributed, filelock and scipy at module scope and mia-train's environment has none
of them. Cross-validated on a real 256³ block: the partition is an exact bijection, differing only
in 132 far-face singletons where BANIS seeds on an affinity pointing out of bounds, all of which
`min_size` drops anyway.

## Storage

Source cubes are read-only, so each pseudo-labelled cube gets a **sidecar** OME-NGFF container
whose `raw` is a symlink to the published array and whose `labels/pseudo_rN` is ours. The label
array is the full cube shape with `fill_value=-1`, and only the labelled blocks are written —
unwritten zarr chunks cost nothing, so a 3000×3000×1350 int32 array occupies **856 KB** where dense
would be 48 GB. `bounding_box` per block restricts sampling to what was actually labelled.

Affinities are transient: 6 channels of float16 over a 384³ block is 680 MB, so `submit.sh` deletes
each cube's affinities as soon as its labels are built rather than keeping ~136 GB per round.

## Running it

```bash
bash experiments/pseudo_labeling/submit.sh --smoke 1   # 20 steps, 4 cubes, one GPU
bash experiments/pseudo_labeling/submit.sh             # both arms, both rounds
```

The chain is train → **label** → train, which no other experiment here does: the second training
stage waits on a *dataset* generated by running the first stage's model over 100 cubes, not just on
a checkpoint path. Round 1's labels come from teacher_0 and are shared by both arms, so they are
generated once.

Thresholds come from `mia_pseudolabel.py calibrate` against seed100 (the benchmark's
hyperparameter-selection cube, which teacher_0 never trained on) and are passed via the
`CC_LOGIT` / `TAU_BG` / `TAU_FG` environment variables so the run record shows what was used.

## The oracle

`train_100` is synthetic and fully labelled. We never train on those labels — but reading them
afterwards measures something a real unlabelled dataset could not: how good the pseudo-labels
actually were. `mia_pseudolabel.py oracle` runs automatically after each labelling stage and
`oracle_table.py` puts the rounds side by side.

| signature | reading |
|---|---|
| precision up, instance up | the loop is working |
| precision flat, instance up | confirmation bias — asserting more, knowing no more |
| precision down, merges up | threshold too permissive to bootstrap on |
| `enrich` near 1 | abstention is untargeted, discarding signal for nothing |

Both a pairwise and an instance-level view are reported, because they can disagree sharply: the
MWS-built smoke sidecar scores **0.982 pairwise precision** while showing **36 merges**. Pairwise
metrics are dominated by within-object pairs and will look fine through an over-merging failure.

**These numbers select nothing.** Using them to pick a threshold, a stopping round or a winning arm
would leak ground truth, and the result would not transfer to data that is genuinely unlabelled.
Selection stays on seed100; the oracle only ever explains, after the fact.

## Choosing the threshold

Calibrated on seed100 — the benchmark's hyperparameter cube, which teacher_0 never trained on —
over 6 blocks:

| logit | coverage | merges | splits |
|---|---|---|---|
| +4 | 0.722 | 7 | 139 |
| +5 | 0.680 | 4 | 138 |
| +6 | 0.626 | 1 | 144 |
| **+7** | **0.559** | **0** | 161 |
| +8 | 0.486 | 0 | 177 |

**Pairwise precision cannot pick this.** It saturates at 0.998–0.99998 across the entire grid,
because it is dominated by within-object pairs. The merge count is what discriminates, and +7 is
the first threshold reaching zero. Going further costs coverage and buys nothing.

`tau_bg = 0.40` owns the background/abstain split among voxels CC dropped — background comes out
at a constant 0.146 against ground truth's 0.184, and only 1.4% of what it calls background is
really foreground. Every voxel lost to a higher CC threshold becomes `ignore`, not `background`,
which is the orphan rule doing its job.

`tau_long = 0.30` is a tripwire rather than a filter: on teacher_0 it fires on ~1e-5 of voxels,
because over 6.2M CC-merged pairs the 1st-percentile long-range affinity is 0.763 — this teacher's
short- and long-range channels simply agree. It is here for later rounds, where a student trained
on its own pseudo-labels may become less coherent. `frac_disputed` per block reports what it caught.

## Queue placement, and why training uses 2 GPUs rather than 8

Labelling runs on `gpu_a100`, training on `gpu_h200`. On one queue they contend: a pending
96-slot training job holds slot reservations on every host (LSF reserves for up to 7200 s so
large jobs can assemble a node), which starved the label array — measured, its elements
dispatched one at a time and the array bought nothing over the serial job it replaced. Label
elements take 4 slots, not the queue's 12-per-GPU ratio: measured peak host memory is 11.3 GB,
and over-requesting would strand most of a node across 10 concurrent elements.

Training was originally 8 GPUs × batch 1 = a whole 96-slot node, which needs 8 free GPUs *and* 96
free cores with matching affinity on one host — it sat behind 700+ pending jobs for hours. The
same global batch on **2 GPUs × batch 4** dispatches in under two minutes, and measurement on
identical data says the small layout is also the more efficient one:

| | 8 × H200, batch 1 | 2 × H200, batch 4 |
|---|---|---|
| samples/s | 31.5 | 16.4 |
| MFU | 6.0% | **9.9%** |
| `data_wait_frac` | **0.13** | **0.0008** |
| per 10k-step stage | 42 min, $4.51 | 81 min, **$2.17** |

52% of the throughput on 25% of the hardware. The 8-rank layout was *starved*: one 256³ crop per
rank per step could not keep eight input pipelines fed, so 13% of every step was spent waiting on
data. Four crops per rank across two ranks, with 12 cores per rank instead of 1.5, removes that
almost entirely. Twice the wall clock per stage, half the cost, and it actually starts.

Because a 2-GPU request is reliably satisfied on H200, the twin-queue machinery this needed when
it wanted whole nodes is gone — one queue, one job per stage, no lock. `claim.sh` is kept in the
directory for reference; nothing submits through it. Re-enable it if a stage ever needs a whole
node again.

## Status

Launched 2026-08-18 at `cc_logit=+7, tau_bg=0.40, tau_fg=0.50, tau_long=0.30, min_size=200,
gt_weight=0.3`, 100 cubes × 2 blocks of 384³, sharded 10 ways.

Still untuned, and guesses rather than measurements:

- `GT_WEIGHT = 0.3` — the share of samples drawn from real labels.
- `N_CUBES = 100`, `BLOCKS = 2` at 384³ — ~40 GB of labels per round.
- `max_steps = 50000` for the students, half the teacher's 100k, on the reasoning that the
  training set is ~20x larger in volume. Not validated.
