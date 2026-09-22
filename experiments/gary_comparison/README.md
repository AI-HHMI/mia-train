# gary_comparison: our models against a colleague's, on his volume and his split

One volume, the FlyEM hemibrain Ellipsoid Body crop, split spatially exactly as the colleague split
it, so that whatever we train here is trained, selected and scored on the same voxels as his models.
The point is a like-for-like number, so everything about the data is pinned down in this directory
and nothing about it is left to a run config.

    experiments/gary_comparison/
    ├── README.md
    └── data/
        ├── hemibrain_eb_train.yaml   z <  3000
        ├── hemibrain_eb_val.yaml     3000 <= z < 4000
        └── hemibrain_eb_test.yaml    x, y, z all >= 4000 (one corner)

The three YAMLs differ only in `name` and `bounding_box`; verify with

    cd experiments/gary_comparison/data
    diff <(grep -v '^#' hemibrain_eb_train.yaml) <(grep -v '^#' hemibrain_eb_val.yaml)

## The volume

| | |
| --- | --- |
| store | `/groups/miaai/miaai/lmd-v0.0.1/data/em-drosophila-flyem-hemibrain/crop-001_EllipsoidBody_x24000_y23000_z17000.zarr` |
| format | OME-Zarr 0.5 on zarr v3 (TensorSwitch export), chunks 512^3 |
| axes | `x, y, z`, in that order on disk |
| image | `raw`, uint8, 5000^3 at level 0, 8 nm isotropic, pyramid `s0`..`s5` |
| labels | `labels/proofread-cell-hemibrain-v1.2`, uint64, same shape / pyramid / axis order |
| source | hemibrain v1.2 EM (`clahe_yz/jpeg`) and neuron segmentation, crop origin (24000, 23000, 17000) voxels |

Also present but not used: `labels/auto_pred-mitochondria-hemibrain-v1.2` (16 nm, 2500^3).

This is the same store the `lmd_ssl_v1` finetune set calls `hemibrain_ellipsoid_body`, restricted
there to the central 1024^3 `[1988, 3012)^3` (97.1% foreground, 4,186 instances). Every model from
that lineage and from `sam_lmd_v1` has therefore **trained on voxels inside this experiment's train
and val boxes** (the central cube lies at z 1988..3012, which straddles the 3000 boundary) but not
inside the test corner. That makes them admissible as test-set comparisons and *inadmissible* as
val-set comparisons.

## The split

| split | `bounding_box` (x, y, z; level-0 voxels; `[lo, hi)`) | voxels | share |
| --- | --- | --- | --- |
| train | `[[0, 5000], [0, 5000], [0, 3000]]` | 75.0 G | 60% |
| val | `[[0, 5000], [0, 5000], [3000, 4000]]` | 25.0 G | 20% |
| test | `[[4000, 5000], [4000, 5000], [4000, 5000]]` | 1.0 G | 0.8% |
| unused | `z >= 4000` outside the test corner | 24.0 G | 19.2% |

**Assumption, stated once:** the colleague's boxes are read as `[[x0, x1], [y0, y1], [z0, z1]]` in
the store's own level-0 voxel indices. That is also what miao's `bounding_box` means under
`output_axes: lcxyz`, so the lists are copied verbatim. If his convention were z-first the train and
val slabs would instead be cut along x. The volume is a cube, so nothing in the metadata can tell
the two readings apart; confirm with him before comparing numbers.

The three boxes are disjoint, and miao keeps every sampled window strictly inside its box, so a
training crop never touches val or test tissue. Neurons of course cross the planes; a spatial split
shares *identities* across splits by construction (counted below).

## Conventions the YAMLs follow

- `resolutions: [[8, 8, 8]]` is the stored level-0 voxel size, so miao stays on level 0 and
  resamples nothing. Predicting at another resolution is refused by `VolumeGrid` (it only reads
  level 0); change the patch, not the resolution.
- `output_axes: lcxyz` and `patch_size: [256, 256, 256]`, as every instance-segmentation config in
  this repo. A run config may lower the patch inline; raising it is fine up to 1000 (the val and test
  boxes are 1000 deep).
- `defer_image_ops: true` in the YAML *and* in every `[data]` / `[val_data]` section: resampling and
  normalisation run on the device, not in the loader workers.
- `samples_per_epoch` in the YAML is a placeholder; the run config sets it, and should set it large,
  because every epoch boundary respawns the workers.

Referenced from a run config as

    [data]
    name = "miao_volumes"
    config_path = "experiments/gary_comparison/data/hemibrain_eb_train.yaml"
    samples_per_epoch = 100000
    defer_image_ops = true

    [val_data]
    name = "miao_volumes"
    config_path = "experiments/gary_comparison/data/hemibrain_eb_val.yaml"
    samples_per_epoch = 32
    defer_image_ops = true

## Scoring the test corner: a coverage caveat to settle first

`src/predict.py <run_dir> --data-config experiments/gary_comparison/data/hemibrain_eb_test.yaml`
writes the prediction and the co-registered ground truth on one lattice, and mia-evals scores them.
The lattice (`src/prediction/grid.py`) tiles the box with whole windows at a half-window stride,
drops the tail that does not complete a stride, and centres what remains in the box. At patch 256
a 1000-wide box takes 6 tiles = 896 voxels per axis:

| stride | scored region | share of the 1000^3 test voxels |
| --- | --- | --- |
| patch/2 (default) | `[4052, 4948)^3` = 896^3 | 71.9% |
| patch/4 | `[4020, 4980)^3` = 960^3 | 88.5% |

If the colleague scored his full 1000^3, our number is on a strict subset of his voxels: same
tissue, missing a 52-voxel (0.4 um) rind. Scoring the exact box would need the lattice to extend
below 4000 (into val and train tissue, which is legitimate as *context* but the predictions there
must then be cropped away before scoring), which `VolumeGrid` does not do today. Decide which of
the three you want before comparing: (a) accept the 896^3 subset and ask him for his number on the
same sub-box, (b) `steps_per_patch = 4`, (c) extend `VolumeGrid` with a separate score box.

## Verification

Measured by `/nrs/scicompsoft/orhane/mia-train-experiments/gary_comparison/probes/verify_splits/verify_splits.py`
(results in `verify_splits.json` beside it; regenerate with the `bsub` line in its docstring).

Job 154307009 (2026-09-16, 42 s on 8 CPU slots, 6.3 GB peak). All 18 crops -- six per split, drawn
through `VolumeDataset` exactly as a run would -- landed strictly inside their split box. miao chose
level 0 with a 256^3 read (no resampling), storage axes `xyz`, labels arriving as int64.

| split | instances per 256^3 crop | foreground per crop | s3 census (64 nm): foreground | distinct ids at s3 |
| --- | --- | --- | --- | --- |
| train | 78-168 (mean 121) | 43-100% (mean 87%) | 96.3% | 249,401 |
| val | 75-214 (mean 149) | 94-100% (mean 98%) | 95.1% | 90,410 |
| test | 59-119 (mean 96) | 98-100% (mean 99%) | 97.8% | 2,953 |

Id counts at s3 undercount: thin processes vanish at 64 nm. The central 1024^3 alone holds 4,186
instances at level 0, so expect a few thousand in the test corner.

Identities crossing the split planes, at s3: 2,646 ids appear in both train and val (3% of val's),
136 in both train and test (5% of test's), 259 in both val and test (9% of test's); 2,678 of the
test corner's 2,953 ids appear in neither train nor val.

Id 0 is 3.7% of the volume and is not uniform: one train crop was 57% id 0. `affinity_seg` treats
id 0 as **background, not as unknown**: its target is `(a == b) & (a > 0)`, so every pair touching
an id-0 voxel is a cut *and stays in the loss*; only `ignore_index = -1` is excluded, and miao never
emits -1. If those regions are glia or extracellular space that is the right lesson; if they are
unproofread neurons the model is being taught to shatter them. Worth a look at that crop
(`[687, 943) x [4217, 4473) x [661, 917)`) before the first run; remapping 0 -> -1 would need a
label transform, which nothing in the data path does today.

## Arms

| arm | config | what it is |
| --- | --- | --- |
| 1a | `1a_dinov3_axial_subpixel.toml` | released DINOv3 ViT-L/16 inflated to 3D, axial RoPE, SDPA + compile, `affinity_seg` sub-pixel head from step 0 in one 500k-step stage (lr 3e-4, 5k warmup, linear decay) |
| 1b | `1b_scratch_axial_subpixel.toml` | arm 1a from random initialisation: no `[init]` section, every other non-comment line identical. 1a vs 1b is the value of the pretrained encoder on this volume |
| 1c | `1c_dinov3_axial_subpixel_1m.toml` | arm 1a trained twice as long: `max_steps` 1M and `checkpoint_every` 100k, nothing else changed (so the linear decay stretches to 1M). 1a vs 1c is the value of a longer schedule; submit with `WALL=168:00` |
| 2a | `2a_simmim_lmd_mask60.toml` | **SSL only.** SimMIM pretraining of a 3D DINOv3 ViT-L/16 from random init on the census-20260920 lmd corpus (226 volumes), 500k steps at global batch 128 (16/rank, one B300 node), mask ratio 0.6, axial RoPE, SDPA + compile, in-plane rotation as the only augmentation; validates reconstruction on the hemibrain val slab. Created 2026-09-20 (written as 1M steps, cut to 500k before launch); submit with `WALL=336:00` |
| 2b | `2b_simmim_lmd_mask85.toml` | arm 2a with `mask_ratio = 0.85`, nothing else (`diff` the two). 2a vs 2b, read through their eventual fine-tunes, is the value of a harder pretext task in 3D |
| 3a | `3a_simmim60_axial_subpixel.toml` | arm 1a's recipe with `[init]` pointing at arm 2a's final SSL checkpoint (`prefix = "model."`, no inflation, no `skip`); every other non-comment line identical to 1a. 3a vs 1a: in-domain SimMIM against natural-image pretraining; 3a vs 1b: against no pretraining |
| 3b | `3b_simmim85_axial_subpixel.toml` | the same from arm 2b's checkpoint. 3a vs 3b is the SSL mask ratio, read at equal supervised budget |

    bash experiments/gary_comparison/submit.sh 1a_dinov3_axial_subpixel            # 20-step smoke on 1 GPU, then the real run chained on done(smoke)
    bash experiments/gary_comparison/submit.sh --dry-run 1a_dinov3_axial_subpixel  # print the bsub lines, write the job scripts
    bash experiments/gary_comparison/tensorboard.sh [port] [--smoke] [--list]        # serve the arms' curves; --list only rebuilds the link tree

Everything lands under `/nrs/scicompsoft/orhane/mia-train-experiments/gary_comparison/` in the layout's
own places and nowhere else: run directories in `runs/` (via `--output-root`; bare `--resume` continues the
newest), and LSF logs and the generated job scripts in `jobs/`. Everything belonging to a smoke run -- its
derived config, job script, LSF logs, run directory and TensorBoard link -- sits one level down, in
`jobs/smoke/`, `runs/smoke/` and `tensorboard/smoke/`, so the production runs are never cluttered by it.
Probe scripts with their results go to `probes/<question>/`. A running job's LSF log in `jobs/` is written
live (this cluster's LSF streams `-o`), so `tail` it; `bpeek <jobid>` shows the same.

Arm 1a departs from lmd_ssl_v1's arm 2 in two ways that are choices, not just settings: the sub-pixel head
starts cold with `decoder_zero_init_output = false` (the zeroed default sends no gradient to the encoder at
step 0, and the repo's three earlier cold single-stage attempts collapsed onto the trivial predictor), and
axial RoPE means the pretrained attention weights meet rotations they were not trained with. The collapse
signature to check at the first `[train]` lines: `affinity_accuracy` equal to `target_positive_rate` to
three decimals with `boundary_accuracy` near zero.

**Launch log.** 2026-09-16 15:22: smoke job 154312366 passed (20 steps, 169 s; boundary accuracy rose
0 -> 0.29, loss 0.685 -> 0.586, so the head left the trivial predictor within 15 steps at a 2-step warmup).
Real run job 154312367 on gpu_b300, 8 GPUs, `-W 96:00 -r`; expected 21-31 h at 0.15-0.22 s/step.
Dispatched 21:55 on i02u02 (run `runs/gary__1a_dinov3_axial_subpixel_20260916_215544`). Measured 30.2 crops/s =
0.26 s/step, so ~37 h of stepping: finish about Fri 2026-09-18 midday. **The cold head DID start on the trivial
predictor and escaped**: at step 1k affinity accuracy 0.88748 vs positive rate 0.887476 with boundary accuracy
0.0002; step 2k 0.904 vs 0.902, boundary 0.07; step 3k 0.912 vs 0.893, boundary 0.56; step 7k 0.955 vs 0.897,
boundary 0.67, loss 0.26 -> 0.11. The earlier NISB failures dipped near step 400 and never recovered; this one
recovered between 1k and 3k with `decoder_zero_init_output = false` and the 5k warmup. `data_wait_frac` 0.003.

Arm 1b (from scratch) submitted 2026-09-16 23:51: smoke job 154330430, real run 154330431 chained on it, same
queue, node size and 96 h wall as 1a (its step should cost the same: the init changes no shapes).

Arm 1c (1M steps) submitted 2026-09-19 12:45 after 1a and 1b both scored better at every later checkpoint
and were still improving at 500k: smoke job 154373730, real run 154373731 chained on it, 168 h wall (1a's
0.26 s/step puts 1M steps at ~72 h). Read its checkpoints against 1a's both by step and by remaining-schedule
fraction: at step 500k it sits at half the peak LR where 1a had finished decaying.
**Arm 1c finished 2026-09-22 15:15** (1M steps, 74.4 h, run `runs/gary__1c_dinov3_axial_subpixel_1m_20260919_124840`);
val loss at 100k-multiples 0.132, 0.109, 0.099, 0.108, 0.127, 0.092, 0.125, 0.113, 0.117, 0.113 (32 crops, noisy).
Scored at the **final checkpoint only**, by decision of 2026-09-22 (jobs 154398281-154398284; the 300k and 500k
scorings that had been queued were cancelled before producing anything).

## SSL arms (2a, 2b): SimMIM pretraining on the lmd corpus

Created 2026-09-20 on request. Two configs that differ in one non-comment line
(`mask_ratio` 0.6 vs 0.85; `diff 2a_simmim_lmd_mask60.toml 2b_simmim_lmd_mask85.toml`). Each pretrains a 3D
DINOv3 ViT-L/16 from random initialisation with SimMIM for 500k steps at global batch 128 (16 per rank on one
B300 node) on lmd-configs' census-20260920 single-scale pretraining corpus,
`/groups/miaai/miaai/lmd-v0.0.1/configs/configs/pretraining/singlescale/baseline/8nm_p256_swe0.4_min1e8_up8_census20260920.yaml`.
Only the SSL stage exists; the supervised stage (arm 1a's recipe initialised from an SSL checkpoint) is
deliberately not chained yet. The template is lmd_ssl_v1's arm-1 stage A; every departure from it is listed
and justified in the config header (SDPA + `compile`, 1M steps, the new corpus, 1M samples per epoch, the
hemibrain val slab as validation, the gary logging cadence, and in-plane rotation as the only augmentation).

**The corpus.** Described by lmd-configs' own `describe.py` in
`/nrs/scicompsoft/orhane/mia-train-experiments/gary_comparison/probes/ssl_corpus_and_mask/describe.txt`
(regenerate with `run.sh` there): 226 volumes, 89.8 T level-0 voxels, 8 nm ladder, 256^3 patches,
size-weighting exponent 0.4, so an epoch has an effective sample size of 49 volumes and its top 10 volumes
take 35% of it; Drosophila 35% and mouse 31% of an epoch; EM 77%, ExM 23%; 21.3 MB of stored voxels read per
sample. 64M samples is ~12 passes over the corpus by voxel count, but with replacement and weighted, so
"epochs" mean little here.

**The test corner is in the pretraining data.** The corpus keeps the eval volumes by its own decision
(`hold_out_eval_volumes: false`: pretraining reads no labels, and its header lists every overlap), and it
holds the hemibrain Ellipsoid Body crop whole, at 0.79% of an epoch. A 256^3 crop touches the
`[4000, 5000)^3` test corner when its origin exceeds 3744 on all three axes, which is 0.9% of that volume's
crops, so about 0.007% of all samples (~9k of 128M) show test-corner voxels, unlabelled. For the comparison
with the colleague this means our SSL encoder will have seen a small amount of test-corner *images*. The
strict alternative is a derived copy of the corpus YAML with `bounding_box: [[0, 5000], [0, 5000], [0, 4000]]`
(the train + val boxes) on that one volume; not done, because the corpus's decision of 2026-09-20 was to
document overlaps rather than hold out, and the exposure is tiny.

**Why the mask ratio, and why 0.85.** SimMIM explains its own mask-ratio results through the average
distance from a masked pixel to the nearest visible one: too near and the target is recoverable by
interpolation, too far and it is unpredictable. In 3D every block has 26 neighbours instead of 8, so the same
ratio leaves visible context much nearer. Measured on this grid (256^3 crop, 16-voxel patches, 32-voxel masked
blocks; `mask_avgdist.py` in the probe directory above):

| mask ratio | 3D: mean distance to nearest visible patch | 3D: masked patches with no visible face neighbour | 2D, same grid |
| ---: | ---: | ---: | ---: |
| 0.6 | 19 vox | 27% | 23 px, 40% |
| 0.75 | 22 vox | 48% | 30 px, 61% |
| 0.85 | 27 vox | 65% | 38 px, 74% |
| 0.9 | 32 vox | 76% | 51 px, 84% |

At 0.6 the 3D task is easier than the paper's 2D one at the same ratio; 0.85 is about where the paper's
0.7 sat, inside the range it found flat for 32-px blocks. The other defensible knob was the block size
(`mask_granularity` 2 -> 3, 48-voxel blocks); the ratio was chosen because it is the paper's headline
parameter and leaves the block geometry, and with it the per-patch loss normalisation, unchanged.

**What previous SimMIM runs say about a long run.** Asked 2026-09-20: was SimMIM saturating? Answered with
smoothed curves, not per-step losses: `/nrs/scicompsoft/orhane/probes/simmim_saturation/` (`results.md`,
`curves.png`, script). Rolling mean over 5k-step windows of the logged train loss, val as logged (the
trainer's own pass, no augmentation):

| run | init, data, global batch, length | smoothed train loss 10k -> 50k -> end | second half's share of the total drop | val loss best -> end |
| --- | --- | --- | ---: | --- |
| lmd_ssl_v1 1a | scratch, lmd 91 vol, 128, 100k | 0.149 -> 0.149 -> 0.150 | -6% | 0.087 (15k) -> 0.094 |
| init_comparison 3a | scratch, NISB 5 cubes, 8, 100k | 0.100 -> 0.093 -> 0.093 | 2% | 0.091 (95k) -> 0.093 |
| new_ssl_recipe 2a | LVD init, NISB, 8, 71.5k (20k frozen) | 0.107 -> 0.102 -> 0.103 | -28% | 0.101 (35k) -> 0.104 |
| new_ssl_recipe 3a | LVD init, NISB, 8, 58.4k | 0.101 -> 0.105 -> 0.105 | rose | 0.101 (25k) -> 0.104 |
| new_ssl_recipe 4a | LVD init, NISB, 32, 100k | 0.101 -> 0.105 -> 0.100 | annealing only | 0.097 (100k), 0.098 at 15k |

**Yes, on the pretext loss: every run plateaued within 10-25k steps.** The late-phase slope of the raw
per-step loss is within two standard errors of zero for every run but init 3a (-0.0007 +- 0.0002 per 10k
steps, i.e. 0.7% per 100k), and the one late decrease that exists (4a, 0.105 -> 0.100 over 50k-100k) tracks
the linear LR decay reaching zero -- the loss recovering from a mid-run rise, not new learning. Two things
keep this from meaning "stop at 25k": `visible_l1`, the copy-through error on unmasked patches, kept falling
throughout lmd 1a (0.41 at 10k -> 0.28 at 100k), so the network was still changing; and masked-image
modelling generally shows fine-tuned accuracy improving with pretraining length while the pretext loss is
nearly flat. The pretext loss is not the readout; the fine-tuned score is. Practical consequence: judge these
arms by the supervised stage at several SSL checkpoints (100k, 300k and 500k are the natural ones), not by
their curves. Note also that lmd 1a's train loss carried an irreducible term from BANIS' photometric and
slice augmentations (SimMIM's target is its input, so per-voxel noise and displaced sections enter the
target), which is why it trained at 0.15 and validated at 0.09 while the rotation-only NISB runs showed no
gap. Arms 2a/2b therefore keep in-plane rotation only, the papers' geometric-only recipe (decided
2026-09-20), so their train and val losses sit on one scale. Smooth `val/loss` over five or more points.

**Before a long SSL run: probe the kept lmd_ssl_v1 checkpoints.** lmd_ssl_v1 arm 1 kept all ten SSL
checkpoints (`runs/lmd1__1a_dinov3_simmim_pretrain_20260825_122541/checkpoints/step_{10000..100000}`, 4.8 GB
each). Fine-tuning arm 1a's recipe from, say, steps 10k, 30k and 100k on the hemibrain train box and scoring
the test corner asks directly whether SSL steps past the loss plateau help downstream, at the measured
0.26 s/step of arms 1a-1c on a B300 node:

| supervised steps per probe | wall per probe | three probes on one node | three probes on three nodes |
| ---: | ---: | ---: | ---: |
| 100k | 7.2 h | 22 h | 7.2 h |
| 300k | 22 h | 2.7 d | 22 h |
| 500k (arm 1a's length) | 36 h | 4.5 d | 1.5 d |

Plus a few hours of prediction and scoring per probe, off the training node. 100k-step probes are the
cheap ranking proxy: at 100k the scored 1a/1b ordering already matched the 500k one (mws 0.149 vs 0.140,
then 0.164 vs 0.153), and 1a and 1b at 100k are already scored, so they serve as the LVD-init and scratch
references at no cost. Not launched (2026-09-20); the decision is open.

**Runtime and wall.** 500k steps x 128 = 64M samples (the configs were written for 1M and cut to 500k before launch). B300 compiled 8-rank SimMIM measured 188.7 ms/step at
4 per rank (170 samples/s); at 16 per rank the launch-bound part amortises, so expect 170-230 samples/s, i.e.
3.2-4.4 days -- if the loader keeps up, which means 3.6-4.9 GB/s of stored voxels from `/groups` (the August
H200 run sustained 75-100 samples/s at `data_wait_frac` ~0 without `defer_image_ops`). Submit with
`WALL=336:00`, the queue's maximum (20160 min = 14 days); a run capped at 100 samples/s needs 7.4 days, inside the wall; below ~53 samples/s it
finishes through one `--no-smoke` resubmission, which continues via `--resume`. Read `samples_per_s` and
`data_wait_frac` at the first `[train]` line (step 1k). Checkpoints every 100k steps, 5 x 4.8 GB per arm.

    WALL=336:00 bash experiments/gary_comparison/submit.sh 2a_simmim_lmd_mask60
    WALL=336:00 bash experiments/gary_comparison/submit.sh 2b_simmim_lmd_mask85

Both take a full gpu_b300 node after a 20-step smoke on one GPU; the smoke also exercises stochastic depth's
`randperm(b)[:k]` path under `compile` (16 samples per rank, so `drop_path_rate = 0.1` is functional here,
unlike in arms 1a-1c), which no earlier compiled run took.

**Launch log.** 2026-09-20 15:25, both arms at 500k steps with rotation-only augmentation, `WALL=336:00`,
full gpu_b300 nodes: 2a smoke job 154377884 -> real job 154377885 (chained on `done(smoke)`); 2b smoke job
154377886 -> real job 154377887. The plan is to watch both for saturation (smoothed `val/loss`) and to start
the supervised stage from a 100k-multiple checkpoint before the SSL runs finish if the curves justify it.
The first `[train]` line (step 1k) gives `samples_per_s` and `data_wait_frac`; the smoke logs live in
`jobs/smoke/`, the real logs in `jobs/`, both streamed live.
Both smokes passed at 15:43 (237 s each, 20 steps on one GPU at batch 16, val at step 20, checkpoint written;
masked_fraction 0.5996 / 0.8496; 12 GB host memory). The real runs dispatched at once: 2a on i07u02 (run
`runs/gary__2a_simmim_lmd_mask60_*`), 2b on i04u22.
**Measured at step 2k: 399 (2a) and 401 (2b) samples/s, MFU 26%, `data_wait_frac` 0.002**, i.e. 0.32 s/step,
about twice the rate estimated above (compiled B300 at batch 16 is compute-bound, not launch-bound). 500k steps
therefore take ~44 h: finish about Tue 2026-09-22 midday, with checkpoints landing every ~9 h (100k at
~Mon 00:40, 200k ~09:35, 300k ~18:30, 400k ~Tue 03:25). Losses at 2k: 0.117 (2a) and 0.129 (2b), lr still
warming; grad_norm spikes of 27-40 in warmup are clipped to 1.0 and match lmd 1a's early behaviour. `visible_l1`
rises early (1.1 -> 1.9 / 2.3 by 2k): no loss constrains the visible patches, and the rotation-only NISB runs
showed the same early rise before it fell.
**Both SSL runs finished 2026-09-22** (2a 16:18, 2b 16:22; 48.6 h each; 365 samples/s at the end with
`data_wait_frac` 0.04; checkpoints 100k..500k). Un-augmented val loss on the hemibrain slab (128 crops), 2a / 2b:
10k 0.151 / 0.189, 50k 0.130 / 0.170, 100k 0.129 / 0.168, 200k 0.129 / 0.166, 300k 0.122 / 0.156, 400k 0.121 / 0.150,
500k 0.123 / 0.155; final train loss 0.070 / 0.089. So with rotation-only augmentation the pretext loss kept
improving until ~300-400k (7% for 2a, 12% for 2b between 50k and 400k) and was flat over the last 100k -- a later
plateau than the 10-25k of every earlier run, but a plateau. The two arms' losses are not comparable with each other
(different mask ratios); their value is decided by the supervised stage, which has not been launched.

**The supervised stage (arms 3a, 3b), launched 2026-09-22:** arm 1a's config with
`[init] path = "<SSL run>/checkpoints/step_500000"`, `prefix = "model."`, `strict = true` (the DCP checkpoint
of the whole SimMIM algorithm holds the encoder under `model.`; the prefix selects it and drops the SimMIM head),
no `inflate_2d_to_3d`, no `skip`, RoPE `vanilla` as in the SSL stage. 500k steps, global batch 8, BANIS' full
augmentation, cold sub-pixel head with `decoder_zero_init_output = false`, so the step-1k collapse check of arm 1a
applies. Compare with 1a and 1b at equal supervised steps (100k, 300k, 500k were scored for those).
Submitted 2026-09-22 16:50: 3a smoke 154398315 -> real 154398316, 3b smoke 154398317 -> real 154398318, full
gpu_b300 nodes, `WALL=96:00` like 1a/1b (0.26 s/step -> ~37 h).

**The supervised stage, as originally sketched:** arm 1a's config with
`[init] path = "<run>/checkpoints/step_<N>"`, `prefix = "model."`, `strict = true` (no `inflate_2d_to_3d`,
no `skip`), RoPE already `vanilla`. Compare with 1a (LVD init) and 1b (scratch) at equal supervised steps,
scored on the test corner as before.

**What to read in TensorBoard** (`tensorboard.sh` serves both arms): `train/loss` (masked L1) against
`train/visible_l1`; `val/loss` on the hemibrain slab; `train/masked_fraction` (0.5996 and 0.8496: the ratio
is rounded to whole 32-voxel blocks); `samples_per_s` >= 170 and `data_wait_frac` ~0; `grad_norm` spikes in
the first few thousand steps (lmd 1a had 48 at step 300 and recovered).

## Scoring

    bash experiments/gary_comparison/score.sh 1a_dinov3_axial_subpixel 500000          # predict fit + test, then MWS and CC routes
    bash experiments/gary_comparison/score.sh 1a_dinov3_axial_subpixel 500000 --dry-run

The task in mia-evals is `gary_comparison_neuron_instance`: the test corner from
`configs/gary_comparison_neuron_instance/data/test.yaml` (a copy of `data/hemibrain_eb_test.yaml`), scored on its
lattice-aligned 896^3 centre, ranked by panoptic quality with VOI, SQ, RQ and adapted Rand error
reported. Post-processing is fitted on `configs/gary_comparison_neuron_instance/data/fit.yaml`, a 1000^3 block of the
validation slab directly beneath the test corner (`[4000, 5000) x [4000, 5000) x [3000, 4000)`): the
same lateral tissue, inside the split the protocol reserves for selection, disjoint from the test box.
Two routes, in two scoring configs sharing the task: `mws` (mutex watershed at repulsive stride 1,
then a size filter swept over 0/500/5k/20k/50k voxels) and `cc_threshold` (thresholded components at
logits 0/3/6 crossed with the same filter), the second being the cheap baseline that says what the
long-range channels buy. Predictions run on gpu_b300, like the training, because instance counts
differ across GPU generations.

Records are named `<run>.step<N>.<route>`, e.g.
`gary__1a_dinov3_axial_subpixel_20260916_215544.step500000.mws`, so a row names its checkpoint
directory outright. Predictions live in `eval/<arm>/step<N>/{fit,test}/` under the experiment's /nrs
directory; scored labellings and scratch under `/nrs/scicompsoft/orhane/mia-evals/gary_comparison_neuron_instance/`.

Checkpoints scored for arms 1a and 1b: steps 100k, 300k and 500k each. Arm 1b (from scratch) finished on
2026-09-18 (run `runs/gary__1b_scratch_axial_subpixel_20260917_093923`); it too left the trivial predictor by
step 3k and its validation loss followed 1a's shape (0.148 at 100k, 0.100 at 300k, 0.118 at 500k), so the
pretrained encoder did not change the learning curve's shape. Its scoring jobs were submitted 2026-09-18 evening
(LSF jobs 154372480-154372491). Validation loss (32 crops, noisy) had its
best 100k-window at 300-400k (mean 0.087) and rose again to 0.112 over 400-500k while training loss
stayed near 0.09, so the final checkpoint is not obviously the best one; the three points show the
trend.

## Results (2026-09-19): pretrained vs scratch, test corner, 896^3 centre

Test panoptic quality, post-processing fitted on the fit block (every mws record chose min_size 20000; every
cc record logit +6 with min_size 5000, except 1b at 100k which chose 20000):

| step | 1a DINOv3 init, mws | 1b scratch, mws | 1a cc_threshold | 1b cc_threshold |
| ---: | ---: | ---: | ---: | ---: |
| 100k | 0.149 | 0.140 | 0.110 | 0.081 |
| 300k | 0.157 | 0.151 | 0.112 | 0.101 |
| 500k | **0.164** | 0.153 | 0.118 | 0.114 |

- **The pretrained encoder wins at every checkpoint on both routes**, by 0.011 pq at 500k under the watershed
  and 0.005 under components. The difference is in merges: at 500k/mws, VOI-merge is 0.42 (1a) vs 0.48 (1b)
  while VOI-split is equal (0.95 vs 0.96). Under cc_threshold the gap shrinks with training (0.029 -> 0.011
  -> 0.005), so at this data scale (75 G training voxels) the random init catches up slowly rather than
  failing.
- **Mutex watershed beats thresholded components by ~0.04 pq for both arms**, as on lmd_ssl_v1; the
  long-range channels pay.
- **Both arms were still improving at 500k**, under both routes, despite the validation-loss dip at 300-400k:
  val loss on 32 crops is not the selection signal to trust here; scoring is.
- The size filter dominates absolute pq (unfiltered watershed: 0.003 on the fit block): pq counts every
  surviving fragment, and 20000 voxels (10 um^3) was the fitted floor at every step. Numbers are comparable
  within this table and with anything scored on the same 896^3 region and metric; the colleague's protocol
  is still to be confirmed (see below).

Records: `leaderboard/gary_comparison_neuron_instance/` in mia-evals; scored labellings under
`/nrs/scicompsoft/orhane/mia-evals/gary_comparison_neuron_instance/scored/<record>/`.

## Still to get from the colleague

- The axis-order reading of his boxes (see the assumption above).
- Which voxels he scored (the whole 1000^3 corner or a sub-box), and with which metric (voxel
  panoptic quality / VOI as mia-evals reports, or something skeleton-based).
- Whether his models saw any data beyond the train box (pretraining corpora, other volumes), so the
  comparison can be framed as same-data or not.

## Next

Models. Nothing here presupposes an architecture; the natural first arm is the `lmd_ssl_v1` arm-2
recipe (released DINOv3 ViT-L/16 inflated to 3D, `affinity_seg`, two-stage interpolate -> sub-pixel
head) pointed at these YAMLs, since that is the best-scored affinity lineage in this repo and its
configs need only the two `config_path` lines changed.
