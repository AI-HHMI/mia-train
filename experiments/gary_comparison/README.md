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

    bash experiments/gary_comparison/submit.sh 1a_dinov3_axial_subpixel            # 20-step smoke on 1 GPU, then the real run chained on done(smoke)
    bash experiments/gary_comparison/submit.sh --dry-run 1a_dinov3_axial_subpixel  # print the bsub lines, write the job scripts
    bash experiments/gary_comparison/tensorboard.sh [port] [--smoke] [--list]        # serve the arms' curves; --list only rebuilds the link tree

Everything lands under `/nrs/scicompsoft/orhane/mia-train-experiments/gary_comparison/` in the layout's
own places and nowhere else: run directories in `runs/` (via `--output-root`; bare `--resume` continues the
newest), and LSF logs and the generated job scripts in `jobs/`. Everything belonging to a smoke run -- its
derived config, job script, LSF logs, run directory and TensorBoard link -- sits one level down, in
`jobs/smoke/`, `runs/smoke/` and `tensorboard/smoke/`, so the production runs are never cluttered by it.
Probe scripts with their results go to `probes/<question>/`. Live stdout of a running job: `bpeek <jobid>`.

Arm 1a departs from lmd_ssl_v1's arm 2 in two ways that are choices, not just settings: the sub-pixel head
starts cold with `decoder_zero_init_output = false` (the zeroed default sends no gradient to the encoder at
step 0, and the repo's three earlier cold single-stage attempts collapsed onto the trivial predictor), and
axial RoPE means the pretrained attention weights meet rotations they were not trained with. The collapse
signature to check at the first `[train]` lines: `affinity_accuracy` equal to `target_positive_rate` to
three decimals with `boundary_accuracy` near zero.

**Launch log.** 2026-09-16 15:22: smoke job 154312366 passed (20 steps, 169 s; boundary accuracy rose
0 -> 0.29, loss 0.685 -> 0.586, so the head left the trivial predictor within 15 steps at a 2-step warmup).
Real run job 154312367 on gpu_b300, 8 GPUs, `-W 96:00 -r`; expected 21-31 h at 0.15-0.22 s/step.

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
