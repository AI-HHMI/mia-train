# Does SSL on the lmd corpus beat a pretrained checkpoint?

Three ways of getting an encoder, then **one shared two-stage finetune** on instance segmentation.
The finetune is the comparison; the arms differ only in what they hand it.

| arm | encoder comes from | data | RoPE |
| --- | --- | --- | --- |
| 1 | random init -> SimMIM, 100k steps | 90 volumes, 8 nm | `vanilla` (axial 3D) |
| 2 | released DINOv3 LVD-1689M checkpoint, no SSL | — | `superposition` |
| 3 | random init -> MuViT-MAE, 100k steps | 90 volumes, 8/16/32 nm | intrinsic (world-coordinate) |

Arm 2 is the baseline: pretrained checkpoint plus finetuning is what any SSL result has to beat,
and the NISB work established we had never actually cleared it.

    bash experiments/lmd_ssl_v1/submit.sh            # all three arms, all stages
    bash experiments/lmd_ssl_v1/submit.sh --dry-run  # print the bsub lines
    bash experiments/lmd_ssl_v1/submit.sh --smoke    # 20 steps of every stage, one GPU

Configs are generated -- `make_configs.py` for the eight TOMLs, `make_splits.py` for the four data
YAMLs. Edit those, not the outputs.

## The shared finetune

Two stages, 50k each, because a sub-pixel head cannot train from a cold encoder. Measured: its
output layer is zero-initialised, so at step 0 **no gradient reaches the encoder at all**, and the
early signal is ~1e-5 against the interpolating decoder's ~8.6. Stage B trains the interpolating
head; stage C swaps in the sub-pixel head, warm-started from B's encoder.

Peak LR is **3e-4 for stage B and 1e-4 for stage C**. Stage C continues an encoder that has already
converged under a decaying schedule, and restarting at the full peak would undo that. A lower peak
buys that without a stateful multi-stage scheduler.

Everything else is identical across arms: linear schedule, `min_lr_ratio = 0.001`, 3k warmup,
`layerwise_lr_decay = 1.0`, `patch_embed_lr_mult = 1.0`, bf16, one full node per stage
(`dp_shard = 8`, batch 1/rank -> global batch 8).

`layerwise_lr_decay` and `patch_embed_lr_mult` must stay at 1.0. Arm 2's superposition RoPE hides
its entire depth term behind a **zero-initialised scalar**, so anything that damps the encoder's
learning rate leaves z-position at zero and the model is effectively 2D, silently.

## Evaluation data: 4 finetune / 4 validation

The 8-volume instance-segmentation eval set splits into four natural pairs, so one of each pair
goes to either side and every modality appears in both:

| | finetune | validation |
| --- | --- | --- |
| mouse cortex ssTEM | `kasthuri15_ac3` | `kasthuri15_ac4` |
| zebrafish EM | `zebrafish_quadcube1` | `zebrafish_doublecube1` |
| mouse ExM (LICONN) | `liconn_mouse_dg` | `liconn_mouse_hippocampus` |
| Drosophila EM / ExM-at-scale | `hemibrain_ellipsoid_body` | `liconn_expid82` |
| | 6.7 G vox, 10,023 inst | 9.2 G vox, 18,195 inst |

AC3-train / AC4-val is the conventional Kasthuri protocol; the zebrafish crops are from different
coordinates. Volumes are equally weighted -- size-proportional would send ~80% of finetuning to the
zebrafish crop, defeating the point of the split.

**Volume-wise rather than within-volume**, which was the first plan. Splitting each annotated box
in two is infeasible at patch 256 for the multi-scale arm: `bounding_box` must contain the
*coarsest* window (256 x 32 nm = 8192 nm per axis) and only 2 of 8 boxes can hold two disjoint
copies of that. Volume-wise keeps every box whole, so both variants work at patch 256.

**With only 8 volumes the validation set does double duty** as model selection and reported score.
Select checkpoints on training-volume val patches and report on the 4 held-out volumes, or the
reported number is selected-on.

## Eval volumes are held out of pretraining

Arms 1 and 3 pretrain on `lmd_pretraining_*_min1Gvox_noeval_*`: the corpus with the 8 eval volumes
removed. 6 of the 8 are otherwise *in* the pretraining set as the same stores, so without this the
SSL arms would pretrain on the very volumes they are scored on while arm 2 never sees them. The
hold-out costs ~0.1% of the corpus.

## Caveats, in the order they would bite

- **Arm 1 vs arm 3 changes three things at once** -- architecture, objective, and scale ladder.
  This answers "which recipe wins", not "does multiscale help". Model size is *not* one of the
  three: MuViT is sized to 314.9M against DINOv3's 306.6M, within 3%.
- **`shift_slice` perturbs arm 3's cross-level registration.** It displaces content by a fixed
  number of voxels, and one voxel is a different physical distance at each level: a 10-voxel shift
  is 10 world units at level 0, 20 at level 1, 40 at level 2. Augmentation is held identical across
  arms anyway -- dropping it for arm 3 alone trades a known confound for an unknown one -- but this
  is the first thing to ablate if arm 3 underperforms. In-plane rotation was checked and is
  **safe**: the coordinate grid maps onto itself under x<->y swap and both flips, and every level
  turns together.
- **Arm 3's step cost is unmeasured.** Its sequence is 12288 tokens against 4096 and attention is
  quadratic, so its wall-clock limits are set high rather than extrapolated. Read `samples_per_s`
  from the first arm-3 log and tighten `WALL_SSL_M` / `WALL_FT_M` in `submit.sh`.
- **`data/` is not stable.** It went 440 -> 570 stores in one afternoon while this was being built,
  and volume counts moved 106 -> 104 -> 97 -> 91 -> 90. Regenerate the pretraining configs and
  re-run `--check` before trusting a stale one.
- **Rank on nERL, not on `val/boundary_accuracy`.** The latter is per-voxel and blind to
  instance-level fragmentation: scoring pseudo-labelling checkpoints once showed nERL swinging
  0.5844 -> 0.3889 -> 0.5518 while val boundary accuracy sat flat at 0.925-0.945.

## Core changes this experiment required

Three, all in `mia-train`, all with tests:

- `MuViT3D.patch_features` -- returns the **finest level's** tokens and grid, so a dense head can
  drive a multi-scale encoder. It previously declined. The coarser levels are not discarded; they
  reach the finest tokens through joint attention.
- `affinity_seg._prepare_labels` -- a multi-scale batch now supervises level 0 instead of raising.
- `affinity_seg._step` -- spatial shape from `shape[-3:]` rather than `shape[2:]`. Identical for
  `(B,C,D,H,W)`; the old form silently picked up the channel axis for `(B,L,C,D,H,W)`.
