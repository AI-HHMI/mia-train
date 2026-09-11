# Does a Segment-Anything model beat affinity prediction on the lmd eval set -- and can it use the unlabeled corpus?

`lmd_ssl_v1` trained and scored two DINOv3 ViT-L affinity models on eight instance-segmentation
volumes: arm 1 (random init, SimMIM on 87 unlabeled volumes, then finetune) and arm 2 (the released
LVD-1689M checkpoint, then finetune), scored with mutex watershed and a fitted size filter at
pq 0.2369 and 0.2287. This experiment puts the repo's promptable segmentation strategy
(`algorithms/promptable_seg.py`, a Segment Anything for volumes) through **the same encoder, the
same 4/4 volume split, the same schedule and augmentation, and the same scoring protocol**, and
asks two things:

1. **SAM vs affinities**, like for like: the LVD checkpoint plus the GT finetune, with the affinity
   head replaced by the promptable one. This is `base` round 0 -- the analogue of arm 2.
2. **Can SAM use the unlabeled corpus arm 1 pretrained on?** SAM needs instance labels; the 87
   volumes have none. The paper's answer is its data engine: a model trained on labelled data
   annotates unlabeled data, and the model is retrained on the result, iteratively. Rounds 1-2 of
   every arm are that engine, with the human out of the loop -- the analogue of arm 1, in the sense
   that it is the same unlabeled data used a different way.

On top of that, an **architecture sweep** of the mask head's knobs that decide how much fine
spatial detail a mask can carry, each as a full chain identical to `base` but for one knob.

    bash experiments/sam_lmd_v1/submit.sh --smoke base    # 20-step rounds, one tiny block, one GPU
    bash experiments/sam_lmd_v1/submit.sh                 # every arm, every round, on gpu_b300
    bash experiments/sam_lmd_v1/tensorboard.sh            # watch it
    bash experiments/sam_lmd_v1/predict_eval.sh base 2    # eight volumes -> instances artifacts
    bash experiments/sam_lmd_v1/score.sh base 2           # -> the lmd_ssl_v1_neuron_instance table

Configs are generated: `make_configs.py` emits the 24 TOMLs (8 arms x 3 rounds); an arm's identity
is its entry in `ARMS` and nothing else. `unlabeled_corpus.yaml` is a pinned copy of the data config
arm 1 pretrained on, because the corpus tooling has since been rebuilt and lists 189 volumes today.

## The chain, per arm

| round | init | data | steps | peak LR |
| --- | --- | --- | --- | --- |
| r0 | DINOv3 LVD-1689M, inflated (superposition RoPE) | 4 GT finetune volumes | 100k | 3e-4 |
| L1 | -- | r0 labels **1 block** (512^3 at 8 nm) of each of the 87 unlabeled volumes | | |
| r1 | r0's whole model | GT (weight 0.5) + L1's blocks | 50k | 1e-4 |
| L2 | -- | r1 labels **4 blocks** per volume (a superset of L1's cells) | | |
| r2 | r1's whole model | GT (0.5) + L2's blocks | 50k | 1e-4 |

Everything in `[model]`, `[trainer]` and `[augment]` is lmd_ssl_v1's, with three deliberate
departures, each stated in the generated TOML where it applies:

- **One 100k schedule for round 0, not 50k + 50k.** Arms 1/2 split the finetune only because the
  zero-initialised sub-pixel affinity head cannot train from a cold encoder. SAM has one decoder
  and nothing to stage. Rounds 1-2 take stage C's lower peak for stage C's reason: a warm start
  continues a model that converged under a decaying schedule.
- **SDPA + `torch.compile` instead of FlashAttention-4.** The same function to reduction order;
  FA4 cannot be compiled, compilation is worth 1.66x on B300 for this host-bound step, and FA4 buys
  nothing at 4096 tokens (promptable_seg_v1/RESULTS.md).
- **`defer_image_ops = false`**, against this repo's usual setting. With deferral, the geometric
  augmentation runs on the device *after* the worker has drawn the prompts (`PromptTargets` is a
  sample transform on the un-augmented labels), so every click and box would be misregistered with
  the rotated image. No earlier promptable run hit this -- none had geometric augmentation -- but
  this one inherits BANIS's rotation and slice shifts from arms 1/2. In the workers the order is
  augmentation first, prompt sampling second, and deferral was throughput-neutral here anyway.

`samples_per_epoch` is 10,000 rather than the finetune stages' 1,000 -- statistically identical
(random sampling with replacement), and it removes the worker respawn every 125 steps -- and
`num_workers` is 8 rather than 6, since here the workers also run the connected-components pass
and prompt sampling. Neither touches what the model sees.

## The data engine

The paper's fully automatic stage, with the model where the annotators were:

1. **Label.** `pseudolabel.py label` runs `PromptGridPredictor` -- the same segment-everything
   pass `predict.py` uses -- over a block of an unlabeled volume: 14^3 clicks per tile, masks kept
   only if the IoU head predicts >= **0.7** and the boundary is stable under a +-1 logit shift
   (>= **0.8**), NMS at 0.7, tiles reconciled by propagating each painted object into the next tile
   as a mask prompt. Stricter gates than at eval (0.5 / 0.5), because a training target is judged
   by precision: a missed object costs nothing (its voxels stay unclaimed and are never prompted
   for), a merged or truncated one is trained on.
2. **Store.** Labels go into a **sidecar** OME-Zarr per volume -- `raw` symlinked to the read-only
   published store, `labels/sam_rN` ours, written on the source's own level-0 grid so miao
   co-registers them by construction (`blocks.py`). Unlabelled voxels read as -1 from the fill
   value; unclaimed voxels inside a block are 0. Each block is its own `volumes:` entry with its own
   `bounding_box`, so no crop spans two blocks and ids need only be unique within one.
3. **Train.** `make_round_config.py` mixes the four GT volumes (weight 0.5) with every non-empty
   block (sharing 0.5); the round warm-starts from the model that labelled it
   (`[init].target = "algorithm"`), under the full augmentation -- the student sees noised inputs
   the teacher never did, which is what makes iterated self-training more than a fixed point.
4. **Grow and repeat.** Blocks per volume 1 -> 4 (~1.7x, then ~7x the GT's voxel count), positions
   a fixed per-volume permutation, so round 2 relabels round 1's cells with the better model and
   adds new ones -- the paper retrained six times for the same reason.

Two properties of `promptable_seg` mean the pseudo-labels need no algorithm change: prompts are
drawn only on ids > 0, so unlabelled space is never clicked on; and a mask's target is "this
object against everything else", so an unlabelled neighbour is correctly negative. The one way a
pseudo-label misleads is a truncated mask, which the stability gate exists to reject.

**The diagnostic.** Every labelling array also runs the labeller over blocks of the four GT
volumes and scores it against their truth (`pseudolabel.py diagnose`): precision of pseudo-masks
at IoU 0.5, merges (a pseudo-mask that two true objects each make >= 10% of), recall of true
objects >= 512 voxels, claimed fraction. The finalise job prints the table; `diag/<arm>/sam_rN/`
holds it. It is **in-sample for the teacher** -- those volumes trained it -- so it is an upper
bound on label quality, and it explains rather than selects: the rounds' knobs are fixed above,
not tuned on it.

## The arms

Each is the full chain, differing from `base` in one knob of `[algorithm]`:

| arm | knob | why |
| --- | --- | --- |
| `base` | stride 4 (`mask_upscale 4`), 32 mask features, 2 decoder layers, dim 256 | the reference head (`configs/promptable_lmd.toml`) |
| `stride2` | `mask_upscale 8` -> masks at 2 voxels (16 nm) | a *perfect* stride-4 model caps at 0.755 IoU (0.555 on objects under 4k voxels) vs 0.912 at stride 2, measured on this corpus (`sam3d/stride_ceiling.py`); 8x the decoder output |
| `stride1` | `mask_upscale 16` -> voxel-resolution masks | the ceiling removed; 64x the decoder output, only feasible on B300 memory, expected slowest by far |
| `stride2_small` | stride 2 **and** `min_object_voxels 64` | the 512 floor exists because stride 4 pools a 64-voxel object into one cell; at stride 2 small objects become learnable -- the one arm that also changes a data knob |
| `feat64` | `mask_feature_dim 64` | width of the fine feature volume every mask is read out of |
| `refine4` | `mask_refine_depth 4` | the only layers running *at* mask resolution; seam receptive field 5 -> 9 cells (new knob, plumbed through `promptable_seg`) |
| `deep4` | `decoder_depth 4` | more prompt-image attention (the paper's data-engine model used 3) |
| `wide512` | `prompt_dim 512` | neck and decoder width |

Because every arm has its own round 0, the sweep is measured twice: on ground truth alone (r0) and
after the engine (r2). A knob that helps at r0 but not at r2 -- or the reverse -- is a finding
about the interaction, not noise.

## Scoring

`predict.py` already writes an `instances` artifact plus the co-registered ground truth for a
promptable run, on the lattice the affinity artifacts were predicted on (same YAMLs, same patch).
The MWS leaderboard rows are scored from stored labellings through
`lmd_ssl_v1_neuron_instance_mws_{fit,test}.toml` -- a `size_filter` swept over
[0, 500, 5000, 50000] on the finetune half, applied once to the reported half -- so a SAM row goes
through exactly those files with no watershed. Same region, same metric, same fitted parameter:
**the SAM row and the MWS row differ only in what wrote the labelling.**

Mask-generator settings at eval are fixed (`predict_eval.sh`: 14^3 clicks, gates 0.5/0.5,
propagate reconciliation), taken from promptable_seg_v1/RESULTS.md and applied identically to
every arm; the size filter is the only fitted parameter, as it was for MWS. Every round is scored
at its final step, as arms 1/2 were, so the validation set selects nothing.

## Caveats, in the order they would bite

- **Init differs from arm 1.** Arm 1 started from random weights; every SAM chain starts from the
  LVD checkpoint, because the engine needs a labelled-data teacher to begin and the paper's SAM
  likewise began from a pretrained (MAE) encoder. So r2 vs arm 1 compares two ways of using the
  corpus from different starting points; **r2 vs r0 is the clean "does the unlabeled data help SAM"
  test**, and r0 vs arm 2 the clean "SAM vs affinities" one.
- **The stride arms' cost is unmeasured.** `stride2` and `stride1` multiply the decoder's output
  tensors by 8 and 64; the walls in `submit.sh` are scaled 1.5x / 2x by guess. Read
  `samples_per_s` in TensorBoard and tighten them; a `-W` kill does not requeue.
- **GPU generation.** All SAM arms train and predict on B300; arms 1/3 were predicted on H100 and
  arm 2 on L4. The same code gives ~2% different instance counts across generations (measured),
  which is below the differences that would matter here but not zero.
- **The diagnostic is in-sample** (see above). The number that is not is the leaderboard row.
- **Storage.** A round's sidecars are ~0.2-1 GB per block compressed; L2 is ~350 blocks. The
  scratch tree's README records what regenerates what; delete a round's sidecars once its student
  is scored, and the `eval/` artifacts once their record exists.
- **Arms 1/2's `drop_path_rate = 0.1` was a no-op, and these configs write 0.0.** DINOv3's
  stochastic depth drops whole *samples* from a block's residual branch (`randperm(b)[:k]` with
  `k = max(int(0.9 b), 1)`), and at one sample per rank the subset is always the whole batch. So
  0.0 here is the same function arms 1/2 trained, stated honestly. Found because the degenerate
  `randperm(b)[:b]` trips an inductor pattern bug under `torch.compile` (never hit before: every
  earlier compiled run had drop path off), which killed the first smoke run.
- **Training jobs run with `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0`.** Inductor's channels-last
  rewrite of conv graphs switches on for `stride1` (3D convolutions over 256-cubes) and propagates
  a permuted layout into a cuDNN attention input, which torch 2.13 does not constrain -- the job
  dies at its first compiled step. Off for every arm, so no two arms differ by a layout decision;
  the cost, if any, falls on the stride arms' voxel-resolution convolutions, which is the step time
  to read off `samples_per_s`.
- **The mask-only refinement round** the paper trains (a round with a mask prompt and no new click)
  is still missing from `promptable_seg`, as promptable_seg_v1 noted. It would make the IoU head
  trustworthy on propagated masks; it is a recipe change and stays out of this comparison.

## Core changes this experiment needed

- `promptable_seg`: `mask_refine_depth` / `mask_upscale_hidden` reach the decoder's sub-pixel
  expansion (the `refine4` knob). Test in `tests/unit/test_promptable_seg.py`.
- `layers/dinov3/block.py`: the stochastic-depth path is taken only when it would actually drop a
  sample. Numerically identical (at the whole-batch subset it was a gather, an `index_add` and a
  scale of 1), and it removes the `randperm[:b]` that inductor's `randperm_index` pattern cannot
  bind under `torch.compile` -- the failure that stopped every compiled run with `drop_path > 0`
  at batch 1. Test in `tests/unit/test_dinov3_layers.py`.
- `prediction.grid.VolumeGrid`: builds for a volume with no `label_key` (`label_level` is then
  None), and refuses a volume miao would read above pyramid level 0 rather than reading it from
  the wrong coordinates -- an invariant that was assumed and never checked. Both measured against
  the 87 unlabeled volumes (all level 0 at 8 nm). Tests in `tests/unit/test_predict.py`.

Model-free tooling tests: `python -m pytest experiments/sam_lmd_v1/test_tools.py`.
