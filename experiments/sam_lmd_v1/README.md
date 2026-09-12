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

    bash experiments/sam_lmd_v1/submit.sh --smoke --rounds 0 feat64   # 20 steps, one GPU
    bash experiments/sam_lmd_v1/submit.sh --rounds 0 feat64           # VERSION 3: round 0 only
    bash experiments/sam_lmd_v1/submit.sh                             # every arm, every round
    bash experiments/sam_lmd_v1/tensorboard.sh                        # watch it
    bash experiments/sam_lmd_v1/predict_eval.sh feat64 0              # eight volumes -> instances
    bash experiments/sam_lmd_v1/score.sh feat64 0                     # -> the leaderboard table

**Status (2026-09-12, version 3).** Two full launches were stopped and their artifacts deleted;
what they taught is in "What versions 1 and 2 established" below, with every number. Version 3
trains ONE arm (`feat64`) for round 0 only, 200k steps, the original prompting recipe, with the
mask loss fixed to ignore unlabelled voxels and the labeller reconciling windows by agreement.
The data-engine rounds and the other arms wait until that model's single-window recall says it
is worth labelling with.

Configs are generated: `make_configs.py` emits the 24 TOMLs (8 arms x 3 rounds); an arm's identity
is its entry in `ARMS` and nothing else. `unlabeled_corpus.yaml` is a pinned copy of the data config
arm 1 pretrained on, because the corpus tooling has since been rebuilt and lists 189 volumes today.

## The chain, per arm

| round | init | data | steps | peak LR |
| --- | --- | --- | --- | --- |
| r0 | DINOv3 LVD-1689M, inflated (superposition RoPE) | 4 GT finetune volumes | 200k (v3; 100k in v1/v2) | 3e-4 |
| L1 | -- | r0 labels **1 block** (512^3 at 8 nm) of each of the 87 unlabeled volumes | | |
| r1 | r0's whole model | GT (weight 0.5) + L1's blocks | 50k | 1e-4 |
| L2 | -- | r1 labels **4 blocks** per volume (a superset of L1's cells) | | |
| r2 | r1's whole model | GT (0.5) + L2's blocks | 50k | 1e-4 |

Everything in `[model]`, `[trainer]` and `[augment]` is lmd_ssl_v1's, with three deliberate
departures, each stated in the generated TOML where it applies:

- **One schedule for round 0, not 50k + 50k.** Arms 1/2 split the finetune only because the
  zero-initialised sub-pixel affinity head cannot train from a cold encoder. SAM has one decoder
  and nothing to stage. Version 3 doubles it to 200k (linear decay, checkpoints every 12.5k):
  at 100k the training curve was still rising when the rate reached its floor, and training and
  validation IoU were identical, so the models had not fit even the four training volumes. Rounds 1-2 take stage C's lower peak for stage C's reason: a warm start
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

`samples_per_epoch` is 100,000 rather than the finetune stages' 1,000 -- statistically identical
(random sampling with replacement), and it moves the worker respawn from every 125 steps to every
12,500 -- and
`num_workers` is 8 rather than 6, since here the workers also run the connected-components pass
and prompt sampling. Neither touches what the model sees.

## The data engine

The paper's fully automatic stage, with the model where the annotators were:

1. **Label.** `pseudolabel.py label` runs `PromptGridPredictor` -- the same segment-everything
   pass `predict.py` uses -- over a block of an unlabeled volume: 14^3 clicks per tile, masks kept
   only if the IoU head predicts >= **0.7** and the boundary is stable under a +-1 logit shift
   (>= **0.8**), NMS at 0.7, windows reconciled by **agreement** (`tile_merge = "consensus"`: two
   windows' masks are one object only where they agree in the region both saw; disputed cells stay
   unlabelled -- see the sweep below for why the earlier `propagate` rule was replaced). Stricter
   gates than at eval (0.5 / 0.5), because a training target is judged
   by precision: a missed object costs nothing (its voxels stay unclaimed and are never prompted
   for), a merged or truncated one is trained on.
2. **Store.** Labels go into a **sidecar** OME-Zarr per volume -- `raw` symlinked to the read-only
   published store, `labels/sam_rN` ours, written on the source's own level-0 grid so miao
   co-registers them by construction (`blocks.py`). Everything the teacher did not claim reads as
   -1 -- never 0, because 0 asserts "background" and a teacher with recall far below one cannot
   make that assertion; since version 3 the losses treat -1 as silence (below). Each block is its
   own `volumes:` entry with its own `bounding_box`, so no crop spans two blocks and ids need only
   be unique within one.
3. **Train.** `make_round_config.py` mixes the four GT volumes (weight 0.5) with every non-empty
   block (sharing 0.5); the round warm-starts from the model that labelled it
   (`[init].target = "algorithm"`), under the full augmentation -- the student sees noised inputs
   the teacher never did, which is what makes iterated self-training more than a fixed point.
4. **Grow and repeat.** Blocks per volume 1 -> 4 (~1.7x, then ~7x the GT's voxel count), positions
   a fixed per-volume permutation, so round 2 relabels round 1's cells with the better model and
   adds new ones -- the paper retrained six times for the same reason.

Three properties of `promptable_seg` make a partial labelling a usable target: prompts are drawn
only on ids > 0, so unlabelled space is never clicked on; a mask's target is "this object against
everything else", so a labelled neighbour is correctly negative; and -- since version 3 --
unlabelled voxels (-1) are **silence** in every loss, in the IoU the head is trained to predict,
and in the correction clicks (`targets.pooled_known`, the `weight` argument of every function in
`promptable/losses.py`). Versions 1 and 2 lacked the third: `pooled_masks` clamps labels at 0, so
every unclaimed voxel trained as "not this object", and a model trained on the truncated
pseudo-pieces the labeller produced would have learned a boundary wherever a window ended. That
was found by reading, not by measurement -- no pseudo-label round ever finished -- but it would
have capped every round.

**The diagnostic.** Every labelling array also runs the labeller over blocks of the four GT
volumes and scores it against their truth (`pseudolabel.py diagnose`): precision of pseudo-masks
at IoU 0.5, merges (a pseudo-mask that two true objects each make >= 10% of), recall of true
objects >= 512 voxels, claimed fraction. The finalise job prints the table; `diag/<arm>/sam_rN/`
holds it. It is **in-sample for the teacher** -- those volumes trained it -- so it is an upper
bound on label quality, and it explains rather than selects: the rounds' knobs are fixed above,
not tuned on it.

## The v2 prompting recipe (retired in version 3; knobs default to off)

Version 3 trains with the original recipe -- interior click or noised box, then two correction
rounds of one click each (foreground where the mask missed, background where it spilled). The v2
additions below were built for a head-calibration failure that the probe later showed did not
exist (the head was calibrated; the precision was lost in tile assembly), and in an undertrained
model they cost what is scarcest: a quarter of the prompt slots train only the head, and a third
of the remaining clicks are made harder than the grid poses. They also cost throughput (0.53 vs
0.40 s/step on `base`). The knobs remain, off; the probe's `offPass` column measures what they
were for, and a small `offobject_prob` can return if background clicks start passing the gate.

What version 1 established is that the IoU head was never trained on the prompts the mask generator
issues. Training sampled every round-0 prompt *inside a true object* (a uniformly random interior
voxel, or the object's noised box), so the head learned "how good is this candidate, given a click
that is on a real object" -- and at segment-everything time it was asked about clicks on membranes,
on unannotated space and hard against boundaries, where its answer was untrained and, as measured,
wrong in the confident direction. Every round of every arm now trains with three kinds of round-0
prompt, drawn per slot in the dataloader worker (`promptable/targets.PromptTargets`):

| kind | share of the 16 slots | the prompt | the target |
| --- | --- | --- | --- |
| interior click or box | ~70% of object slots x (`box_prob` 0.5 click / 0.5 box) | as before | the object |
| **boundary click** | `boundary_prob` = 30% of object clicks | a *positive* click drawn from the object's rim -- the voxels within `boundary_radius` = 1 of a different label -- right against the membrane | the object |
| **off-object click** | `offobject_prob` = 25% of slots (~4 per crop) | a *positive* (foreground-labelled) click on a voxel with label 0: membrane, extracellular space, anything no annotator called an object -- the same prompt the grid issues everywhere | none |

Positive and negative prompts both appear, as before: round 0 is one positive prompt; the two
correction rounds then add one click each, foreground where the prediction missed the target and
background where it spilled, plus the previous mask. What changes is *where* the round-0 positive
click can land, and what the model is told about it:

- A **boundary click** is the hardest legitimate positive: the object is the target, the click is
  one voxel from its neighbour. Uniform sampling produces these rarely for thick objects and the
  grid produces them constantly for thin neurites (a 1-voxel rim is 5-15% of a neurite's volume).
  It trains the decoder to resolve the side and the head to know when that is uncertain.
- An **off-object click** has no object. The mask losses skip the slot -- training the decoder
  towards an empty mask is the collapse every mask model starts in -- and only the IoU head is
  trained, to predict 0 for every candidate the decoder produces from it. It is an ordinary
  foreground click, *not* a background-labelled one, because that is what the grid sends. Label 0
  only: -1 (unknown) is never sampled, which is why the sidecars store unclaimed voxels as -1.
  `offobject_pred_iou` in TensorBoard is the number this exists for; it must fall towards 0 while
  `first_iou_error` on object prompts stays put.

Correction rounds are unchanged for off-object slots: every predicted voxel is a spill, so the
correction is a background click on the prediction, and the head is again trained to 0 -- the same
situation propagation puts it in at inference.

## What versions 1 and 2 established

Everything below is measured on the same four 512^3 blocks of the finetune volumes (in-sample for
the model, so an upper bound) unless stated; the tables are in the version sections further down.

1. **The model is undertrained, and that is the underlying cause.** At 100k steps training and
   validation one-click IoU were both ~0.43 and still rising when the linear schedule reached its
   floor. Its kept masks are as precise as the metric allows (single-window precision 0.83 against
   an oracle ceiling of 0.77-0.87), but it draws a good mask for only about half the objects it is
   clicked on: single-window recall 0.13 against a perfect model's 0.95; on dense hemibrain it
   draws 19 masks per window where a perfect model draws 120 (~16% recall).
2. **The window-combining rule was a real, separate loss, and is fixed.** The same masks scored
   0.83 with no combining and 0.17-0.18 after the first-come `canvas`/`propagate` rules; a perfect
   model loses half its precision through those rules and gets WORSE with more windows (0.41 ->
   0.21 at a quarter-window step). The agreement rule (`consensus`) puts a perfect model at the
   ceiling (0.75 vs 0.77, recall 0.96) and holds there with more windows; on the real model it
   gives 0.31 / 0.08, and the remainder is missing partners: 47% of pieces reaching a seam have no
   mask to join on the other side.
3. **Finer window steps help only with a rule that can use them.** Under the old rule step 64 made
   everything worse; under consensus a 50%-recall oracle goes from recall 0.71 to 0.88.
4. **Undertraining explains the pseudo-label precision too.** A perfect model that misses half of
   its objects per window, under consensus, scores exactly like the real model on three of the four
   volumes (0.67 / 0.45 / 0.43 vs 0.67 / 0.50 / 0.42).
5. **The v2 recipe targeted a non-problem** (item 2) and was retired; the loss ignored no voxel
   (item 6) and now does.
6. **The mask loss trained unlabelled voxels as background.** Fixed in version 3 (`pooled_known`).

Version 3 therefore trains longer before anything else, and reads the model with the two numbers
that matter: single-window precision and recall (`calibration_probe.py`, `assembly_sweep.sh`'s
`single_tile` row) and the consensus block diagnostic, at 50k, 100k, 150k and 200k.

## The arms

Each is the full chain, differing from `base` in one knob of `[algorithm]`. **Version 3 runs only
`feat64`**: at 100k it tied or led every other head knob on validation IoU in both launches
(0.446 / 0.445 one-click), matched `deep4` on the labeller's per-candidate probe within noise, and
costs 0.40 s/step against `deep4`'s 0.50 -- a quarter more steps per hour, which is the one thing
an undertrained model needs. The suite returns once round 0 is good enough to label with.

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

Mask-generator settings at eval are fixed (`predict_eval.sh`: 14^3 clicks, gates 0.5/0.5, taken
from promptable_seg_v1/RESULTS.md; windows reconciled by agreement, `consensus`, since version 3,
because the earlier `propagate` rule measured worst of every rule on the GT blocks) and applied
identically to every arm; the size filter is the only fitted parameter, as it was for MWS. Every round is scored
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
  trustworthy on continued masks, which is what would let `propagate` be gated and revisited; it
  is a recipe change and stays out of this comparison.
- **Missing partners at seams.** With agreement-based assembly the remaining loss is objects found
  in one window and not in the next (47% of pieces reaching a seam on hemibrain). Only better
  per-window recall (the model) or a second look (a click inside the neighbour's piece, gated like
  any discovery) can close it; the second is not built.

## Version 1 (2026-09-11) and why it was stopped

The first launch trained every arm's round 0 to completion or near it, then stopped before any
data-engine round trained. Kept here as text; every artifact (632 GB of checkpoints, the sidecars,
the diagnostics) was deleted when the experiment was restarted on the v2 recipe below.

Round 0 on ground truth alone (val = the four held-out volumes, `first_voxel_iou` = one click):

| arm | step | first_voxel_iou | final_voxel_iou | first_iou_error | s/step (8 x B300) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 100k | 0.440 | 0.548 | 0.117 | 0.40 |
| `feat64` | 100k | 0.446 | 0.560 | 0.117 | 0.40 |
| `wide512` | 100k | 0.438 | 0.549 | 0.111 | 0.40 |
| `refine4` | 100k | 0.444 | 0.565 | 0.122 | 0.41 |
| `stride2` | 80k | 0.421 | 0.543 | 0.112 | 0.54 |
| `stride2_small` | 80k | 0.361 | 0.483 | 0.110 | 0.55 |
| `deep4` | 95k | 0.392 | 0.504 | 0.125 | 0.50 |
| `stride1` | 15k | 0.267 | 0.369 | 0.087 | 3.05 |

The four head knobs that finished moved nothing (0.438-0.446). `stride2_small` and `deep4` trailed
at equal steps; `stride1` is 7.5x the step cost of `base`.

**The round-0 labeller's pseudo-labels were imprecise**, scored against truth on one 512^3 block of
each of the four *fit* volumes (in-sample for the teacher, so an upper bound):

| teacher | masks kept | precision@0.5 | recall | merges |
| --- | ---: | ---: | ---: | ---: |
| `base` | 528 | 0.191 | 0.097 | 68 (13%) |
| `feat64` | 570 | 0.175 | 0.096 | 74 (13%) |
| `wide512` | 435 | 0.166 | 0.069 | 61 (14%) |
| `refine4` | 502 | 0.157 | 0.076 | 49 (10%) |

Per volume the picture was the same for every teacher: hemibrain ~0.11 with 45-69 merges at ~75%
claimed; kasthuri 0.2-0.5 with almost nothing claimed; liconn 0.22-0.33; zebrafish ~0.21-0.25.
The IoU head had predicted >= 0.7 for every kept mask, and their mean best-IoU against any true
object was 0.24.

**Two merge-aware post-hoc filters were built and measured** (`filter_sweep.sh`, `feat64` teacher,
same blocks) and neither helped:

| filter | masks | precision@0.5 | recall | merges |
| --- | ---: | ---: | ---: | ---: |
| gates only | 570 | 0.175 | 0.096 | 74 |
| `split_tiled_wholes` | 571 | 0.175 | 0.096 | 74 |
| `consistency_clicks=3`, top, >= 0.5 | 309 | 0.181 | 0.054 | 47 |
| `consistency_clicks=3`, top, >= 0.7 | 276 | 0.170 | 0.045 | 42 |
| `consistency_clicks=3`, best, >= 0.5 | 372 | 0.185 | 0.066 | 62 |

Tiling detected nothing -- the model never offers a merge's lobes as separate confident masks --
and click-consistency rejected good and bad masks at the same rate. Both filters remain in
`amg.py` (off by default) because they are correct and cheap; they are just not the fix.

**Diagnosis.** The failure is not merges (13%) but confidence: the IoU head is trained only on
prompts placed *inside a true object* -- an interior click or a noised box -- and never sees the
prompts segment-everything actually issues, which land on membranes, on unannotated space and
against object boundaries. On those its output is untrained, and it said 0.7 for masks worth 0.24.
No gate on an untrained number can be tightened into a filter. That is the v2 recipe's target.

> **Retracted 2026-09-12.** The per-candidate probe below shows the head *was* calibrated on the
> grid's prompts and the gated masks *were* precise inside a tile. The precision was lost after the
> gates, in tile assembly. The v2 recipe answered a question the data did not ask.

## Version 2 (2026-09-12) and why it was stopped

Same sweep, v2 prompting recipe, `samples_per_epoch = 100k`. Four arms finished round 0 and were
labelled; the other four were killed mid-round when the experiment was stopped. Every artifact
still exists at the time of writing (`sam1__*` run directories, `pseudo/`, `diag/`, `probe/`).

Round 0 (val = the four held-out volumes, step 100k; `first_iou` now includes boundary clicks):

| arm | first_iou | first_voxel_iou | final_iou | first_iou_error | offobject_pred_iou |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deep4` | 0.439 | 0.429 | 0.554 | 0.126 | 0.137 |
| `feat64` | 0.445 | 0.430 | 0.570 | 0.128 | 0.143 |
| `refine4` | 0.412 | 0.402 | 0.553 | 0.143 | 0.142 |
| `wide512` | 0.407 | 0.394 | 0.564 | 0.127 | 0.155 |

Training-set `first_iou` was the same as validation (0.42-0.44 in the last 10k-step window) and
the smoothed curve was still rising when the cosine schedule reached its floor: the models never
fit the four finetune volumes. `offobject_pred_iou` did not fall towards 0; it rose from ~0.05 to
~0.14 (it is the max over three candidates). `first_iou_error` stayed at 0.12-0.13 throughout.

**The round-0 pseudo-labels were exactly as imprecise as in v1**, on the same fit-volume blocks:

| teacher | masks kept | precision@0.5 | recall | merges |
| --- | ---: | ---: | ---: | ---: |
| `deep4` | 335 | 0.179 | 0.058 | 45 (13%) |
| `feat64` | 420 | 0.148 | 0.060 | 48 (11%) |
| `refine4` | 418 | 0.148 | 0.060 | 45 (11%) |
| `wide512` | 214 | 0.178 | 0.037 | 19 (9%) |

Over the 87 unlabeled volumes the sidecars were sparse: a median of 5-19 instances and 2-6% of
voxels claimed per 512^3 block, so round 1 would have trained on ground truth plus a sprinkle.

**Where the precision actually goes** (`calibration_probe.py`, every candidate of every grid
click on every tile of the same blocks, scored against the tile's connected components on the
mask grid exactly as `losses.mask_iou` does; `deep4` teacher, `feat64` within 0.01 of every row):

| volume | clicks | on object | passing / click | precision@0.5 of passing | mean true IoU | merges | head corr | head-top >= 0.5 | oracle >= 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hemibrain | 74088 | 0.99 | 0.16 | 0.895 | 0.77 | 2.2% | 0.75 | 0.54 | 0.57 |
| kasthuri | 10976 | 0.90 | 0.02 | 0.992 | 0.75 | 0.0% | 0.71 | 0.32 | 0.35 |
| liconn | 32928 | 0.72 | 0.68 | 1.000 | 0.90 | 0.1% | 0.79 | 0.89 | 0.92 |
| zebrafish | 49392 | 0.65 | 0.01 | 0.924 | 0.75 | 0.0% | 0.63 | 0.32 | 0.36 |
| pooled | 167384 | 0.83 | 0.21 | **0.964** | 0.85 | 0.8% | 0.76 | 0.53 | 0.56 |

Calibration of the head on grid prompts, pooled: predicted 0.7-0.8 -> true 0.78 (precision
0.93), 0.8-0.9 -> 0.88 (0.995), 0.9-1.0 -> 0.92 (1.0). Off-object clicks contributed 0.5% of the
passing candidates. So **96% of the masks the gates keep are precise inside their tile, and the
labelling that leaves the block is 15-18% precise**: the loss is in what happens between --
`propagate` continuations (accepted on coverage, not on the head, by design), canvas merging
across the 50%-overlapping tiles, and the fact that a neurite spanning three tiles is one true
object but at most a tile of any mask. Merges are created there too (0.8% before, 9-16% after).
The v1 diagnosis was wrong, and no prompting recipe or head knob could have changed this. Two
other facts from the probe: the gate pass rate differs 60x between volumes (0.68 candidates per
click on liconn, 0.01 on zebrafish), and the decoder answers only ~53% of on-object clicks with a
mask of IoU >= 0.5 even when the head picks well (oracle 56%), so recall is capped by the decoder.

Also on record: the `base` round-0 job slowed from 0.5 to ~8 s/step at step 70.2k with one rank
starved for data (`data_wait_frac_max` 0.5-0.94; five GPUs at 100% in collective waits, three at
40-70%), on a node the admins had closed for a driver-update rollout along with every other B300
node, which also left the whole pending chain unable to dispatch. The experiment was stopped and
all jobs killed at the user's request on 2026-09-12.

**The assembly sweep** (`assembly_sweep.sh`, 2026-09-12; `deep4` teacher, the same 512^3 blocks,
same gates 0.7 / 0.8, H100; `single_tile` = eight 288^3 blocks per volume, each exactly one tile,
so nothing is ever joined; `fragments` = true objects split over two or more pseudo ids each
holding >= 10%):

| assembly | masks | precision@0.5 | recall | merges | fragments | purity | mean best IoU | claimed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `propagate` (the labeller) | 337 | 0.184 | 0.060 | 14.2% | 15.3% | 0.86 | 0.27 | 34% |
| `canvas` | 373 | 0.169 | 0.061 | 3.5% | 13.3% | 0.90 | 0.25 | 33% |
| `none` | 766 | 0.072 | 0.053 | 26.1% | 14.6% | 0.80 | 0.13 | 33% |
| `edge_discard` + canvas | 92 | 0.413 | 0.037 | 4.3% | 1.8% | 0.92 | 0.43 | 7% |
| `edge_discard` + none | 110 | 0.345 | 0.037 | 12.7% | 1.9% | 0.87 | 0.36 | 7% |
| **`single_tile` (no assembly)** | 302 | **0.834** | 0.126 | 1.7% | 1.8% | 0.92 | 0.69 | 33% |
| `propagate`, window step 64 | 731 | 0.083 | 0.055 | 34.5% | 29.3% | 0.74 | 0.16 | 46% |
| `canvas`, window step 64 | 1170 | 0.059 | 0.062 | 27.8% | 18.7% | 0.80 | 0.11 | 41% |
| `edge_discard` + canvas, window step 64 | 114 | 0.368 | 0.038 | 4.4% | 2.2% | 0.91 | 0.42 | 10% |
| **`consensus`** (agree in the shared region, dispute -> unlabelled) | 265 | **0.306** | 0.078 | 4.2% | 10.0% | 0.91 | 0.34 | 33% |
| `consensus`, agreement 0.7 | 276 | 0.286 | 0.076 | 5.8% | 10.0% | 0.90 | 0.33 | 32% |
| `consensus`, window step 64 | 344 | 0.305 | 0.094 | 9.3% | 13.2% | 0.88 | 0.33 | 40% |

Per volume, precision@0.5 under `propagate` -> `single_tile`: hemibrain 0.127 -> 0.853, kasthuri
0.571 -> 1.000, liconn 0.240 -> 0.847, zebrafish 0.310 -> 0.667. Read:

- Same tiles, same gates: 0.83 precise before assembly, 0.17-0.18 after. The loss is the assembly.
- The dominant failure is **failed joins**, not merges. Under `canvas` only 3.5% of masks merge two
  objects, but 13% of true objects are split into pieces and the kept masks are 90% pure yet
  small against the whole object (mean best IoU 0.25). The canvas rule joins a new mask only when
  an existing id lies under half of it; a neurite entering a tile has at most its overlap half
  under old paint, so the join is a coin flip and the seam becomes a hard false boundary.
- `propagate` adds merges (14% vs 3.5%) without curing fragmentation (15%): the un-gated
  continuations spill, and they do not reach the pieces the grid found under new ids.
- `none` is a trap, not a control: painting only unclaimed voxels turns every later mask into a
  spill-enriched leftover, hence 26% merges.
- `edge_discard` removes both failure modes (1.8% / 4%) and doubles precision, but only to 0.41
  and at 7% coverage. The face test is literal: a mask that stops one cell short of a tile face
  passes, and is a truncated piece of an object that continues. A margin of a few cells should
  bring it near the single-tile 0.83; untested.
- Recall is the model's problem, not the assembly's: 0.126 with no assembly at all.
- **A finer window step makes the joined labellings worse, not better** (step 64 instead of 128:
  125 windows per block instead of 27, 5x the time; the finer lattice also covers a little more of
  each block, 1115 true objects against 1041). Under `canvas` precision fell from 0.169 to 0.059,
  merges rose from 3.5% to 28%, and 1170 masks were produced for 1115 true objects; `propagate`
  fell from 0.184 to 0.083. The rule paints first-come: a later mask only fills what is still
  free and inherits an id when half of it lies over old paint, so every window's spill is painted
  for good under a neighbour's id and nothing is ever revised. More looks multiply the mistakes
  and add none of the corrections. `edge_discard` is nearly insensitive to the step (0.413 ->
  0.368, coverage 7% -> 10%), as it should be, since it never joins. A combination rule that gets
  worse as evidence accumulates is the wrong rule; the fix is agreement (or a vote) across
  windows, not a finer step.
- **The agreement rule (`tile_merge = "consensus"`, `amg.consensus_labelling`) recovers part of
  the loss and stops the degradation.** Two windows' masks are one object only if they agree
  where both looked (IoU in the shared region >= 0.5, mutual best match); cells two windows assign
  to different objects are left unlabelled. Precision 0.17 -> 0.31, recall 0.06 -> 0.08,
  fragments 13% -> 10%, merges unchanged at 4%; per volume hemibrain 0.13 -> 0.19, liconn 0.24 ->
  0.50, zebrafish 0.31 -> 0.42, kasthuri 0.57 -> 0.67. A stricter agreement bar (0.7) changes
  nothing. At window step 64 the rule holds its precision (0.305) and gains recall (0.094) where
  the old rule collapsed, but merges rise to 9%: a spill painted where no other window drew a mask
  is never contradicted. Still far from the single-tile 0.83, and hemibrain still fragments (95 of
  573 objects). The rule's own bookkeeping says why (`consensus:` line of the run, hemibrain): of
  2839 masks reaching into a region shared with another window, 1504 were joined, 479 met no mask
  at all on the other side, and 856 "disagreed" with a best IoU of 0.03, i.e. touched a sliver of
  some neighbour and nothing of their own object. So 47% of the pieces that reach a seam have no
  partner to join: the object was found in one window and missed in the next. That is per-window
  recall, which no joining rule can repair; a second look (a click inside the known piece, placed
  in the neighbouring window and gated like any discovery) is the lever that remains at inference
  time. On liconn the same count is 670 joined / 65 disagreed / 129 unmet, and consensus reaches
  precision 0.50 with recall 0.49 there.
**The oracle control** (`oracle_sweep.sh`, `pseudolabel.py oracle`, 2026-09-12): the model
replaced by the ground truth inside every window -- one perfect mask per object component of
>= 512 voxels, at the mask stride -- pushed through the same rules on the same blocks. `r50` keeps
each window's perfect masks with probability 0.5, independently per window: a perfect model that
misses half of what it sees, which is what the real one does on three of the four volumes.

| rule | real `deep4` prec / recall | perfect model prec / recall | perfect model, 50% recall per window |
| --- | ---: | ---: | ---: |
| single window, no combining | 0.834 / 0.126 | 0.765 / 0.952 | 0.771 / 0.476 |
| `none` | 0.072 / 0.053 | 0.271 / 0.758 | |
| `canvas` | 0.169 / 0.061 | 0.406 / 0.841 | 0.387 / 0.629 |
| `canvas`, step 64 | 0.059 / 0.062 | 0.209 / 0.736 | |
| `edge_discard` + canvas | 0.413 / 0.037 | 0.721 / 0.736 | |
| `consensus` | 0.306 / 0.078 | **0.750 / 0.957** | 0.598 / 0.710 |
| `consensus`, step 64 | 0.305 / 0.094 | 0.743 / 0.955 | **0.690 / 0.882** |

Per volume under `consensus`, perfect model -> 50%-recall perfect model -> real model: hemibrain
0.85 -> 0.72 -> 0.19; kasthuri 0.83 -> 0.67 -> 0.67; liconn 0.59 -> 0.45 -> 0.50; zebrafish 0.61
-> 0.43 -> 0.42. Read:

- The metric's ceiling is ~0.77 pooled (0.87 on hemibrain), not 1.0: an object that leaves and
  re-enters a window is one truth object and two components, and masks live on a 4-voxel grid.
  The real model's single-window precision (0.83) is AT that ceiling; its single-window recall
  (0.13 against 0.95) is the whole gap. Undertraining is the underlying cause.
- `consensus` with perfect masks reaches the ceiling (0.750 vs 0.765) at recall 0.96, and holds
  it at step 64. The rule is no longer a loss. `canvas` halves a perfect model's precision (0.41)
  and a perfect model gets worse under it with more windows (0.21 at step 64): the old rule's
  failure needed no model noise at all.
- A perfect model with 50% per-window recall under `consensus` scores exactly like the real model
  on kasthuri, liconn and zebrafish. On hemibrain the real model is far below it (0.19 vs 0.72)
  because its per-window recall there is ~16%: 19 masks per window against the oracle's 120.
- With 50% recall, step 64 lifts recall from 0.71 to 0.88 under `consensus` (an object missed in
  one window is caught in another), so finer steps do pay once the rule can use the extra views.

- **A flaw in the training path, independent of assembly:** `pooled_masks` clamps labels at 0, so
  the mask loss treats unlabelled voxels (-1, everything the labeller did not claim) as "not this
  object". Trained on truncated pseudo-pieces, the model is taught a boundary at every window
  edge. Any restart must mask -1 out of the focal/dice losses and the IoU target.

What a restart has to change first is the tile assembly, not the model: join tile masks only
where two gated masks agree in their shared overlap (symmetric IoU, one partner each), give
`edge_discard` a face margin so a truncated mask cannot pass as complete, turn anything ambiguous
into ignore (-1) rather than a guess, and gate continuations before using them. `predict_eval.sh` uses the same `propagate` assembly, so the
leaderboard number is exposed to the same loss.

## Core changes this experiment needed

- `promptable/targets.pooled_known`, the `weight` argument of `focal_loss`, `dice_loss`,
  `mask_iou` and `voxel_mask_iou`, and `PromptableSegmentation._correction(known=...)`: unlabelled
  voxels (-1) are silence in the losses, the head's IoU target and the correction clicks; a cell's
  target is its object's share of the LABELLED voxels. All-ones weights reproduce the old numbers
  exactly (tested), so ground-truth-only training is unchanged. Tests in
  `tests/unit/test_promptable_seg.py`.
- `promptable/amg`: `tile_merge = "consensus"` (`consensus_labelling`, `tile_labelling`), with
  `agree_thresh` and `min_support`; the labeller's and the eval script's default. Tests in
  `tests/unit/test_amg.py`.
- `prediction.grid.VolumeGrid(steps_per_patch=...)` and the labeller's `--tile-step`: the window
  step is a parameter (default unchanged).
- `promptable_seg` / `promptable/targets`: the v2 prompt kinds -- `offobject_prob`,
  `boundary_prob`, `boundary_radius` (all default 0 / off, so every existing config is
  unchanged), the `object_offobject` slot flag, a head-only loss path for off-object slots, and
  the `offobject_pred_iou` / `offobject_fraction` metrics. Tests in
  `tests/unit/test_promptable_targets.py` and `test_promptable_seg.py`.
- `promptable/amg`: `split_tiled_wholes` and `consistency_clicks` (off by default) -- correct,
  measured, and not the fix; kept because they are cheap and may matter once the head is trained.
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
  `steps_per_patch` (default 2 = the half-window overlap every scored run used) makes the window
  step a parameter; reads are rounded up to a multiple of it so the native and output lattices
  stay in step. The labeller exposes it as `--tile-step` (output voxels). Tests in
  `tests/unit/test_predict.py`.

- `experiments/sam_lmd_v1/oracle_sweep.sh` / `pseudolabel.py oracle`: the same assembly rules
  fed perfect per-window masks cut from the ground truth (optionally kept with probability
  `--recall` per window), the control that separates the model's share of the loss from the
  rules'. CPU only. Results under `$STAGE/assembly_sweep/oracle/`.
- `experiments/sam_lmd_v1/assembly_sweep.sh`: the labeller's block diagnostic under every tile
  assembly (now including `consensus` and window step 64) plus a no-assembly single-tile reference; `pseudolabel.compare_labellings` gained
  `fragments`, `truth_best_share` and `pseudo_purity` (tested in `test_tools.py`) and the CLI an
  `--edge-discard` flag. Results under `$STAGE/assembly_sweep/<arm>/`.
- `experiments/sam_lmd_v1/calibration_probe.py` / `.sh`: every grid candidate of a teacher on the
  diagnostic blocks, with its predicted IoU, stability, true IoU (mask grid and voxel), merge
  partners and gate outcome; `summarize` prints calibration bins and a threshold sweep. Results
  under `$STAGE/probe/<arm>_r0/`. Its IoU arithmetic is pinned against a brute-force loop in
  `test_tools.py`.

Model-free tooling tests: `python -m pytest experiments/sam_lmd_v1/test_tools.py`.
