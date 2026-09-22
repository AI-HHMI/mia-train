# Does a Segment-Anything model beat affinity prediction on the lmd eval set -- and can it use the unlabeled corpus?

`lmd_ssl_v1` trained and scored two DINOv3 ViT-L affinity models on eight instance-segmentation
volumes: arm 1 (random init, SimMIM on 87 unlabeled volumes, then finetune) and arm 2 (the released
LVD-1689M checkpoint, then finetune), scored with mutex watershed and a fitted size filter at
pq 0.2369 and 0.2287. This experiment puts the repo's promptable segmentation strategy
(`algorithms/promptable_seg.py`, a Segment Anything for volumes) through **the same encoder, the
same 4/4 volume split, the same schedule and augmentation, and the same scoring protocol**, and
asks two things:

1. **SAM vs affinities**, like for like: the LVD checkpoint plus the GT finetune, with the affinity
   head replaced by the promptable one. This is round 0 -- the analogue of arm 2.
2. **Can SAM use the unlabeled corpus arm 1 pretrained on?** SAM needs instance labels; the 87
   volumes have none. The paper's answer is its data engine: a model trained on labelled data
   annotates unlabeled data, and the model is retrained on the result, iteratively. Rounds 1-2 of
   every arm are that engine, with the human out of the loop -- the analogue of arm 1, in the sense
   that it is the same unlabeled data used a different way. No engine round has trained yet: every
   version so far has been about making round 0 good enough to label with.

Versions 1-2 also swept the mask head's knobs; version 4 replaced that with arms on the encoder's
scale, the batch and the prompting recipe (below).

    bash experiments/sam_lmd_v1/submit.sh --smoke --rounds 0 feat64   # 20 steps, one GPU
    bash experiments/sam_lmd_v1/submit.sh --rounds 0 feat64           # VERSION 3: round 0 only
    bash experiments/sam_lmd_v1/submit.sh                             # every arm, every round
    bash experiments/sam_lmd_v1/tensorboard.sh                        # watch it
    bash experiments/sam_lmd_v1/predict_eval.sh feat64 0              # eight volumes -> instances
    bash experiments/sam_lmd_v1/score.sh feat64 0                     # -> the leaderboard table

**Status (2026-09-22; version 4: arms 1-9 done and scored, arms 3-9 on the leaderboard, arm 10
training).** Nine arms on the encoder's scale, the batch size and the training
recipe (below); version 3's single arm is superseded and its round-0 checkpoints remain for
reference. **Arm 8 (128 objects per crop, click pairs, mask feedback) leads the
`lmd_ssl_v1_neuron_instance` leaderboard at pq 0.2786, arm 7 (64 objects) is second at 0.2591, both
above the affinity rows 1c 0.2369 and 2c 0.2287**; arm 6 0.2244, arm 4 0.2094, arm 5 0.2038. Objects
per training crop is the lever: assembled block recall 0.343 -> 0.407 -> 0.450 -> 0.542 (arms 4, 6,
7, 8) at a flat precision of 0.57-0.60, and pq followed. Arm 3 (patch 8, mask stride 2) is the worst
arm on every diagnostic, last of the SAM rows at pq 0.1734, and costs 5-10x the compute. Arms 6-8 were predicted on H200 GPUs, arms 4
and 5 on B300 (see "GPU architecture" under Caveats).
Configs are generated: `make_configs.py` emits the TOMLs in this directory (the original head
sweep's 8 arms x 3 rounds, and the version-4 round-0 arms); an arm's identity is its entry in `ARMS`
and nothing else. `unlabeled_corpus.yaml` is a pinned copy of the data config arm 1 pretrained on,
because the corpus tooling has since been rebuilt and lists 189 volumes today.

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

- **One schedule for round 0, not 50k + 50k.** Arms 1/2 staged the finetune only because a
  zero-initialised sub-pixel affinity head cannot train from a cold encoder; SAM has one decoder and
  nothing to stage. Version 3 doubled it to 200k (linear decay, checkpoints every 12.5k) because at
  100k the curves were still rising at the rate's floor. Rounds 1-2 take stage C's lower peak: a
  warm start continues a model that converged under a decaying schedule.
- **SDPA + `torch.compile` instead of FlashAttention-4.** FA4 cannot be compiled, compilation is
  worth 1.66x on B300 for this host-bound step, and FA4 buys nothing at 4096 tokens
  (promptable_seg_v1/RESULTS.md).
- **`defer_image_ops = false`**, against this repo's usual setting: with deferral the geometric
  augmentation runs on the device *after* the worker has drawn the prompts (`PromptTargets` is a
  sample transform on the un-augmented labels), so every click and box would be misregistered with
  the rotated image. In the workers the order is augmentation first, prompt sampling second, and
  deferral was throughput-neutral here anyway.

`samples_per_epoch` is 100,000 rather than 1,000 (statistically identical under sampling with
replacement; it moves the worker respawn from every 125 steps to every 12,500) and `num_workers` is
8 rather than 6, since the workers also run the components pass and prompt sampling. Neither
touches what the model sees.

## How the model is prompted during training, step by step

`PromptableSegmentation._step` (`src/algorithms/promptable_seg.py`) with `PromptTargets`
(`src/algorithms/promptable/targets.py`) drawing the prompts in the dataloader worker. This is the
original ("v1") recipe, used by version 3 and version-4 arms 1-5; arms 6-10 add the click-pair
corrections and mask feedback described under Version 4. The retired v2 prompt kinds (below) are off. Per training crop of 256^3 voxels:

1. **Pick objects.** The worker splits the crop's labels into connected pieces and draws up to
   `masks_per_sample` = 16 objects at random from those of at least `min_object_voxels` (512 at
   8 nm, 4096 at 4 nm: the same physical size the labeller's size floor uses). A crop with no
   eligible object contributes nothing to the loss rather than being dropped.
2. **One first prompt per object.** With probability `box_prob` = 0.5 a single positive click at a
   voxel drawn uniformly inside the object; otherwise the object's bounding box with each corner
   jittered by normal noise of `box_noise` = 10% of that side, capped at `box_noise_max` = 20
   voxels, so the model also learns from boxes that are too loose or too tight. No negative
   click, no boundary or off-object click, in this round.
3. **Three candidates.** The decoder answers the first prompt with `num_multimask_outputs` = 3
   masks and a predicted IoU for each. The candidate with the lowest mask loss against the true
   object is the one that receives the gradient (`best_of`); the IoU head is trained on every
   candidate to predict its real IoU (squared error).
4. **Two correction rounds** (`rounds` = 3 in total). Each round adds exactly one click, drawn at
   random among the mask-grid cells where the current mask is wrong: a positive click in a cell
   the mask missed, a negative click in a cell it spilled into -- never in unlabelled space. The
   previous round's best mask is fed back to the decoder as logits, and the decoder now returns
   one mask, not three. An object whose mask is already right gets a padding token, not an
   invented click.
5. **Loss.** Per round, `focal_weight` = 20 x focal + `dice_weight` = 1 x dice over the LABELLED
   cells of the chosen candidate, plus `iou_weight` = 1 x the head's error; the three rounds are
   averaged. Unlabelled voxels (-1) are silence in all three terms since version 3.

So an object sees at most three prompts. TensorBoard's `first_iou` is the chosen candidate after
step 3, `final_iou` after the two corrections, `first_iou_error` the head's calibration; the
validation set (32 crops) reports the same. The gap to labelling time (next section) is steps 2
and 4: the labeller issues only single positive clicks on a grid, no boxes, no corrections, and
takes the model's first answer.

## How a checkpoint becomes labels, step by step

`PromptGridPredictor` (`src/algorithms/promptable/amg.py`) with the labeller's settings
(`LABEL_AMG` in `pseudolabel.py`; since 2026-09-15 the evaluation pass reads the same settings).
The same code labels the unlabeled corpus, scores the GT blocks, produces the pictures and writes
the leaderboard artifacts.

1. **Cut the block into windows.** The model only ever sees a window of 256 voxels on a side: 1 um
   at the 4 nm lattice, 2 um at 8 nm. Windows step by half a window, so every spot of tissue is
   seen by several; a 4 um block holds 343 windows at 4 nm and 27 at 8 nm.
2. **Encode each window once**; everything below reuses that embedding.
3. **Click on a regular grid.** `points_per_side = 14` cell-centred clicks per axis, 2744 per
   window, one every 73 nm at 4 nm (146 nm at 8 nm), every one a single positive point: no negative
   clicks, boxes or correction rounds, which exist only in training. Each click returns 3 candidate
   masks (`num_multimask_outputs`) with the IoU the head predicts for each, 8232 candidates per
   window, on the mask-cell grid (16 nm at 4 nm, 32 nm at 8 nm), expanded to voxels when painted.
4. **Gate every candidate on three tests.** Predicted IoU >= `pred_iou_thresh` (0.7); stability >=
   `stability_thresh` (0.8), i.e. the mask thresholded at logit +1 and at -1 must overlap by at
   least that IoU, so a boundary that moves under a small change of threshold is dropped whatever
   the head said; size between `min_mask_voxels` (4096 at 4 nm = 512 at 8 nm, a 64 nm cube) and
   `max_mask_fraction` (0.95) of the window. In arm 1 about 1.3 candidates per click survive.
5. **Remove nesting, then duplicates.** A mask lying >= `containment_thresh` (0.8) inside a larger
   one, and not a near-duplicate of it, is dropped in favour of the larger (`prefer = "whole"`: a
   nucleus inside a cell must not become a second object); then greedy NMS on mask IoU at
   `nms_iou` (0.7), the higher predicted IoU staying. Ten to thirty masks per window remain.
6. **Paint the window's own label map**, highest predicted IoU first; where two survivors still
   overlap, the first painted keeps the voxel (`tile_labelling`).
7. **Glue the windows** (`tile_merge = "consensus"`, `consensus_labelling`). For every pair of
   overlapping windows, a mask from one and a mask from the other are the same object if each is
   the other's best match in the shared region and their IoU there is >= `agree_thresh` (0.5).
   Joins are transitive (union-find), which is how one bad mask can glue two chains (Version 4).
   Each voxel takes the object its windows agree on; where they disagree it is left unlabelled
   (written as -1 in the sidecars, never 0); `min_support` (1) is how many windows must have seen a
   piece for it to be kept. The run prints, per block, how many overlap checks joined, disagreed,
   or found no mask in the other window.

Where each stage is measured: `calibration_probe.py` scores every candidate of step 3 against the
truth (per-candidate precision, calibration, gate pass rate); the `single_tile` row of the assembly
sweep scores one window after step 6; the `consensus` row scores the assembled block after step 7;
the gallery draws all of them.

## The data engine

The paper's fully automatic stage, with the model where the annotators were. Built and tested
(`pseudolabel.py`, `blocks.py`, `make_round_config.py`); no round has trained yet.

1. **Label.** `pseudolabel.py label` runs the segment-everything pass above over a block of an
   unlabeled volume with the labeller's gates. The gates were chosen for a training target, which
   is judged by precision: a missed object costs nothing (its voxels stay unclaimed and are never
   prompted for), a merged or truncated one is trained on.
2. **Store.** Labels go into a **sidecar** OME-Zarr per volume -- `raw` symlinked to the read-only
   published store, `labels/sam_rN` ours, on the source's own level-0 grid so miao co-registers
   them by construction. Everything the teacher did not claim reads as -1, never 0: 0 asserts
   "background", which a teacher with recall far below one cannot assert. Each block is its own
   `volumes:` entry with its own `bounding_box`, so ids need only be unique within one.
3. **Train.** `make_round_config.py` mixes the four GT volumes (weight 0.5) with every non-empty
   block (sharing 0.5); the round warm-starts from the model that labelled it (`[init].target =
   "algorithm"`) under the full augmentation, so the student sees noised inputs the teacher never
   did -- what makes iterated self-training more than a fixed point.
4. **Grow and repeat.** Blocks per volume 1 -> 4 (~1.7x, then ~7x the GT's voxel count), positions
   a fixed per-volume permutation, so round 2 relabels round 1's cells with the better model and
   adds new ones; the paper retrained six times for the same reason.

Three properties of `promptable_seg` make a partial labelling a usable target: prompts are drawn
only on ids > 0, so unlabelled space is never clicked on; a mask's target is "this object against
everything else", so a labelled neighbour is correctly negative; and, since version 3, unlabelled
voxels (-1) are **silence** in every loss, in the IoU the head is trained to predict and in the
correction clicks (`targets.pooled_known`, the `weight` argument of every function in
`promptable/losses.py`). Versions 1 and 2 lacked the third -- `pooled_masks` clamped labels at 0 --
so a round would have taught a boundary wherever a window ended; found by reading, it would have
capped every round.

**The diagnostic.** Every labelling also runs the labeller over blocks of the four GT volumes and
scores it against their truth (`pseudolabel.py diagnose`): precision of pseudo-masks at IoU 0.5,
merges (a pseudo-mask that two true objects each make >= 10% of), recall of true objects >= 512
voxels, claimed fraction, plus fragments, purity and swallowed objects. It is **in-sample for the
teacher**, so an upper bound on label quality, and it explains rather than selects: the rounds'
knobs are fixed above, not tuned on it.

## The v2 prompting recipe (retired in version 3; knobs default to off)

Version 2 added two round-0 prompt kinds to train the IoU head on the prompts the grid actually
issues: **boundary clicks** (`boundary_prob` = 30% of object clicks, drawn from the object's rim
within `boundary_radius` = 1 of another label) and **off-object clicks** (`offobject_prob` = 25% of
the slots: a positive click on label 0 with no target, training only the head to predict 0;
`offobject_pred_iou` in TensorBoard). It was built for a head-calibration failure that the
per-candidate probe later showed did not exist -- the head was calibrated; the precision was lost
in tile assembly (Version 2 below) -- and in an undertrained model it cost what is scarcest: a
quarter of the prompt slots trained only the head, a third of the remaining clicks were made harder
than the grid poses, and throughput fell (0.53 vs 0.40 s/step on `base`). The knobs remain in
`promptable_seg` / `PromptTargets`, off; the probe's `offPass` column measures what they were for,
and a small `offobject_prob` can return if background clicks start passing the gate.

## What versions 1 and 2 established

Everything below is measured on the same four 512^3 blocks of the finetune volumes (in-sample for
the model, so an upper bound) unless stated; the key tables are in the version sections further down.

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

## The head sweep (versions 1-2; superseded)

The first two versions swept the mask head's knobs, each arm a full chain differing from `base`
(stride 4 = `mask_upscale 4`, 32 mask features, 2 decoder layers, dim 256, the reference head of
`configs/promptable_lmd.toml`) in one `[algorithm]` knob: `stride2` (`mask_upscale 8`, 16 nm masks;
a *perfect* stride-4 model caps at 0.755 IoU against 0.912 at stride 2 on this corpus,
`sam3d/stride_ceiling.py`), `stride1` (`mask_upscale 16`, voxel-resolution masks, 64x the decoder
output), `stride2_small` (stride 2 and `min_object_voxels 64`, the one arm that also changed a data
knob), `feat64` (`mask_feature_dim 64`), `refine4` (`mask_refine_depth 4`, seam receptive field
5 -> 9 cells), `deep4` (`decoder_depth 4`) and `wide512` (`prompt_dim 512`). Measured twice, the
head knobs that finished moved nothing (Version 1 below), so version 3 kept only `feat64` -- it tied
or led every other head knob on validation IoU in both launches (0.446 / 0.445 one-click) at
0.40 s/step against `deep4`'s 0.50 -- and version 4 replaced the sweep with arms on the encoder's
scale, the batch and the training recipe. The 24 generated TOMLs (`<arm>_r{0,1,2}.toml`) remain in
this directory.

## Scoring

`predict.py` already writes an `instances` artifact plus the co-registered ground truth for a
promptable run, on the lattice the affinity artifacts were predicted on (same YAMLs, same patch).
The MWS leaderboard rows are scored from stored labellings through
`lmd_ssl_v1_neuron_instance_mws_{fit,test}.toml` -- a `size_filter` swept over
[0, 500, 5000, 50000] on the finetune half, applied once to the reported half -- so a SAM row goes
through exactly those files with no watershed. Same region, same metric, same fitted parameter:
**the SAM row and the MWS row differ only in what wrote the labelling.**

Mask-generator settings at eval are the labeller's, fixed (`predict_eval.sh` reads
`pseudolabel.LABEL_AMG`: 14^3 clicks, gates 0.7 / 0.8, NMS 0.7, windows reconciled by agreement),
applied identically to every arm; the size filter is the only fitted parameter, as it was for MWS.
Until 2026-09-15 the eval gates were 0.5 / 0.5 (an early sweep, within noise of 0.5 / 0.8; no
artifact was ever scored with them); one protocol means every diagnostic describes the pass that
is scored. Every round is
scored at its final step, as arms 1/2 were, so the validation set selects nothing.

**The window follows the run (2026-09-22).** `predict.py` refuses a data config whose `patch_size`
differs from the model's `img_size`, because RoPE normalises coordinates by the runtime grid, so a
model must be labelled with the crop it trained at. `predict_eval.sh`, `assembly_sweep.sh`,
`calibration_probe.sh` and `labelling_gallery.sh` therefore read `img_size` from the run's
`resolved_config.json` and, when it is not 256, switch to the generated split copies
`data/lmd_{finetune,val}_singlescale_crop<N>.yaml` and scale the click grid to keep the labeller's
click spacing of 256/14 = 18.3 voxels: 7 clicks per side at 128 (arm 9), 19 at 352 (arm 10), the
same rule SAM's generator applies across its crop layers. The single-window blocks of the sweep
shrink to the window plus 32 (160 for arm 9, as 288 is 256 + 32), so `single_tile` stays one
window; the assembled 512-voxel blocks and the leaderboard lattice are unchanged, so those numbers
compare across windows while the single-window row does not. Everything else -- gates, NMS,
consensus, size floor, the size filter fitted on the finetune half -- is identical.

## Caveats, in the order they would bite

- **GPU architecture is neither recorded nor constant.** The same checkpoint and inputs give
  slightly different labellings on different GPU generations (an earlier measurement on one block:
  520 vs 530 instances, pq 0.0503 vs 0.0486). Arms 4 and 5 were predicted on B300, arms 6-8 on H200
  (2026-09-21, when the B300 queue had one open node), arm 3 on B300, and the affinity rows on H100
  and L4; the mia-evals records do not say which. Re-predicting arms 4 and 5 on H200
  (`predict_eval.sh <arm> 0 --step 200000`, about 2.5 h wall) would put every SAM row on one
  architecture. Arm 3's mask stride 2 needs a B300: the labeller peaks above 141 GB and the probe
  needs `POINTS_PER_BATCH=16` (its default 64 peaks near 260 GB even there).
- **Init differs from arm 1.** Arm 1 started from random weights; every SAM chain starts from the
  LVD checkpoint, because the engine needs a labelled-data teacher to begin. So r2 vs arm 1 compares
  two ways of using the corpus from different starting points; **r2 vs r0 is the clean "does the
  unlabeled data help SAM" test**, and r0 vs arm 2 the clean "SAM vs affinities" one.
- **The fixed 0.7 gate is not neutral between arms with differently calibrated heads.** Arm 5's
  head under-predicts IoU by ~0.1 and arms 6-7's by ~0.04, and on the volume least like the training
  data almost nothing passes (expid82 pq 0.005-0.023 against arm 4's 0.057 and arm 8's 0.098).
  Fitting the gate per arm on the finetune half, as the size filter is, would be the fair protocol;
  it is a change to make once, deliberately, for every row.
- **The diagnostic is in-sample** (see above). The number that is not is the leaderboard row.
- **Label upsampling past 2^32 voxels** was wrong until 2026-09-15: `F.interpolate(mode="nearest")`
  on CUDA indexes its output with 32 bits, so on zebrafish doublecube1 (1920^3) 61% of the voxels
  came out wrong. `amg.upsample_cells` now repeats cells on the host with numpy (exact;
  `tests/unit/test_amg.py`). No table or picture here is affected: the scored blocks are at most
  1024^3.
- **Storage.** A round's sidecars are ~0.2-1 GB per block compressed; L2 is ~350 blocks. Delete a
  round's sidecars once its student is scored, and the `eval/` artifacts once their record exists.
- **Arms 1/2's `drop_path_rate = 0.1` was a no-op, and these configs write 0.0.** DINOv3's
  stochastic depth drops whole *samples* (`randperm(b)[:k]`, `k = max(int(0.9 b), 1)`), and at one
  sample per rank the subset is always the whole batch; 0.0 is the same function, stated honestly.
  Found because the degenerate `randperm(b)[:b]` trips an inductor pattern bug under
  `torch.compile`, which killed the first smoke run.
- **Training jobs run with `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0`.** Inductor's channels-last
  rewrite of conv graphs propagates a permuted layout into a cuDNN attention input, which torch 2.13
  does not constrain, and the job dies at its first compiled step. Off for every arm.
- **The mask-only refinement round** the paper trains is still missing from `promptable_seg`. It
  would make the IoU head trustworthy on continued masks, which is what would let `propagate` be
  gated and revisited; it is a recipe change and stays out of this comparison.

## Version 1 (2026-09-11) and why it was stopped

The first launch trained every head-sweep arm's round 0 to completion or near it (validation
one-click IoU at 100k: `base` 0.440, `feat64` 0.446, `wide512` 0.438, `refine4` 0.444; `stride2`
0.421 and `stride2_small` 0.361 at 80k, `deep4` 0.392 at 95k; `stride1` 0.267 at 15k and 7.5x
`base`'s 0.40 s/step), then stopped before any engine round trained. Its round-0 pseudo-labels were
imprecise on the four fit-volume blocks -- precision@0.5 0.16-0.19, recall 0.07-0.10, 10-14% merges
-- although the IoU head had predicted >= 0.7 for every kept mask. Two merge-aware filters were
built and measured (`filter_sweep.sh`): `split_tiled_wholes` changed nothing (the model never
offers a merge's lobes as separate confident masks) and `consistency_clicks` rejected good and bad
masks at the same rate; both remain in `amg.py`, off. The diagnosis at the time -- an IoU head
untrained on the prompts the grid issues -- was retracted on 2026-09-12, when the per-candidate
probe showed the head calibrated and the gated masks precise inside a tile: the precision was lost
in tile assembly (Version 2). Every artifact (632 GB of checkpoints, sidecars and diagnostics) was
deleted at the restart.

## Version 2 (2026-09-12) and why it was stopped

Same sweep, v2 prompting recipe, `samples_per_epoch = 100k`. Four arms finished round 0 (validation
one-click IoU 0.41-0.45, training IoU the same and still rising at the schedule's floor;
`offobject_pred_iou` rose rather than fell), their pseudo-labels were exactly as imprecise as
version 1's (precision@0.5 0.15-0.18, recall 0.04-0.06), and the sidecars claimed 2-6% of the
voxels per unlabeled block. Stopped 2026-09-12. What it was worth is the three measurements behind
the list above, all on the same four 512^3 blocks of the fit volumes with the `deep4` teacher and
gates 0.7 / 0.8.

**The per-candidate probe** (`calibration_probe.py`), pooled over 167,384 grid clicks: 0.21
candidates pass the gates per click and **96.4% of them are precise at IoU 0.5** (mean true IoU
0.85, 0.8% merges); the head is calibrated on grid prompts (predicted 0.7-0.8 -> true 0.78, 0.9-1.0
-> 0.92); off-object clicks are 0.5% of the passing set; the pass rate differs 60x between volumes
(0.68 per click on liconn, 0.01 on zebrafish); and the decoder answers only ~53% of on-object
clicks with a mask of IoU >= 0.5 even under an oracle pick (56%).

**The assembly sweep** (`assembly_sweep.sh`, H100; `single_tile` = one window per block, nothing
joined; `fragments` = true objects split over >= 2 pseudo ids each holding >= 10%):

| assembly | masks | precision@0.5 | recall | merges | fragments | purity | claimed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `propagate` (the labeller then) | 337 | 0.184 | 0.060 | 14.2% | 15.3% | 0.86 | 34% |
| `canvas` | 373 | 0.169 | 0.061 | 3.5% | 13.3% | 0.90 | 33% |
| `edge_discard` + canvas | 92 | 0.413 | 0.037 | 4.3% | 1.8% | 0.92 | 7% |
| **`single_tile` (no assembly)** | 302 | **0.834** | 0.126 | 1.7% | 1.8% | 0.92 | 33% |
| `canvas`, window step 64 | 1170 | 0.059 | 0.062 | 27.8% | 18.7% | 0.80 | 41% |
| **`consensus`** (agree in the shared region, dispute -> unlabelled) | 265 | **0.306** | 0.078 | 4.2% | 10.0% | 0.91 | 33% |
| `consensus`, window step 64 | 344 | 0.305 | 0.094 | 9.3% | 13.2% | 0.88 | 40% |

The first-come rules join a new mask only when an existing id lies under half of it, so a neurite
entering a tile is a coin flip and every window's spill is painted for good, which is why a finer
step made them worse. `consensus` (mutual best match with IoU >= 0.5 in the shared region, disputed
cells unlabelled) took precision 0.17 -> 0.31 and stopped the degradation with step; its hemibrain
bookkeeping -- 2839 masks reaching a shared region, 1504 joined, 479 unmet, 856 "disagreed" at a
best IoU of 0.03 -- put the remaining loss on per-window recall.

**The oracle control** (`oracle_sweep.sh`, `pseudolabel.py oracle`: the ground truth's own masks
inside every window, one per object component of >= 512 voxels at the mask stride; `r50` keeps each
window's masks with probability 0.5):

| rule | real `deep4` prec / recall | perfect model | perfect model, 50% recall per window |
| --- | ---: | ---: | ---: |
| single window, no combining | 0.834 / 0.126 | 0.765 / 0.952 | 0.771 / 0.476 |
| `canvas` | 0.169 / 0.061 | 0.406 / 0.841 | 0.387 / 0.629 |
| `canvas`, step 64 | 0.059 / 0.062 | 0.209 / 0.736 | |
| `consensus` | 0.306 / 0.078 | **0.750 / 0.957** | 0.598 / 0.710 |
| `consensus`, step 64 | 0.305 / 0.094 | 0.743 / 0.955 | **0.690 / 0.882** |

The metric's ceiling is ~0.77 pooled (0.87 on hemibrain): an object that leaves and re-enters a
window is one truth object and two components, and masks live on a 4-voxel grid. The real model's
single-window precision was at that ceiling; its recall was the gap. `consensus` reaches the ceiling
with perfect masks and holds it at step 64, a 50%-recall perfect model under it scores like the
real model on three of the four volumes, and a finer step then lifts recall 0.71 -> 0.88.

## Version 3 (2026-09-12/13): `feat64`, round 0, 200k steps

One arm, ground truth only, v1 recipe, loss ignoring unlabelled voxels (job 154276643, run
`sam1__feat64_r0_20260912_180421`, 0.42 s/step, 16 checkpoints). Validation one-click IoU per 25k
window rose 0.27 -> 0.46 (last evaluation 0.52), training IoU 0.27 -> 0.53: a train-val gap (~0.07)
opened for the first time and the curve was still rising at the schedule's floor. Label quality on
the same four GT blocks as every earlier table (H100, gates 0.7 / 0.8):

| checkpoint | single window prec / recall | consensus block prec / recall | merges | fragments | purity | claimed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | 0.668 / 0.145 | 0.214 / 0.085 | 8.0% | 7.9% | 0.88 | 33% |
| 100k | 0.758 / 0.135 | 0.256 / 0.089 | 7.7% | 8.4% | 0.90 | 31% |
| 150k | 0.803 / 0.148 | 0.380 / 0.107 | 5.8% | 11.5% | 0.90 | 40% |
| **200k** | **0.759 / 0.215** | **0.324 / 0.148** | 6.1% | 14.2% | 0.89 | 47% |
| v2 `deep4` 100k | 0.834 / 0.126 | 0.306 / 0.078 | 4.2% | 10.0% | 0.91 | 33% |
| perfect model | 0.765 / 0.952 | 0.750 / 0.957 | 2.6% | 12.8% | 0.90 | 78% |

Per volume at 200k, consensus precision@0.5 / merges / fragments: hemibrain 0.227 / 23 / 126,
kasthuri 0.414 / 2 / 1, liconn 0.562 / 3 / 7, zebrafish 0.308 / 1 / 14. Training longer improved
the labels mostly in recall (single window 0.135 -> 0.215 from 100k to 200k, block 0.089 -> 0.148)
at ceiling single-window precision; fragments grew with recall (87 -> 148) because more pieces
reach a seam without a partner. Probe: passing candidates per click 0.19 -> 0.50 at unchanged
precision (0.97), head-top candidate at IoU >= 0.5 for 0.54 -> 0.66 of on-object clicks (oracle
0.59 -> 0.71). A quarter-window step (64 instead of 128; 125 windows per block, 4.7x the time)
bought recall (0.148 -> 0.171) at the price of precision (0.324 -> 0.282), merges (29 -> 60) and
fragments (148 -> 206), so the half-window step stayed the default (`min_support = 2` at step 64 is
the untested knob against exactly this). What the assembled precision would cost a pseudo-label
round is bounded by purity (0.89) and merges (6%), since the loss ignores unlabelled voxels and
consensus leaves disputed cells unlabelled. The curve had not converged, and the train-val gap said
the four volumes were now being fit: more data, or a longer schedule, was the next lever.

### The single-crop tests (2026-09-13): can the model fit one crop at all?

`overfit/`: `feat64_r0.toml` with ONE fixed hemibrain crop as both training and validation data,
no augmentation, 2000 steps, one B300 GPU, from the LVD checkpoint. `8nm` is a 256^3 crop at the
experiment's resolution (token 128 nm, mask cell 32 nm); `4nm` is the central 1 um of the same
crop read at 4 nm (2x upsampled: token 64 nm, cell 16 nm). Same objects, same pipeline.

| test | one-click IoU (grid) | one-click IoU (voxel) | after 2 corrections | loss (round 0) |
| --- | ---: | ---: | ---: | ---: |
| `8nm`, mean of steps 1500-2000 | 0.49 | 0.46 | 0.57 | 0.51 |
| `4nm`, mean of steps 1500-2000 | 0.80 | 0.78 | 0.83 | 0.20 |
| reference: the 200k run on its training crops | 0.55 | 0.53 | 0.63 | 0.45 |

| `8nm_stride2` (mask cell 16 nm, token unchanged), at step 1500, killed | 0.57 | 0.55 | 0.62 | 0.41 |
| for comparison at step 1500: `8nm` / `4nm` | 0.49 / 0.83 | 0.43 / 0.79 | 0.57 / 0.87 | 0.55 / 0.21 |

Read: at 8 nm the model cannot fit even one crop it sees every step -- it reaches the same 0.5-0.6
the 200k run reached on the whole corpus, with the loss flattening -- while the same objects read
at twice the resolution fit to 0.8+ in the same 2000 steps. So the plateau is not data scale,
batch size, augmentation or schedule: it is what the model can represent when a neurite is about
one encoder token (128 nm) wide. `8nm_stride2` (mask cell 16 nm, token unchanged) separated the
two: a finer mask grid alone recovers a fraction (0.49 -> 0.57 at step 1500, against 0.83 at 4 nm),
in line with the `stride2` arms of versions 1 and 2. The token is the constraint; the mask head is
not where the fix is.

### The data at the token's scale (2026-09-13)

`figures/token_scale.py` (output: `$STAGE/figures/token_scale_slices.png`, `token_scale_stats.png`,
`token_scale_stats.json`) measures, on the single-window diagnostic block of each volume, how the
labelled objects compare with the encoder's 128 nm token. No model involved.

| volume | objects in 256^3 | local thickness, median | object voxels within 64 nm of another label | objects per occupied token | tokens holding >= 2 objects |
| --- | ---: | ---: | ---: | ---: | ---: |
| hemibrain | 104 | 121 nm | 52% | 2.11 | 62% |
| kasthuri | 54 | 120 nm | 53% | 2.02 | 58% |
| zebrafish | 99 | 56 nm | 82% | 2.07 | 74% |
| liconn | 26 | 167 nm | 42% | 1.31 | 28% |

Local thickness is twice the distance from an object voxel to the nearest voxel of a different
label, over all object voxels. Read with the probe's per-volume result (share of on-object clicks
answered at IoU >= 0.5, 200k: liconn 0.96, hemibrain 0.64, zebrafish 0.56, kasthuri 0.47): liconn
is the one volume where a token usually holds ONE object (72% of tokens) and objects are thickest,
and it is the one volume the model handles; the other three put two or more objects in most tokens.
The sharper contrast is crowding per token (1.3 vs 2.0-2.1), not thickness (1.4x), and thickness
alone does not order the other three, so the token-scale argument explains liconn-versus-the-rest,
not the full ranking.

## Version 4 (2026-09-13): the token against the neurite, on the real task

The single-crop tests said the version-3 model could not represent the masks at a 128 nm token
and could at 64 nm; the token-scale figures said liconn, the one volume it handles, is the one
where a token usually holds one object. Version 4 tests that on the whole corpus, with three arms
that differ from version 3 in the encoder's scale and share everything else: `feat64` head, LVD
start, v1 prompting recipe, loss ignoring unlabelled voxels, 200k steps linear 3e-4 -> 3e-7,
round 0 only, one B300 node each, and **3D axial RoPE** (`pos_embed_rope_type = "vanilla"`, each
axis a third of the rotary channels) instead of superposition. That last change was requested
for all three arms; it perturbs the pretrained attention weights, which expect the 2D channel
layout, so the arms compare with each other and not with version 3.

| arm | config | lattice | patch | tokens / window | token | mask cell | window | global batch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `arm1_4nm_r0.toml` | 4 nm (2x upsampled) | 16 | 4,096 | 64 nm | 16 nm | 1 um | 8 |
| 2 | `arm2_4nm_gb16_r0.toml` | 4 nm | 16 | 4,096 | 64 nm | 16 nm | 1 um | 16 (2 per rank) |
| 3 | `arm3_p8_r0.toml` | 8 nm | 8 | 32,768 | 64 nm | 16 nm | 2 um | 8 |
| 4 | `arm4_8nm_gb16_r0.toml` | 8 nm | 16 | 4,096 | 128 nm | 32 nm | 2 um | 16 (2 per rank) |
| 5 | `arm5_8nm_gb32_r0.toml` | 8 nm | 16 | 4,096 | 128 nm | 32 nm | 2 um | 32 (4 per rank), 16 workers |
| 6 | `arm6_8nm_gb16_musam_r0.toml` | 8 nm | 16 | 4,096 | 128 nm | 32 nm | 2 um | 16 (2 per rank); 32 objects/crop, click pairs, mask fed back at p = 0.5 |
| 7 | `arm7_8nm_gb16_musam64_r0.toml` | 8 nm | 16 | 4,096 | 128 nm | 32 nm | 2 um | as arm 6 with 64 objects/crop |
| 9 | `arm9_8nm_gb64_musam32_c128_r0.toml` | 8 nm | 16 | 512 | 128 nm | 32 nm | **1 um (128^3 crops)** | 64 (8 per rank), 16 workers; 32 object slots, click pairs, mask fed back at p = 0.5 |
| 10 | `arm10_8nm_gb8_musam128_c352_r0.toml` | 8 nm | 16 | 10,648 | 128 nm | 32 nm | **2.8 um (352^3 crops)** | 8 (1 per rank), 16 workers; arm 8's 128 object slots, click pairs, mask fed back at p = 0.5 |
| 8 | `arm8_8nm_gb16_musam128_r0.toml` | 8 nm | 16 | 4,096 | 128 nm | 32 nm | 2 um | as arm 7 with 128 objects/crop |

Arms 1 and 3 put the same 64 nm token and 16 nm cell on the tissue and differ in field of view
(1 vs 2 um), tokens per window (8x) and native vs interpolated voxels; arm 2 is arm 1 at twice the
samples per step at the same LR. Arms 4 and 5 (added 2026-09-13 evening, after arm 2's first hours
looked strong) ask the batch-size question at version 3's own geometry, 128 nm token and 32 nm
cell, at 2 and 4 crops per rank; against version 3 they differ only in the RoPE and the batch, so
together with arm 1 vs arm 2 they separate "the token was too coarse" from "the batch was too
small". Read a batch-16 arm at step N against a batch-8 arm at step 2N as well as at N. Arm 5
needed 16 dataloader workers per rank: at 8 its node delivered ~28 crops/s and the GPUs waited 15%
of every step (restarted after 300 steps, 2026-09-13 22:40). `min_object_voxels` is a physical
size, so 4096 voxels at 4 nm = 512 at 8 nm; `mask_upscale = 4` at patch 8 is stride 2, the 16 nm
cell. Arm 6 (added 2026-09-15) is arm 4 with the three training changes of Archit et al. 2025,
Segment Anything for Microscopy: `masks_per_sample` 16 -> 32 (their ablation's most important
hyperparameter), `correction_pairs` (each correction round adds a foreground click where the mask
missed AND a background click where it spilled, padding where that error is absent) and
`mask_prompt_prob = 0.5` (the previous mask is fed back half the time, so the model does not lean
on it); `final_iou` in its curves is measured half the time without the mask prompt and is
comparable only with itself. Arm 7 doubles the objects per crop to 64 and arm 8 (added 2026-09-16
evening, while arm 7 was at step 47k) to 128, changing nothing else: 8 vs 7 vs 6 vs 4 is 128 vs 64
vs 32 vs 16 objects, added because the training metrics moved with this knob while the downstream
effect was unknown; arm 7 cost 1.4x arm 6 per step (18.2 vs 25.5 crops/s), so arm 8 was given a 3x
wall (216 h). Arm 8 is the first arm under the 2026-09-16 layout: job scripts, resolved config and
LSF logs in `jobs/`, run directory in `runs/`, smoke artifacts in `jobs/smoke/` and `runs/smoke/`.

**Arm 9: the click-pair recipe on 128^3 crops** (config written 2026-09-21; trained 2026-09-21/22 in 25.5 h, job 154384975, 0.46 s/step; scored 2026-09-22 with 128-voxel windows and 7 clicks per side, see Scoring).
`img_size = 128` and the generated `data/lmd_{finetune,val}_singlescale_crop128.yaml` -- lmd_ssl_v1's
split YAMLs with `patch_size` 128 and nothing else changed -- give 8^3 = 512 tokens per crop instead
of 4096. Two settings move with the crop, by the user's decision: `masks_per_sample` 32 rather
than arm 8's 128, because a 1 um crop holds a few dozen objects of >= 512 voxels at most (the 2 um
windows held 7-119) and the rest of 128 slots would be padding; and 8 crops per rank rather than 2,
global batch 64, so a step sees half of arm 8's tissue (64 x 128^3 against 16 x 256^3 voxels) at
the same 256 object decodes per rank and half the encoder tokens. 16 loader workers, as arm 5
needed above two crops per rank. Because RoPE normalises coordinates by the runtime grid, this
model must be labelled and evaluated with 128-voxel windows: the `crop128` data configs cover the
model side, but the scoring scripts assume 256 windows throughout and need a small adaptation
before this arm can be scored. Wall x2 in `submit.sh` as a guess.

**Arm 9 on the leaderboard** (2026-09-22; record `sam1__arm9_..._r0_...step200000.size_filter`, B300
predictions with 128-voxel windows and 7 clicks per side): **pq 0.2017, eighth**, between arm 5
(0.2038) and arm 3 (0.1734); size filter 5000, finetune-half fit 0.360. Per volume against arm 8:
kasthuri15_ac4 0.398 / 0.487, liconn hippocampus 0.306 / 0.391, liconn expid82 0.017 / 0.098,
zebrafish doublecube1 0.086 / 0.139; components sq 0.694 / 0.687, rq 0.282 / 0.395, voi_merge
3.33 / 2.02, voi_split 0.98 / 1.19. So half the tissue per step at the same decodes per rank kept the
mask quality (sq) and lost recognition and merges -- the assembly-time fragmentation and self-
disagreement the block tables and the oracle galleries showed. **The region episode:** the 128
lattice fills more of each volume than the 256 lattice (the tail a 256 stride cannot complete:
kasthuri15_ac4 320x704x704 against 256x640x640), so the first record was scored over a larger
region, and the leaderboard -- which groups rows by scored region and refuses to rank different
regions together -- put it in a table of its own at pq 0.2040. `crop_artifacts.py` crops an arm's
predictions and co-registered truth to another arm's extents (origins are all 0, so only the tail
goes and the OME placement holds); `score.sh` takes `ART_DIR` / `SCRATCH_DIR` to score such a copy
(`eval/arm9_..._r0_lattice256/`, `score/arm9_..._r0_lattice256/`). The sub-region record is kept
at `score/arm9_..._r0/record_lattice128_region.json`. Any arm whose window is not 256 needs this
step before its row is comparable; arm 10 (352) will.

**Why arm 9's single windows leave objects unpainted** (2026-09-22, `probes/unpainted-window-objects/`
under the experiment root). In the gallery's central 128-voxel windows, 20% (kasthuri) and 15%
(hemibrain) of the GT voxels carry no label, against 1.5% and 0.6% in arm 8's 256-voxel windows.
Measured cause, object by object: a small share had no grid click inside them at all (2 of 10 and
8 of 18 unpainted objects, median 2.5-8k voxels in the window against one click per 6,114 voxels;
about 1% of GT voxels), and every other one was clicked -- up to 24 times -- and lost at the gate:
no candidate of any of its clicks had IoU head >= 0.7 and stability >= 0.8 on the same mask. Nothing
was removed later by NMS, containment or painting. On kasthuri the dropped masks were good (true IoU
0.75-0.81) with the head at 0.71-0.73 on one candidate and stability on another, i.e. the fixed gate
cutting a knife-edge; on hemibrain they were genuine failures (head 0.2-0.66, true IoU 0.17-0.77) on
thin processes the window cuts short. Assembly recovers most of it because eight half-overlapping
windows see every voxel, which is why the assembled recall (0.445) is far above the single window's.

**The oracle galleries: gluing error against within-window error** (2026-09-22,
`viz/oracle_w256/` and `viz/oracle_w128/`, `figures/oracle_gallery.sh`; CPU only). The same four
blocks, sections and pictures as an arm's gallery, but every window's masks are the ground truth's
own objects on the 4-voxel mask grid with perfect confidence, glued by the labeller's consensus rule;
`labelling_gallery.py oracle` refuses a split config whose `patch_size` is not the window, because
`VolumeGrid` reads each tile at the config's patch and a 128 window on the 256 config silently ran
at a 2x coarser lattice on the first attempt. Pooled precision@0.5 / recall, single window then
assembled: perfect masks at 256 windows 0.771 / 0.962 then 0.750 / 0.957 (1328 pieces for 1041
objects, 35 merges, 133 fragments, 0 swallowed); arm 8 0.762 / 0.686 then 0.596 / 0.542. Perfect
masks at 128 windows 0.885 / 0.962 then 0.742 / 0.958 (343 windows on hemibrain instead of 27, 34
merges, 140 fragments); arm 9 0.887 / 0.529 then 0.493 / 0.445. Three things follow. (1) Inside a
window the arms' per-piece precision IS the oracle's: the imprecision there is the mask grid and the
512-voxel floor (liconn and zebrafish cap near 0.6 even for perfect masks), not the model; the
model's within-window deficit is recall, 0.69 and 0.53 against 0.96. (2) The gluing rule itself is
cheap: perfect masks lose 0.5 points of recall and 2 (256) to 14 (128) points of precision in
assembly, and 343 windows glue as well as 27. (3) What the arms lose in assembly beyond that --
arm 8 17 points of precision and 14 of recall, arm 9 39 and 8 -- is the model disagreeing with itself
between overlapping windows (disputed voxels, fragments, 69 and 86 swallowed objects against 0).
So the levers are within-window recall (the gate, section above) and cross-window consistency, not
the reconciliation rule.

Pooled over the four blocks (precision@0.5 / recall; merges, fragments, swallowed are assembled counts):

| labelling | single window | assembled | merges | fragments | swallowed |
|---|---|---|---|---|---|
| perfect masks, 256 windows (`viz/oracle_w256`) | 0.771 / 0.962 | 0.750 / 0.957 | 35 | 133 | 0 |
| arm 8 (`viz/arm8_8nm_gb16_musam128_step200000`) | 0.762 / 0.686 | 0.596 / 0.542 | 56 | 184 | 69 |
| perfect masks, 128 windows (`viz/oracle_w128`) | 0.885 / 0.962 | 0.742 / 0.958 | 34 | 140 | 0 |
| arm 9 (`viz/arm9_8nm_gb64_musam32_c128_step200000`) | 0.887 / 0.529 | 0.493 / 0.445 | 41 | 264 | 86 |

Per volume the numbers are in each directory's `<volume>.json` (`window_scores`, `assembled_scores`).

**Arm 10: arm 8's recipe on 352^3 crops** (config written and submitted 2026-09-21; job 154388295
on i07u22, training since 14:22 at 1.91 s/step, MFU 0.14 as arm 8, so 200k steps land around
2026-09-26. Its first submission, 154387757, was dispatched ten seconds after the admins reopened
i07u22 and died with exit 127 before writing a log: the node could not yet reach /nrs.) The question is the encoder's context:
the same 128 objects, click pairs and mask feedback as arm 8, with 22^3 = 10648 tokens per crop
instead of 4096. 352 is the largest cube every GT volume can supply -- kasthuri15 is 100 sections
of 29 nm, 768 x 768 x 362 voxels at 8 nm, so 512 would have dropped it from training and
validation (`TOO_SMALL_FOR_CROP` in the generator records the measurement). One crop per rank
(global batch 8): the mask grid is 88^3, so 128 slots cost 1.3x arm 8's decoder cells per rank (87M
against 67M) and two crops would cost 2.6x. Smoke (20 steps, one B300): 483 s including
compilation, 600 TFLOP per step per rank against arm 8's 438 (1.37x), data wait ~0, no memory
error; the GPU peak was not logged. Measured 1.91 s/step at MFU 0.14 (arm 8's MFU, the upper end of the
1.1-1.9 s expected), so 4.4 days for 200k steps; wall 216 h. A 512^3 variant was configured and
smoke-tested first: its 128^3 mask grid is arm 3's, so it needs 32 object slots to stay within the
decoder budget; at that setting it fits a B300 and costs 958 TFLOP per step per rank (2.2x arm 8).
It was retired for the 352 arm because it could not include kasthuri. As with arm 9, this model
must be labelled and evaluated with 352-voxel windows; the scoring scripts do not support that yet.

**Ceilings at the 4 nm lattice** (`pseudolabel.py oracle`, perfect masks, 2026-09-14; the
reference for every arm-1/arm-2 number): single 1 um windows precision 0.885 / recall 0.988 pooled
(hemibrain 0.96, kasthuri 0.91, liconn 0.83, zebrafish 0.83); consensus on the 4 um blocks 0.760 /
0.983 (hemibrain 0.88, kasthuri 0.78, liconn 0.57, zebrafish 0.64). The assembled ceiling is the
same as at 8 nm (0.750) although a block now has 343 windows instead of 27: the agreement rule
does not lose precision on perfect masks as the seam count grows.

**Arm 1 at 200k** (scored 2026-09-14/15; `assembly_sweep/arm1_4nm_step200000/`,
`probe/arm1_4nm_r0_step200000/`, pictures under `viz/arm1_4nm_step200000/` from
`figures/labelling_gallery.py`). Probe, pooled: the head-top candidate reaches IoU >= 0.5 for 94.5%
of on-object clicks (version 3 66.4%, oracle 97.0%), 1.30 passing candidates per click (version 3
0.50), precision of passing 0.988. Single 1 um windows: 0.842 / 0.626 (ceiling 0.885 / 0.988;
version 3's 2 um windows 0.759 / 0.215), merges 2.4%, fragments 4.5%. Consensus on the 4 um blocks
(216-343 windows each, 3830 s per block on one B300): 0.370 / 0.443 (ceiling 0.760 / 0.983;
version 3 0.324 / 0.148), 1336 pieces for 1115 objects, merges 9.0%, fragments 19.6%, purity 0.828,
78% claimed; per volume hemibrain 0.373, kasthuri 0.355, liconn 0.554 (its ceiling is 0.57),
zebrafish 0.336. Overlap checks on hemibrain: 85% joined, 13% disagreed, 1.4% met no mask
(version 3: 71 / 24 / 5%): the missing-partner loss of versions 2 and 3 is gone, but one overlap in
eight still draws the same tissue differently, and a 4 um block has 343 windows instead of 27. The
pictures: on hemibrain gluing is transitive and ONE piece holds 62% of the block's true foreground
and most of 51 true objects, invisible to the `merges` column, so the table now also reports
`swallowed` -- true objects most of which sit inside a piece that also holds most of another
(hemibrain 98 of 573 = 17%, kasthuri 11%, zebrafish 9%, liconn 6%); inside a single window the
model already merges the two largest processes, so this is the model's dense-neuropil failure
amplified by the gluing. On kasthuri and zebrafish the pieces have the true shapes inside a window,
and the block turns them into two pieces per object (14% of overlap checks find no mask next door)
that leak into unlabelled space (52% and 27% of label-0 voxels claimed). Hence two separate fixes: a
merge guard for dense tissue, and for the sparse volumes a way to supply the missing partner plus
`min_support 2`. Every window's own labelling is saved beside the pictures
(`<volume>_windows.npz`), so rule variants can be tried offline in minutes.

**Arm 4 on the leaderboard** (2026-09-15; `predict_eval.sh` with the labeller's gates, `score.sh`;
record `sam1_arm4_8nm_gb16_r0_step200000`; pictures beside the affinity rows in
`$STAGE/viz/leaderboard_arm4_vs_mws/`, `figures/leaderboard_gallery.py`). Test half, unweighted
mean over the four held-out volumes: **pq 0.2094**, against 0.2369 (1c) and 0.2287 (2c); the size
filter fitted on the finetune half chose 5000 voxels (finetune-half pq 0.2948 / 0.2974 / 0.3075 /
0.2909 for none / 500 / 5000 / 50000; the affinity rows chose 50000 at 0.344, their unfiltered
output being 0.004). Per volume, arm 4 vs 1c vs 2c: kasthuri15_ac4 0.385 / 0.496 / 0.442; liconn
hippocampus 0.317 / 0.308 / 0.306; liconn expid82 0.057 / 0.056 / 0.082; doublecube1 0.078 / 0.088 /
0.085. Components: sq 0.691 vs 0.708 (the 32 nm cell staircase and the partial pieces), rq 0.295 vs
0.324 (fewer objects found), voi_merge 3.45 vs 2.08, voi_split 0.97 vs 1.31. Predicting the eight
volumes took 2 h on B300. Side by side on the same sections, the watershed labels every voxel with
smooth boundaries and merges through unannotated space; the SAM arm leaves a third of the objects
unlabelled, its boundaries step in 4-voxel cells, and its errors are merges of neighbouring
processes. The assessment of 2026-09-15: as an automatic segmenter the prompt-grid design is
structurally behind (coarse mask grid, click grid, transitive gluing, no use for prompts without a
user), and the next experiment should test a dense head with LSD targets on the same encoders.

**Arms 2 and 4 at 200k** (scored 2026-09-15, everything on B300, chained onto the training jobs by
`score_when_done.sh`; tables under `assembly_sweep/<arm>_step200000/`, `probe/<arm>_r0_step200000/`,
pictures under `viz/<arm>_step200000/`). Arm 2 (4 nm, global batch 16) against arm 1 (4 nm, batch 8):
single 1 um windows 0.832 / 0.698 (arm 1 0.842 / 0.626); assembled 0.406 / 0.496 (0.370 / 0.443),
merges 12.8% (9.0%), fragments 15.9% (19.6%), purity 0.83 (0.83), 80% claimed (78%); swallowed 181
of 1115 objects (arm 1 148): hemibrain 105, kasthuri 29, liconn 13, zebrafish 34. Probe head-top >=
0.5 0.965 (0.945), 1.43 passing candidates per click (1.30). Overlap checks on hemibrain 88 / 10 / 2% (arm 1 85 / 13 / 1.4). Validation 175-200k mean 0.666 vs 0.646. Doubling the batch at 4 nm buys a
little recall and a little assembled precision and costs merges; the hemibrain block is the same
one mega-piece. Arm 4 (8 nm, patch 16, batch 16, axial RoPE) against version 3 (8 nm, batch 8,
superposition RoPE): single 2 um windows 0.748 / 0.451 (version 3 0.759 / 0.215); assembled 0.483 /
0.343 (0.324 / 0.148), merges 4.7% (6.1%), fragments 19.7% (14.2%), purity 0.85 (0.89), 71% claimed
(47%); swallowed 68 of 1041 (hemibrain 57, kasthuri 4, liconn 0, zebrafish 7); probe head-top 0.911
(0.664), 1.08 per click (0.50); hemibrain overlaps 85 / 12 / 3 (71 / 24 / 5); validation 0.550 vs
0.476. Per volume assembled precision / recall: hemibrain 0.467 / 0.173 (version 3 0.227), kasthuri
0.583 / 0.485 (0.414), liconn 0.566 / 0.833 (0.562), zebrafish 0.431 / 0.508 (0.308). Batch and RoPE
are confounded in that comparison; arm 5 shares the RoPE. Across arms: arm 4, with 27 windows per
block, has the highest assembled precision so far (0.483) and the fewest swallowed objects (6.5%
against 13-16% for the 4 nm arms) despite the weakest per-window recall (0.451 against 0.63-0.70),
and in its hemibrain picture no single piece dominates; arm 2 has the highest recall at both levels.
Window count governs the merges, token size governs the recall (arm 3, which combines the 64 nm
token with the 2 um window, is below).

**Arm 5 at 200k** (scored 2026-09-16, B300; `assembly_sweep/arm5_8nm_gb32_step200000/`,
`probe/arm5_8nm_gb32_r0_step200000/`, `viz/arm5_8nm_gb32_step200000/`). Arm 5 is arm 4's geometry
at global batch 32 (4 crops per rank, 16 workers). Against arm 4: single 2 um windows 0.801 / 0.433
(arm 4 0.748 / 0.451); assembled 0.596 / 0.361 (0.483 / 0.343) -- the highest assembled precision
of any arm so far -- merges 5.5% (4.7%), fragments 14.6% (19.7%), swallowed 67 of 1041 (68),
purity 0.85, 74% claimed. Per volume assembled precision / recall: hemibrain 0.527 / 0.169 (arm 4
0.467 / 0.173), kasthuri 0.743 / 0.577 (0.583 / 0.485), liconn 0.629 / 0.847 (0.566 / 0.833),
zebrafish 0.574 / 0.538 (0.431 / 0.508). Validation IoU over the last 25k steps 0.556 vs 0.550:
the curve barely moved, the labels did. The probe says why: the IoU head under-predicts (mean
predicted 0.81 against true 0.92, error 0.20 vs arm 4's 0.13), so at the 0.7 gate fewer
candidates pass (0.96 per click vs 1.08) and those that do are right 99.5% of the time; the
head-top candidate hits the object for 94.9% of clicks (arm 4 91.1%). Batch 32 therefore buys a
more selective labeller rather than a better mask model, and the same mechanism that took recall
from 0.451 to 0.433 in a window took assembled precision from 0.48 to 0.60. Overlap checks on
hemibrain 85 / 12 / 3%, as arm 4.

**Arm 5 on the leaderboard** (2026-09-16; record `sam1_arm5_8nm_gb32_r0_step200000`): **pq 0.2038**,
fourth, just below arm 4's 0.2094 (1c 0.2369, 2c 0.2287); size filter 5000 again; finetune-half fit
0.3609 (arm 4 0.3075). Per volume against arm 4: kasthuri15_ac4 0.432 / 0.385, liconn hippocampus
0.321 / 0.317, zebrafish doublecube1 0.057 / 0.078, liconn expid82 **0.005 / 0.057**. Components:
sq 0.722 / 0.691, rq 0.278 / 0.295, voi_merge 3.94 / 3.45, voi_split 0.71 / 0.97. So the better
mask quality and fewer splits are real and carry to the held-out volumes it resembles, but on
expid82 the labeller nearly stopped labelling: 233 masks survived the gates over 320 windows
(arm 4: 2604), 115 objects for 1036 true ones, 4 true positives. The block tables did not see
this because expid82 is not a fit volume. The cause is the same head under-prediction the probe
reported: arm 5's head scores masks ~0.1 lower than they deserve, and on a volume that looks
unlike its training data almost nothing reaches the fixed 0.7 gate. The fixed gate is therefore not neutral between arms (see Caveats); a 0.6 gate for arm 5 is
roughly arm 4's 0.7.

**Arms 6, 7 and 8 at 200k** (scored 2026-09-21; diagnostics on H100 for the first time, which the
probe script's header argues is acceptable; `assembly_sweep/<arm>_step200000/`,
`probe/<arm>_r0_step200000/`, `viz/<arm>_step200000/`). All three are arm 4's geometry and batch
with the muSAM recipe (a positive and a negative click per correction round, the previous mask fed
back with p = 0.5) and 32 / 64 / 128 objects per crop. Single 2 um windows, precision / recall:
0.763 / 0.508, 0.763 / 0.548, 0.741 / 0.617 (arm 4 0.748 / 0.451, arm 5 0.801 / 0.433).
Assembled: 0.570 / 0.407, 0.595 / 0.450, 0.596 / 0.542 (arm 4 0.483 / 0.343, arm 5 0.596 / 0.361);
merges 5.6 / 5.0 / 5.9%, fragments 15.1 / 17.4 / 17.7%, swallowed 72 / 82 / 69 of 1041, claimed
75 / 78 / 79%. Arm 8 produces 946 assembled pieces for 1041 truth objects where arm 5 produced 631,
at arm 5's precision. Per block, assembled recall (arm 5 in brackets): hemibrain 0.190 / 0.225 /
0.349 (0.169), kasthuri 0.677 / 0.754 / 0.785 (0.577), liconn 0.889 / 0.861 / 0.889 (0.847),
zebrafish 0.613 / 0.673 / 0.744 (0.538). The price is hemibrain's merges, 33 / 31 / 40 assembled
(arm 5 26). Probe, pooled at the 0.7 gate: precision 0.992 / 0.993 / 0.993, candidates passing per
click 1.01 / 1.11 / 1.39 (arm 5 0.96), head-top candidate on the object for 94.3 / 95.6 / 96.9% of
clicks (arm 5 94.9%). Arm 8's IoU head is the first without a bias: mean predicted 0.911 against
true 0.913 (arm 5 0.814 vs 0.917; arms 6 and 7 0.867 vs 0.905-0.908). Passing masks from off-object
clicks rise with objects per crop, 25 -> 36 -> 49% of the kept set, so arm 8 also labels more of
the unannotated tissue.

**Arms 6, 7 and 8 on the leaderboard** (2026-09-21; records
`sam1__arm{6,7,8}_*_r0_*.step200000.size_filter`; predictions on gpu_h200 because the B300 queue
had one open node -- see the architecture caveat): **arm 8 pq 0.2786, first; arm 7 0.2591, second;
1c 0.2369; 2c 0.2287; arm 6 0.2244; arm 4 0.2094; arm 5 0.2038.** Size filter 5000 for all three;
finetune-half fit 0.364 / 0.405 / 0.419 (arm 5 0.361, 1c 0.344). Per volume, arm 8 against 1c:
kasthuri15_ac4 0.487 / 0.496, liconn hippocampus 0.391 / 0.308, liconn expid82 0.098 / 0.056,
zebrafish doublecube1 0.139 / 0.088. Components, arm 8 / 1c: sq 0.687 / 0.708, rq 0.395 / 0.324,
voi_merge 2.02 / 2.08, voi_split 1.19 / 1.31 -- fewer merges and fewer splits than the affinity
row with slightly worse mask quality; the lead is recognition. Arms 6 and 7 repeat arm 5's expid82
collapse (0.023 and 0.011; arm 7 is otherwise the best row on kasthuri, 0.504) and arm 8 does not
(0.098): their heads under-predict IoU by about 0.04 (arm 5 by 0.10), and the fixed 0.7 gate then
keeps almost nothing on the volume least like the training data, while arm 8's unbiased head is
gated as intended. The per-arm gate fit proposed under arm 5 would likely move arms 6 and 7 up.
Arm 7's voi_merge 3.33 against arm 8's 2.02 says the extra objects per crop reduced merges on the
held-out volumes as well as raising recall.

**Arm 3 at 200k** (scored 2026-09-21, B300; `assembly_sweep/arm3_p8_step200000/`,
`probe/arm3_p8_r0_step200000/`, `viz/arm3_p8_step200000/`). Patch 8 at 8 nm, so the mask grid has
2-voxel cells, 8x the logits per prompt of every other arm. Single windows 0.716 / 0.464 with the
fewest fragments (5.7%) and the highest purity (0.907) of any arm; assembled 0.404 / 0.368, the
lowest assembled precision of the 8 nm arms (zebrafish 0.307), merges 4.1%, fragments 13.5%.
Probe: gate precision 0.982 (every other arm 0.992-0.995), mean true IoU of kept masks 0.891,
head-top candidate on the object for 86.9% of clicks. So the finer grid does not survive gluing
and is not more precise per candidate either. It is also the expensive arm: 4.5x the prediction
time per volume (kasthuri15_ac4 553 s against arm 4's 122 s), 6-10x on the consensus blocks, GPU
memory above 141 GB in the labeller (the containment step asks for 50 GiB with 100 GiB in use) and
above 268 GB in the probe at its default 64 prompts per batch (`POINTS_PER_BATCH=16` was needed).
**On the leaderboard (scored 2026-09-21 21:45, B300 predictions): pq 0.1734, eighth, below every other SAM arm** (arm 5 0.2038) and above only the cc_threshold rows; size filter 5000, finetune-half fit 0.30 (arm 8 0.42). Per volume: kasthuri15_ac4 0.277, liconn hippocampus 0.292, liconn expid82 0.039, zebrafish doublecube1 0.085. Components: voi_merge 4.16 (the most merges of any row), voi_split 0.94, sq 0.711, rq 0.239. The stride-2 predictions took 12-15 h per large volume on a B300 (hemibrain needed its wall raised from 12 to 30 h).

**Arm 8's labels under a stricter gate and a two-window agreement rule** (2026-09-21,
`viz/arm8_8nm_gb16_musam128_step200000_{iou0.9,support2,iou0.9_support2}/`; the four GT blocks
assembled, everything else the labeller's defaults; baseline gate 0.7 / `min_support` 1 is
0.596 / 0.542 with 946 pieces, 56 merges, 184 fragments, 69 swallowed, 79% claimed). Gate 0.9:
0.631 / 0.194 -- 320 pieces, 11 merges, 93 fragments, 37 swallowed, 56% claimed; kasthuri and
zebrafish keep 15% of their objects. Two windows must agree (`min_support = 2`): 0.632 / 0.509 --
838 pieces, 30 merges, 139 fragments, 51 swallowed, 75% claimed. Both: 0.675 / 0.182. Neither knob
reaches the 0.8 assembled precision a pseudo-label was asked for. The gate removes merges (56 ->
11) but what it leaves is fragments and partials (93 of 320 pieces), because a neighbouring window
without a confident mask leaves the object half-claimed -- the shape of arm 4's 0.9 result -- so a
per-candidate precision of 0.999 (probe) becomes 0.63 assembled. Agreement halves the merges for a
three-point recall cost and is the setting to label with if labelling now; assembled precision
above ~0.65 needs objects completed across windows (a second look, or a larger overlap), not a
stricter gate. For training with unclaimed voxels as ignore, the harmful residue at support 2 is
the 30 merges (3.6% of pieces) and the fragment pairs, not the partials. (Run on H100, H200 and B300 respectively; the differences dwarf the architecture shift.)

**Filters on the assembled labels do not fix it** (2026-09-15). A stricter head gate (0.9 instead
of 0.7, `viz/<arm>_step200000_iou0.9/`) is a merge lever at 4 nm with a heavy recall price -- arm 2:
merges 175 -> 18, swallowed 181 -> 45, hemibrain's mega-piece gone, but recall 0.496 -> 0.262 and
precision only 0.406 -> 0.462, since half the surviving pieces are partials with no confident
neighbour to glue to -- and at 8 nm (arm 4) it removes the sparse volumes' labels almost entirely
(kasthuri 108 -> 6 pieces, zebrafish 313 -> 10) while hemibrain's confident merges survive it. The
three existing merge filters on arm 4's hemibrain block (`prefer = "part"`, `split_tiled_wholes`,
`consistency_clicks = 4`; baseline 212 pieces, 24 merges, 57 swallowed, recall 0.173) change
nothing (25 / 57 and 22 / 55) or cut swallowed objects to 12 only by discarding most large masks,
correct ones included (claimed 90% -> 50%, recall 0.094, precision 0.250). The merged mask is what
the model draws -- the correct small masks never exist as candidates -- so at 8 nm the window-level
merges are the model's belief, to be fixed in training (finer cells, more objects per crop) rather
than by filtering; the gluing then amplifies them (3% of objects swallowed per window, 10%
assembled).

Data configs at 4 nm are generated copies of lmd_ssl_v1's splits with `resolutions` rewritten
(`data/lmd_{finetune,val}_singlescale_4nm.yaml`); boxes are in the stores' own voxels and unchanged.
Reading them: `tensorboard.sh` (the five arms). Scoring them needs the diagnostics told the
lattice: for arms 1/2 the GT config is the 4 nm copy and a 1024-voxel block is the 512-voxel
block of every earlier table (`--gt-config`, `--block 1024`, single-tile blocks 576), and the
labeller's `min_mask_voxels` is 4096 there; arm 3 scores as before.

## Core changes this experiment needed

- `promptable/targets.pooled_known`, the `weight` argument of `focal_loss`, `dice_loss`,
  `mask_iou` and `voxel_mask_iou`, and `PromptableSegmentation._correction(known=...)`: unlabelled
  voxels (-1) are silence in the losses, the head's IoU target and the correction clicks; a cell's
  target is its object's share of the LABELLED voxels. All-ones weights reproduce the old numbers
  exactly (tested), so ground-truth-only training is unchanged. Tests in
  `tests/unit/test_promptable_seg.py`.
- `promptable/amg`: `tile_merge = "consensus"` (`consensus_labelling`, `tile_labelling`), with
  `agree_thresh` and `min_support`, the labeller's and the eval script's default;
  `split_tiled_wholes` and `consistency_clicks` (off by default -- correct, measured, and not the
  fix); `upsample_cells` on the host (the 2^32 bug under Caveats). Tests in `tests/unit/test_amg.py`.
- `promptable_seg` / `promptable/targets`: the retired v2 prompt kinds -- `offobject_prob`,
  `boundary_prob`, `boundary_radius` (all default 0 / off), the `object_offobject` slot flag, a
  head-only loss path for off-object slots and the `offobject_pred_iou` / `offobject_fraction`
  metrics; `mask_refine_depth` / `mask_upscale_hidden` reaching the decoder's sub-pixel expansion
  (the `refine4` knob); `correction_pairs` and `mask_prompt_prob` (arms 6-10; defaults reproduce the
  old behaviour). Tests in `tests/unit/test_promptable_targets.py` and `test_promptable_seg.py`.
- `prediction.grid.VolumeGrid`: `steps_per_patch` (default 2, the half-window overlap every scored
  run used) makes the window step a parameter, exposed by the labeller as `--tile-step`; builds for
  a volume with no `label_key`; and refuses a volume miao would read above pyramid level 0 rather
  than reading it from the wrong coordinates (all 87 unlabeled volumes are level 0 at 8 nm). Tests
  in `tests/unit/test_predict.py`.
- `utils.pretrained.resize_patch_kernel`: the released 16^3 patch kernel (RGB averaged, spread over
  depth) is resized to 8^3 for the patch-8 arm by FlexiViT's pseudo-inverse rule specialised to
  block-mean downsampling -- the SUM over each 2x2x2 block -- so the model starts as the pretrained
  one would respond to the same tissue at half resolution. Tests in `tests/unit/test_pretrained.py`.
- `layers/dinov3/block.py`: the stochastic-depth path is taken only when it would actually drop a
  sample. Numerically identical (at the whole-batch subset it was a gather, an `index_add` and a
  scale of 1), and it removes the `randperm[:b]` that inductor's `randperm_index` pattern cannot
  bind under `torch.compile` -- the failure that stopped every compiled run with `drop_path > 0`
  at batch 1. Test in `tests/unit/test_dinov3_layers.py`.
- Experiment tooling, model-free, tested by `python -m pytest experiments/sam_lmd_v1/test_tools.py`:
  `oracle_sweep.sh` / `pseudolabel.py oracle` (perfect per-window masks through the same rules,
  optionally kept with probability `--recall`; CPU only); `assembly_sweep.sh` (the block diagnostic
  under every assembly plus the single-tile reference; `compare_labellings` gained `fragments`,
  `truth_best_share`, `pseudo_purity` and `swallowed`); `figures/labelling_gallery.py` (image |
  truth | model coloured by true object | error map; every window's labelling saved as
  `<volume>_windows.npz`); `calibration_probe.py` (every grid candidate with predicted IoU,
  stability, true IoU, merge partners and gate outcome, pinned against a brute-force loop);
  `score_when_done.sh` chains sweep + probe + gallery onto a training job's end.
