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

**Status (2026-09-16; version 4: arms 1, 2, 4 and 5 done and scored, arms 4 and 5 on the
leaderboard at 0.2094 and 0.2038, arms 3, 6 and 7 training).** Seven arms on the encoder's scale, the
batch size and the training recipe (below); version 3's single arm is superseded and its round-0
checkpoints remain for reference. Arm 5 (8 nm, batch 32) has the best assembled labels so far,
0.596 / 0.361.
**First leaderboard row: `sam1_arm4_8nm_gb16_r0_step200000` pq 0.2094 against the affinity rows'
0.2369 (1c) and 0.2287 (2c)** -- third place, with fewer splits (voi_split 0.97 vs 1.31) and more
merges (voi_merge 3.45 vs 2.08), mask quality 0.69 vs 0.71, recognition 0.29 vs 0.32. Per volume:
kasthuri15_ac4 0.385 (1c 0.496), liconn hippocampus 0.317 (0.308), liconn expid82 0.057 (0.056),
zebrafish doublecube1 0.078 (0.088). Finetune half at each row's chosen filter: 0.308 vs 0.344. Arm 4 (8 nm, batch 16): assembled 0.483 / 0.343,
the highest precision and fewest swallowed objects so far; arm 2 (4 nm, batch 16): 0.406 / 0.496,
the highest recall. Arm 1 (4 nm) at 200k: single-window precision 0.842 / recall
0.626 (version 3: 0.759 / 0.215; ceiling 0.885 / 0.988), assembled pseudo-labels 0.370 / 0.443
(version 3: 0.324 / 0.148; ceiling 0.760 / 0.983), merges 9%, fragments 20%, purity 0.83. Recall
tripled; assembled precision barely moved, and the seam bookkeeping says why ("Version 4").

**Version 3 (2026-09-13, round 0 done).** Two full launches were stopped and their
artifacts deleted; what they taught is in "What versions 1 and 2 established" below, with every
number. Version 3 trained ONE arm (`feat64`) for round 0 only, 200k steps, the original prompting
recipe, with the mask loss fixed to ignore unlabelled voxels and the labeller reconciling windows
by agreement. Results in "Version 3" below: single-window recall 0.135 -> 0.215 from 100k to 200k
at ceiling precision, assembled pseudo-labels 0.32 precision / 0.15 recall (purity 0.89, merges
6%), curve not converged. The data-engine rounds and the other arms are the next decision.

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

## How the model is prompted during training, step by step

`PromptableSegmentation._step` (`src/algorithms/promptable_seg.py`) with `PromptTargets`
(`src/algorithms/promptable/targets.py`) drawing the prompts in the dataloader worker. This is the
original ("v1") recipe, used by version 3 and every version-4 arm; the v2 additions (below) are
off. Per training crop of 256^3 voxels:

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

So an object sees one click or one box and then at most two more clicks, positive or negative:
at most three prompts. TensorBoard's `first_iou` is the chosen candidate after step 3,
`final_iou` after the two corrections, `first_iou_error` the head's calibration; the validation
set (32 crops) reports the same. The gap to labelling time (next section) is steps 2 and 4: the
labeller issues only single positive clicks on a grid, no boxes, no corrections, and takes the
model's first answer.

## How a checkpoint becomes labels, step by step

`PromptGridPredictor` (`src/algorithms/promptable/amg.py`), with the labeller's settings
(`LABEL_AMG` in `pseudolabel.py`; since 2026-09-15 the evaluation pass reads the same settings).
The same code labels the unlabeled corpus, scores the GT blocks, produces the pictures and
writes the leaderboard artifacts.

1. **Cut the block into windows.** The model only ever sees a window of 256 voxels on a side:
   1 um at the 4 nm lattice, 2 um at 8 nm. Windows step by half a window, so every spot of tissue
   is seen by several. A 4 um block holds 343 windows at 4 nm and 27 at 8 nm.
2. **Encode each window once.** One encoder pass per window; everything below reuses that
   embedding.
3. **Click on a regular grid.** `points_per_side = 14` clicks per axis, cell-centred, so 14^3 =
   2744 clicks per window, one every 73 nm at 4 nm (146 nm at 8 nm). Every click is a single
   positive point. No negative clicks, no boxes, no second click on the same object, no correction
   round: the multi-round positive and negative clicks exist only in training. For each click the
   decoder returns 3 candidate masks (`num_multimask_outputs`), each with the IoU the head predicts
   for it, so 8232 candidates per window. Masks live on the mask-cell grid (16 nm at 4 nm, 32 nm
   at 8 nm) and are expanded to voxels when painted.
4. **Gate every candidate on three tests.** Predicted IoU >= `pred_iou_thresh` (0.7);
   stability >= `stability_thresh` (0.8) -- the mask
   thresholded at logit +1 and at -1 must overlap by at least that IoU, so a mask whose boundary
   moves under a small change of threshold is dropped whatever the head said; size between
   `min_mask_voxels` (4096 at 4 nm = 512 at 8 nm = a 64 nm cube) and `max_mask_fraction` (0.95)
   of the window. In arm 1 about 1.3 candidates per click survive, a few thousand per window.
5. **Remove nesting, then duplicates.** A mask lying >= `containment_thresh` (0.8) inside a
   larger one, and not a near-duplicate of it, is dropped in favour of the larger (`prefer =
   "whole"`: the targets are whole cells, so a nucleus inside a cell must not become a second
   object). Then greedy NMS on mask IoU at `nms_iou` (0.7): of two masks that overlap that much,
   the higher predicted IoU stays. Ten to thirty masks per window remain.
6. **Paint the window's own label map.** Highest predicted IoU first; where two survivors still
   overlap, the first painted keeps the voxel (`tile_labelling`).
7. **Glue the windows** (`tile_merge = "consensus"`, `consensus_labelling`). For every pair of
   windows that overlap, look only at the region both saw: a mask from one and a mask from the
   other are the same object if each is the other's best match there and their IoU in that region
   is >= `agree_thresh` (0.5). Joins are transitive (union-find): A = B and B = C makes one object
   of all three, which is how one bad mask can glue two chains (Version 4). Each voxel then takes
   the object its windows agree on; where the windows that see it disagree it is left unlabelled
   (written as -1 in the sidecars, never 0); `min_support` (1) is how many windows must have seen
   a piece for it to be kept. The run prints, per block, how many overlap checks joined, disagreed,
   or found no mask in the other window.

Two checks exist in the code and are off by default: `consistency_clicks` (click inside a
finished mask and ask whether the model gives the same mask back; a merge gives back a lobe) and
`split_tiled_wholes` (drop a mask that is the union of several other survivors). Both were
measured in version 2 against a model whose failure lay elsewhere; neither has been re-tried on
the version-4 models.

Where each stage is measured: `calibration_probe.py` scores every candidate of step 3 against
the truth (per-candidate precision, calibration, gate pass rate); the `single_tile` row of the
assembly sweep scores one window after step 6; the `consensus` row scores the assembled block
after step 7; the gallery draws all of them.

## The data engine

The paper's fully automatic stage, with the model where the annotators were:

1. **Label.** `pseudolabel.py label` runs `PromptGridPredictor` -- the same segment-everything
   pass `predict.py` uses, described step by step in the section above -- over a block of an
   unlabeled volume: 14^3 single positive clicks per window, masks kept only if the IoU head
   predicts >= **0.7** and the boundary is stable under a +-1 logit shift (>= **0.8**), NMS at
   0.7, windows reconciled by **agreement** (`tile_merge = "consensus"`; see the sweep below for
   why the earlier `propagate` rule was replaced). The gates were chosen for a training target, which is judged
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
rounds of one click each (foreground where the mask missed, background where it spilled); it is
spelled out step by step in "How the model is prompted during training" above. The v2
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

Mask-generator settings at eval are the labeller's, fixed (`predict_eval.sh` reads
`pseudolabel.LABEL_AMG`: 14^3 clicks, gates 0.7 / 0.8, NMS 0.7, windows reconciled by agreement),
applied identically to every arm; the size filter is the only fitted parameter, as it was for MWS.
Until 2026-09-15 the eval gates were 0.5 / 0.5, from a seven-setting sweep in
promptable_seg_v1/RESULTS.md on one 384-voxel block of liconn_expid82 with an early model, where
they beat 0.5 / 0.8 by 0.003 pq -- within noise -- and no artifact was ever scored with them; the
probe shows the 0.5 -> 0.7 step drops 7% of passing candidates that are right 62-77% of the
time, and one protocol means every diagnostic describes the pass that is scored. Every round is
scored at its final step, as arms 1/2 were, so the validation set selects nothing.

## Caveats, in the order they would bite

- **Label upsampling past 2^32 voxels.** Until 2026-09-15 the labeller brought its mask-cell
  labelling back to voxels with `F.interpolate(mode="nearest")` on the GPU, whose CUDA kernel
  indexes the output with 32 bits: on zebrafish doublecube1 (1920^3 = 7.1 G voxels) 61% of the
  voxels came out wrong and the artifact held 93 M negative ids, and mia-evals' size filter
  crashed on them (`np.bincount`, "negative elements"). Every smaller artifact was intact (the
  4.25 G-voxel quadcube1 included; the limit is 2^32 elements, 1626^3). `amg.upsample_cells`
  now repeats cells on the host with numpy. Measured with a 480^3 -> 1920^3 GPU test, exact
  against `repeat_interleave`; test in `tests/unit/test_amg.py`. No table or picture in this
  README is affected: the scored blocks are at most 1024^3.

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

## Version 3 (2026-09-12/13): `feat64`, round 0, 200k steps

One arm, ground truth only, v1 recipe, loss ignoring unlabelled voxels (job 154276643, run
`sam1__feat64_r0_20260912_180421`, 0.42 s/step, 16 checkpoints). Validation one-click IoU per 25k
window: 0.27, 0.33, 0.36, 0.40, 0.41, 0.41, 0.42, 0.46 (last evaluation 0.52); training IoU
0.27 -> 0.53, so a train-val gap (~0.07) opened for the first time and the curve was still rising
at the schedule's floor. Label quality on the same four GT blocks as every earlier table
(H100, gates 0.7 / 0.8):

| checkpoint | single window prec / recall | consensus block prec / recall | merges | fragments | purity | claimed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | 0.668 / 0.145 | 0.214 / 0.085 | 8.0% | 7.9% | 0.88 | 33% |
| 100k | 0.758 / 0.135 | 0.256 / 0.089 | 7.7% | 8.4% | 0.90 | 31% |
| 150k | 0.803 / 0.148 | 0.380 / 0.107 | 5.8% | 11.5% | 0.90 | 40% |
| **200k** | **0.759 / 0.215** | **0.324 / 0.148** | 6.1% | 14.2% | 0.89 | 47% |
| v2 `deep4` 100k | 0.834 / 0.126 | 0.306 / 0.078 | 4.2% | 10.0% | 0.91 | 33% |
| perfect model | 0.765 / 0.952 | 0.750 / 0.957 | 2.6% | 12.8% | 0.90 | 78% |

Per volume at 200k, consensus precision@0.5 / merges / fragments: hemibrain 0.227 / 23 / 126,
kasthuri 0.414 / 2 / 1, liconn 0.562 / 3 / 7, zebrafish 0.308 / 1 / 14.

**Window step 64 instead of 128** (a quarter window; 125 windows per 512 block instead of 27,
4.7x the labelling time), consensus:

| checkpoint | step | masks | prec / recall | merges | fragments | purity | claimed | s / block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100k | 128 | 363 | 0.256 / 0.089 | 7.7% | 8.4% | 0.90 | 31% | 71 |
| 100k | 64 | 522 | 0.222 / 0.104 | 8.6% | 12.7% | 0.87 | 38% | 390 |
| 200k | 128 | 476 | 0.324 / 0.148 | 6.1% | 14.2% | 0.89 | 47% | 139 |
| 200k | 64 | 677 | 0.282 / 0.171 | 8.9% | 18.5% | 0.85 | 54% | 654 |

Per volume at 200k, step 128 -> 64: hemibrain 0.227 -> 0.248, kasthuri 0.414 -> 0.322, liconn
0.562 -> 0.488, zebrafish 0.308 -> 0.205. The finer step buys recall (+16%) at the price of
precision (-13%), merges (29 -> 60) and fragments (148 -> 206): with a real model, every extra
window is also an extra chance for a spill mask to be painted where no other window contradicts
it, and at ~20-30% per-window recall the extra views bring fewer partners than the half-recall
oracle enjoyed (it gained recall 0.71 -> 0.88 AND precision). Not worth 4.7x the time at this
model quality; the half-window step stays the default. `min_support = 2` at step 64 is the untested
knob against exactly this failure (a cell claimed by one window only is dropped).

The per-candidate probe (100k -> 200k): candidates passing the gates per grid click 0.19 -> 0.50;
precision of what passes 0.970 -> 0.967 (at the ceiling both times); on-object clicks answered
with a mask of IoU >= 0.5 by the head's top pick 0.54 -> 0.66 (oracle pick 0.59 -> 0.71); head
correlation 0.83 -> 0.85; on hemibrain, passing candidates per click 0.12 -> 0.47. The share of
BACKGROUND clicks that pass the gate rose 0.050 -> 0.113 (liconn 0.14 -> 0.32) -- the quantity
the retired v2 recipe trained down to 0.003 -- but they contribute 6.7% of the kept masks and
the passing precision did not move, so most of them are masks of the neighbouring object.

Read:

- **Training longer improved the labels, mostly in recall.** Single-window recall 0.135 -> 0.215
  between 100k and 200k, block recall 0.089 -> 0.148, twice the objects found per window; precision
  held at the metric's ceiling for single windows (0.76 vs 0.77 for a perfect model) and rose from
  0.26 to 0.32 assembled. Against the version 2 model at its 100k end point, recall is up 70%
  (single window) and 90% (assembled).
- **It is still a long way from a perfect model's 0.95 recall**, and the assembled precision is
  low for the reason established before: pieces found in one window and missed in the next.
  Fragments grew with recall (87 -> 148) because more pieces reach a seam without a partner.
- **What the low IoU-precision now costs is less than it reads.** Since the loss ignores unlabelled
  voxels and consensus leaves disputed cells unlabelled, a truncated piece is a valid target for
  the voxels it covers and teaches no boundary at the seam. The numbers that bound the damage a
  pseudo-label round would do are purity (0.89: 11% of a mask's voxels belong to something else)
  and merges (6%), not precision@0.5 against whole objects.
- **The curve had not converged.** The last 50k steps produced the largest recall gain, the
  validation IoU was still rising at the floor, and the train-val gap says the four volumes are
  now being fit -- the point at which more data (a pseudo-label round, or a longer schedule with
  the same data) is the next lever.

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

Read: at 8 nm the model cannot fit even one crop it sees every step -- it reaches the same
0.5-0.6 the 200k run reached on the whole corpus, with the loss flattening. The same objects,
read at twice the resolution, fit to 0.8+ in the same 2000 steps. So the plateau is not data
scale, not batch size, not augmentation and not the schedule: it is what the model can represent
when a neurite is about one encoder token (128 nm) wide. The 4 nm test halves both the token and
the mask cell; `overfit_8nm_stride2.toml` (mask cell 16 nm, token unchanged) separated the two:
a finer mask grid alone recovers a fraction (0.49 -> 0.57 at step 1500, against 0.83 at 4 nm),
in line with the `stride2` arms of versions 1 and 2, which never beat the base head. The token
is the constraint; the mask head is not where the fix is. Killed at step 1500 on that evidence.

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
label, over all object voxels (a boundary-weighted proxy). Read with the probe's per-volume result
(share of on-object clicks answered at IoU >= 0.5, 200k: liconn 0.96, hemibrain 0.64, zebrafish
0.56, kasthuri 0.47): liconn is the one volume where a token usually holds ONE object (72% of
tokens) and objects are thickest, and it is the one volume the model handles. The other three
put two or more objects in most tokens. The contrast in thickness is about 1.4x, not an order of
magnitude; the sharper contrast is crowding per token (1.3 vs 2.0-2.1). Thickness alone does not
order the other three (zebrafish is thinnest yet mid-table; kasthuri is 29 nm sections upsampled
3.6x in z), so the token-scale argument explains liconn-versus-the-rest, not the full ranking.

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
| 8 | `arm8_8nm_gb16_musam128_r0.toml` | 8 nm | 16 | 4,096 | 128 nm | 32 nm | 2 um | as arm 7 with 128 objects/crop |

Arms 1 and 3 put the same token and the same mask cell on the tissue; they differ in field of
view (1 vs 2 um), tokens per window (8x) and native vs interpolated voxels. Arm 2 is arm 1 at
twice the samples per step at the same LR. Arms 4 and 5 (added 2026-09-13 evening, after arm 2's
first hours looked strong) ask the batch-size question at version 3's own geometry, 128 nm token
and 32 nm cell, at 2 and 4 crops per rank; against version 3 they differ only in the RoPE and the
batch, so together with arm 1 vs arm 2 they separate "the token was too coarse" from "the batch
was too small". Read a batch-16 arm at step N against a batch-8 arm at step 2N as well as at N.
Arm 5 runs 16 dataloader workers per rank instead of the usual 8: at 8 its node delivered ~28
crops/s, the same as arm 4 with half the crops per step, and the GPUs waited 15% of every step;
it was restarted with 16 after 300 steps (2026-09-13 22:40). `min_object_voxels` is a physical size, so it is 4096
voxels at 4 nm (= 512 at 8 nm); `mask_upscale = 4` at patch 8 is stride 2, i.e. the 16 nm cell.
Arm 6 (added 2026-09-15) is arm 4 with the three training changes taken from Archit et al. 2025,
Segment Anything for Microscopy: `masks_per_sample` 16 -> 32 (their ablation's most important
hyperparameter), `correction_pairs` (each correction round adds a foreground click where the mask
missed AND a background click where it spilled, padding where that error is absent, instead of one
or the other) and `mask_prompt_prob = 0.5` (the previous mask is fed back to the decoder half the
time rather than always; a model that always sees it leans on it and degrades when given points
alone). Both knobs are new algorithm arguments whose defaults reproduce the old behaviour, tested
in `tests/unit/test_promptable_seg.py`. Arm 6 against arm 4 is the three changes together;
`final_iou` in its curves is measured half the time without the mask prompt and is comparable
only with itself.
Arm 8 (added 2026-09-16 evening, while arm 7 was at step 47k) doubles the objects per crop once more, to
128, and changes nothing else: 8 vs 7 vs 6 vs 4 is 128 vs 64 vs 32 vs 16 objects. It exists because the
training metrics moved with this knob (arm 7's `first_iou` ran ahead of arm 6's at equal steps) while
the downstream effect is still unknown; arm 7 also cost 1.4x arm 6 per step (18.2 vs 25.5 crops/s), so
arm 8 is given a 3x wall (216 h) for its expected ~1.3 s/step. It is the first arm submitted under the
2026-09-16 layout: its job scripts, resolved config and LSF logs are in `jobs/`, its run directory in
`runs/`, its smoke artifacts in `jobs/smoke/` and `runs/smoke/`.

**How the patch-8 encoder is initialised.** The released kernel is `(1024, 3, 16, 16)`. The
loader averages RGB to one channel, spreads the kernel over 16 depth slices divided by 16 (a
z-constant volume reproduces the 2D response), and then, because the model's patch is 8, resizes
the 16^3 cube to 8^3 by FlexiViT's pseudo-inverse rule specialised to block-mean downsampling:
the least-squares kernel whose response to a half-resolution patch matches the full kernel's is
the SUM over each 2x2x2 block (`utils.pretrained.resize_patch_kernel`). It is exact for any patch
that is constant over those blocks, so the patch-8 model starts as the pretrained model would
respond to the same tissue seen at half resolution -- the 4 nm runs' trick folded into the first
layer. RoPE needs nothing: coordinates are normalised to the runtime grid, so 32^3 positions are
as valid as 16^3. Tests in `tests/unit/test_pretrained.py`.

**Ceilings at the 4 nm lattice** (`pseudolabel.py oracle`, perfect masks, 2026-09-14; the
reference for every arm-1/arm-2 number): single 1 um windows precision 0.885 / recall 0.988 pooled
(hemibrain 0.96, kasthuri 0.91, liconn 0.83, zebrafish 0.83); consensus on the 4 um blocks 0.760 /
0.983 (hemibrain 0.88, kasthuri 0.78, liconn 0.57, zebrafish 0.64). The assembled ceiling is the
same as at 8 nm (0.750) although a block now has 343 windows instead of 27: the agreement rule
does not lose precision on perfect masks as the seam count grows.

**Arm 1 at 200k** (scored 2026-09-14/15; `assembly_sweep/arm1_4nm_step200000/table.txt`,
`probe/arm1_4nm_r0_step200000/table.txt`; the probe and single-window rows ran on H100, the
consensus row on B300, so allow the ~2% architecture effect when comparing them). Probe, pooled:
the head-top candidate reaches IoU >= 0.5 for 94.5% of on-object clicks (version 3 66.4%, oracle
97.0%), 1.30 passing candidates per click (version 3 0.50), precision of passing 0.988. Single
1 um windows (32 blocks of 320 lattice voxels per volume): precision 0.842 / recall 0.626
(ceiling 0.885 / 0.988; version 3's 2 um windows 0.759 / 0.215), merges 2.4%, fragments 4.5%;
per volume hemibrain 0.92, kasthuri 0.87, liconn 0.88, zebrafish 0.78. Consensus on the 4 um
blocks (216-343 windows each, 3830 s per block on one B300): precision 0.370 / recall 0.443
(ceiling 0.760 / 0.983; version 3 0.324 / 0.148), 1336 pieces for 1115 objects, merges 9.0%,
fragments 19.6%, purity 0.828, 78% of the voxels claimed; per volume hemibrain 0.373 (88 merges /
100 fragments; ceiling 0.88), kasthuri 0.355 (17 / 38; 0.78), liconn 0.554 (3 / 11; 0.57, i.e.
at its ceiling), zebrafish 0.336 (12 / 70; 0.64). Seam bookkeeping on hemibrain: 58,511
mask-in-shared-region checks, 85% joined, 13% disagreed (mean best IoU 0.11), 1.4% met no mask in
the other window (version 3: 71% / 24% / 5% of 5,383). So the partner is now almost always there
-- the missing-detection loss that dominated versions 2 and 3 is gone -- but in one overlap in
eight the two windows still draw the same tissue differently, and a 4 um block has 343 windows
instead of 27, so every object crosses many more overlaps than before. Single-window quality is
at 95% of its precision ceiling and 63% of its recall ceiling; the assembled labels are at 49%
and 45% of theirs.

**What the pictures show** (`figures/labelling_gallery.py`; `$STAGE/viz/arm1_4nm_step200000/`;
the re-run on B300 reproduced the table exactly). The assembled failure differs per volume.
Hemibrain: gluing is transitive, and ONE piece ends up holding 62% of the block's true
foreground and most of 51 true objects. The `merges` column does not register it, because a
merge partner must be a tenth of the piece and no single object is, so the table now also
reports `swallowed`: true objects most of which sit inside a piece that also holds most of
another (hemibrain 98 of 573 = 17%, kasthuri 17 = 11%, zebrafish 28 = 9%, liconn 5 = 6%).
Inside a single hemibrain window the model already merges the two largest processes and labels
12 of 42 objects, so this is the model's dense-neuropil failure amplified by the gluing, not the
gluing alone. Kasthuri and zebrafish: inside a window the pieces have the true objects' shapes
and there are no merges; the assembled block turns them into two pieces per object -- 14% of
overlap checks find no mask at all in the neighbouring window -- and the pieces leak into
unlabelled space (the model claims 52% of kasthuri's and 27% of zebrafish's label-0 voxels). The
"spill" pieces are not slivers: median 19-29k saved voxels, purity ~0.8, i.e. partial pieces that
also bleed. Liconn: every small round profile is matched; the block's largest process is merged
with one neighbour. The zebrafish raw has blank sections, with interpolated ground truth there.
For the assembly work this means two separate fixes: a merge guard (refuse to glue a mask that
matches two masks in the neighbour, split it along their boundary instead) for dense tissue,
and for the sparser volumes a way to supply the missing partner (a second look that clicks the
neighbour inside the unmatched piece, or a denser click grid) plus dropping pieces only one
window ever saw (`min_support 2`). Every window's own labelling is saved beside the pictures
(`<volume>_windows.npz`), so rule variants can be tried offline in minutes.

**Arm 4 on the leaderboard** (2026-09-15; `predict_eval.sh` with the labeller's gates, `score.sh`;
record `sam1_arm4_8nm_gb16_r0_step200000` in mia-evals; pictures beside the affinity rows in
`$STAGE/viz/leaderboard_arm4_vs_mws/`, `figures/leaderboard_gallery.py`). Test half, unweighted
mean over the four held-out volumes: **pq 0.2094**, against 0.2369 (1c) and 0.2287 (2c); size
filter fitted on the finetune half chose 5000 voxels (finetune-half pq 0.2948 / 0.2974 / 0.3075 /
0.2909 for none / 500 / 5000 / 50000; the affinity rows chose 50000 at 0.344, their unfiltered
output being 0.004). Per volume, arm 4 vs 1c vs 2c: kasthuri15_ac4 0.385 / 0.496 / 0.442; liconn
hippocampus 0.317 / 0.308 / 0.306; liconn expid82 0.057 / 0.056 / 0.082; doublecube1 0.078 /
0.088 / 0.085. Components: sq 0.691 vs 0.708 (the 32 nm cell staircase and the partial pieces),
rq 0.295 vs 0.324 (fewer objects found), voi_merge 3.45 vs 2.08 (more merges), voi_split 0.97 vs
1.31 (fewer splits). Predicting the eight volumes took 2 h on B300 (doublecube1 twice: the first
artifact was corrupted by the interpolation bug in "Caveats"). The pictures put the two families
side by side on the same sections: the watershed labels every voxel, smooth boundaries, merges
through unannotated space; the SAM arm leaves a third of the objects unlabelled, its boundaries
step in 4-voxel cells, and its errors are merges of neighbouring processes. The assessment drawn
from this is in the discussion of 2026-09-15: as an automatic segmenter the prompt-grid design is
structurally behind (coarse mask grid, click grid, transitive gluing, no use for prompts without
a user), and the next experiment should test a dense head with LSD targets on the same encoders.

**Arms 2 and 4 at 200k** (scored 2026-09-15, everything on B300, chained onto the training jobs by
`score_when_done.sh`; tables under `assembly_sweep/<arm>_step200000/`, `probe/<arm>_r0_step200000/`,
pictures under `viz/<arm>_step200000/`). Arm 2 (4 nm, global batch 16) against arm 1 (4 nm, batch 8):
single 1 um windows 0.832 / 0.698 (arm 1 0.842 / 0.626); assembled 0.406 / 0.496 (0.370 / 0.443),
merges 12.8% (9.0%), fragments 15.9% (19.6%), purity 0.83 (0.83), 80% claimed (78%); swallowed 181
of 1115 objects (arm 1 148): hemibrain 105, kasthuri 29, liconn 13, zebrafish 34. Probe head-top >=
0.5 0.965 (0.945), 1.43 passing candidates per click (1.30). Overlap checks on hemibrain 88% joined /
10% disagreed / 2% unmet (arm 1 85 / 13 / 1.4), kasthuri 80 / 11 / 9.5 (71 / 15 / 14), zebrafish
85 / 6 / 9 (79 / 7 / 14). Validation 175-200k mean 0.666 vs 0.646. Doubling the batch at 4 nm buys a
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
Window count governs the merges, token size governs the recall. The arm that combines the 64 nm
token with the 2 um window is arm 3 (patch 8), still training.

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
unlike its training data almost nothing reaches the fixed 0.7 gate. The gate, fixed by protocol
for every arm, is therefore not neutral between arms with differently calibrated heads; fitting
it per arm on the finetune half, as the size filter is, would be the fair protocol and would
likely lift arm 5 above arm 4 (a 0.6 gate for arm 5 is roughly arm 4's 0.7). Not done: it is a
protocol change to make once, deliberately, for every row.

**A stricter head gate, 0.9 instead of 0.7** (2026-09-15, `viz/<arm>_step200000_iou0.9/`, whole
blocks, everything else unchanged). At 4 nm (arm 2) the gate is a merge lever with a heavy recall
price: pooled merges 175 -> 18, swallowed objects 181 -> 45, hemibrain's mega-piece gone (390 ->
180 pieces, 133 -> 16 merges), liconn's big process matched instead of merged; but recall 0.496 ->
0.262 and precision only 0.406 -> 0.462, because half the surviving pieces are partial -- pure,
under half their object -- where a neighbouring window had no confident mask to glue to. At 8 nm
(arm 4) the same gate removes the labels of the sparse volumes almost entirely (kasthuri 108 -> 6
pieces, zebrafish 313 -> 10; the head there rarely predicts 0.9) while hemibrain keeps its large
merged processes (swallowed 57 -> 32, precision 0.467 -> 0.508, recall 0.173 -> 0.110). So the
gate removes uncertain masks, and at 4 nm most merges are uncertain masks; at 8 nm the merges the
head is confident about survive it. For a pseudo-label at 4 nm, 0.9 is a defensible setting if
merges are the cost that matters and coverage is not; it is not a fix for the assembled precision.

**The three existing merge filters, on arm 4's hemibrain block** (2026-09-15, `viz/arm4_8nm_gb16_
step200000_{part,tiled,consist4,all3}/`; baseline assembled 212 pieces, 24 merges, 57 swallowed,
recall 0.173, 90% claimed). Keeping the contained mask instead of the container (`prefer = "part"`)
and dropping masks tiled by smaller confident ones (`split_tiled_wholes`) change nothing: 25 / 57 and
22 / 55. The merges are therefore not the dedup rules discarding correct small masks -- those small
masks never exist as candidates; the merged mask is what the model draws. Re-clicking inside each
mask (`consistency_clicks = 4`, reject if any click's top answer disagrees) cuts swallowed objects to
12 but does it by discarding most large masks, correct ones included: claimed 90% -> 50%, recall
0.173 -> 0.094, precision 0.467 -> 0.250. All three together equal the consistency run. At the
single-window level arm 4 swallows 4 of 119 objects (3%); assembled, 57 of 573 (10%), so at 8 nm
the gluing's amplification is the larger of the two layers, and the window-level merges themselves
are the model's belief, to be fixed in training (finer cells, more objects per crop) rather than
by filtering.

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
- `experiments/sam_lmd_v1/figures/labelling_gallery.py` / `.sh`: re-labels the four scored blocks
  and saves what the diagnostic discards -- image, truth, the central window's labelling, the
  assembled labelling (all at 8 nm) and every window's own labelling -- then draws image | truth |
  model coloured by the true object under each piece | error map (matched / partial / merge /
  spill / unlabelled truth). `compare_labellings` gained `swallowed` (tested in `test_tools.py`);
  `score_when_done.sh` chains sweep + probe + gallery onto a training job's end.
- `experiments/sam_lmd_v1/calibration_probe.py` / `.sh`: every grid candidate of a teacher on the
  diagnostic blocks, with its predicted IoU, stability, true IoU (mask grid and voxel), merge
  partners and gate outcome; `summarize` prints calibration bins and a threshold sweep. Results
  under `$STAGE/probe/<arm>_r0/`. Its IoU arithmetic is pinned against a brute-force loop in
  `test_tools.py`.

Model-free tooling tests: `python -m pytest experiments/sam_lmd_v1/test_tools.py`.
