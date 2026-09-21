#!/usr/bin/env python3
"""Emit the run configs of sam_lmd_v1: one promptable-segmentation chain per arm, three rounds each.

Every arm is the same chain. Round 0 trains a Segment-Anything-style model (`promptable_seg`) on
the four ground-truth finetune volumes of lmd_ssl_v1, from the released DINOv3 LVD-1689M checkpoint
-- the exact encoder, data split, schedule and augmentation of lmd_ssl_v1's arm 2, with the
affinity head swapped for the promptable one. Rounds 1 and 2 are the data engine: the previous
round's model labels blocks of the 87 unlabeled volumes arm 1 pretrained on (`pseudolabel.py`),
and the model is warm-started and trained on ground truth plus those pseudo-labels.

Arms differ from `base` in exactly one architectural knob each, chosen for how much fine spatial
detail the mask head can express; see ARMS. Writing 24 near-identical TOMLs by hand is how "the
arms differ in one knob" quietly stops being true, so they are generated, and an arm's identity is
its entry in ARMS and nothing else.

    python experiments/sam_lmd_v1/make_configs.py            # write the TOMLs
    python experiments/sam_lmd_v1/make_configs.py --check    # regenerate in memory and diff

See README.md for the design and its caveats.
"""
from __future__ import annotations

import pathlib
import re
import sys
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
SPLITS = "experiments/lmd_ssl_v1"          # the 4/4 volume split this experiment inherits
PREFIX = "sam1__"                          # experiment_name prefix, used by the shell scripts

DINOV3_LVD = ("/groups/miaai/miaai/pretrained_models/dinov3/"
              "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

# ---- the protocol, copied from lmd_ssl_v1 -------------------------------------------------------
#
# Arms 1/2 finetuned 50k steps with the interpolating head and 50k more with the sub-pixel
# head, the split existing only because a zero-initialised sub-pixel head cannot train from a cold
# encoder. SAM has one decoder and nothing to stage, so round 0 is ONE 100k-step schedule at stage
# B's peak LR. Rounds 1 and 2 warm-start a model that has already converged under a decaying
# schedule, which is exactly stage C's situation, so they take stage C's lower peak.
R0_STEPS = 200_000
ROUND_STEPS = 50_000
WARMUP = 3_000
LR_R0 = 3.0e-4
LR_ROUND = 1.0e-4
MIN_LR_RATIO = 0.001
BATCH_PER_RANK = 1
DP_SHARD = 8                    # one full node, global batch 8, as every lmd_ssl_v1 finetune
WORKERS = 8
# 100,000 rather than the finetune stages' 1,000. Statistically identical -- sampling is random
# with replacement either way -- but every epoch boundary respawns the dataloader workers (~4 s,
# visible as a periodic `data_wait_frac_max` spike), and at global batch 8 a 1,000-sample epoch is
# 125 steps and a 10,000-sample one 1,250: the first v2 launch ran at 10,000 and the spikes were
# plainly visible. At 100,000 an epoch is 12,500 steps -- eight boundaries in a 100k-step round.
SAMPLES_PER_EPOCH = 100_000
VAL_SAMPLES = 32                # what arms 1/2 validated on; see lmd_ssl_v1/make_configs.py

# The encoder block of lmd_ssl_v1 arm 2, verbatim except for the attention kernel (see comment).
# Version 4 arms parametrise the patch size, the crop and the RoPE type (see `render`).
MODEL = '''[model]
name = "dinov3_vit3d"
img_size = %(crop)d            # %(grid)d^3 = %(tokens)d tokens at patch %(patch)d
patch_size = %(patch)d
in_chans = 1
embed_dim = 1024          # ViT-L/16, 303M parameters
depth = 24
num_heads = 16
n_storage_tokens = 4
layerscale_init = 1.0e-05 # NOT optional: the released weights were trained with LayerScale
mask_k_bias = true
pos_embed_rope_dtype = "fp32"
%(rope_note)s
pos_embed_rope_type = "%(rope)s"
# 0.0 where arms 1/2 wrote 0.1 -- functionally the same setting. DINOv3's drop path is SAMPLE-level
# stochastic depth (whole samples leave a block's residual branch), and at one sample per rank the
# subset is always the whole batch, so 0.1 dropped nothing in arms 1/2 either. Written as 0.0 so the
# config says what happens, and so it cannot silently switch on if the batch geometry ever changes.
drop_path_rate = 0.0
# SDPA, where arms 1/2 ran FlashAttention-4. The same function to floating-point reduction order;
# the kernel changes because `[trainer].compile` below cannot be combined with FA4 (inductor emits
# an undefined symbol inside flash-attn's CuTeDSL wrapper) and FA4 buys nothing at 4096 tokens.
use_fa4 = false
'''

INIT_LVD = f'''[init]
path = "{DINOV3_LVD}"
# The 2D (1024,3,16,16) patch kernel, averaged over RGB and spread over z; for a patch smaller than
# the checkpoint's 16 it is then pseudo-inverse resized (block sums), see utils.pretrained.
inflate_2d_to_3d = true
skip = ["rope_embed."]    # derived from `base` in init_weights; 2D and 3D split channels apart
strict = true
'''

ROPE_NOTES = {
    "superposition": (
        "# Forced by the initialisation, as in arm 2: the released weights are 2D and "
        "superposition is the\n# variant that inflates. Its depth term sits behind a "
        "ZERO-INITIALISED scalar, so\n# layerwise_lr_decay and patch_embed_lr_mult stay at 1.0 "
        "below, or z-position never learns."
    ),
    "vanilla": (
        "# 3D AXIAL RoPE (version 4): each axis owns a third of the rotary channels. The "
        "pretrained attention\n# weights expect the 2D channel layout, so the encoder starts "
        "perturbed and adapts; every version-4\n# arm shares this, so they compare with each "
        "other, not with "
        "version 3."
    ),
}

INIT_WARM = '''[init]
# A TRUE warm start: encoder, neck, prompt encoder and mask decoder, from the round that labelled
# this round's data. `target = "algorithm"` is what makes that possible -- the default `"model"`
# would load the encoder alone and hand the student a random head to rediscover.
#
# `prefix = ""` because the target moved a level down: an algorithm checkpoint stores its encoder
# under `model.` within the algorithm. `skip = ["encoder."]` because `promptable_seg` registers the
# encoder under two names (`model` and `encoder`), so the checkpoint stores it twice and the
# duplicate would otherwise have no home under `strict`.
target = "algorithm"
prefix = ""
skip = ["encoder."]
strict = true
path = "PREV_CHECKPOINT"  # resolved by submit.sh once the previous round has a run directory
'''

# The `base` arm's promptable head: configs/promptable_lmd.toml's settings, i.e. what the
# 150k-step reference run (promptable_lmd_v2) used. Each arm overrides exactly one of these.
BASE_HEAD: dict[str, object] = {
    "prompt_dim": 256,
    "decoder_depth": 2,
    "decoder_heads": 8,
    "num_multimask_outputs": 3,
    "mask_upscale": 4,
    "mask_feature_dim": 32,
    "mask_refine_depth": 2,
    "masks_per_sample": 16,
    "min_object_voxels": 512,
    "rounds": 3,
    "box_prob": 0.5,
    # The v2 prompting recipe (see README, "Version 1 and why it was stopped"). Every arm.
    "attention_backend": "sdpa",
}

HEAD_NOTES: dict[str, str] = {
    "prompt_dim": "width of the neck and the two-way decoder",
    "decoder_depth": "two-way transformer layers between prompt tokens and the image",
    "decoder_heads": "8 heads; head_dim = prompt_dim / 8",
    "num_multimask_outputs": "whole / part / subpart: the nesting a single click is ambiguous over",
    "mask_upscale": "masks come out at patch_size / mask_upscale voxels; 4 -> stride 4 = 32 nm",
    "mask_feature_dim": "channels of the upscaled feature volume the mask token is dotted against",
    "mask_refine_depth": "3-wide convolutions AT mask resolution after the sub-pixel expansion",
    "masks_per_sample": "objects drawn per crop, all decoded against one encoder pass",
    "min_object_voxels": "smaller objects are never prompted for: at stride 4 they pool to ~1 cell",
    "rounds": "one click or box, then two correction rounds (the reference uses 11)",
    "box_prob": "round 0 is a noised box with this probability, else a click",
    "correction_pairs": "each correction round adds a foreground AND a background click "
                        "(padding where that error is absent) instead of one or the other -- "
                        "the reference's recipe, Archit et al. 2025",
    "mask_prompt_prob": "probability a correction round also sees the previous mask; fed always "
                        "(1.0) the model leans on it and degrades when given points alone "
                        "(Archit et al. 2025)",
    "attention_backend": "the DECODER's kernel; sdpa is the one torch.compile accepts",
}

# One knob per arm. `stride2_small` is the deliberate exception -- it changes a data knob together
# with the stride, because the 512-voxel floor exists only because stride 4 pools a 64-voxel object
# to a single output cell; at stride 2 such objects become learnable, and that is the question.
ARMS: list[dict] = [
    dict(name="base", knobs={},
         blurb="the reference head: masks at stride 4 (32 nm), 32 mask features, 2 decoder layers, "
               "dim 256. Its round 0 is the arm-2 analogue (S2); its rounds 1-2 are S3."),
    dict(name="stride2", knobs={"mask_upscale": 8},
         blurb="masks at stride 2 (16 nm). Measured on this corpus, a PERFECT model at stride 4 "
               "caps at 0.755 IoU (0.555 on objects under 4k voxels) against 0.912 at stride 2 -- "
               "the single largest fine-detail ceiling in the head. 8x the decoder's output "
               "tensors."),
    dict(name="stride1", knobs={"mask_upscale": 16},
         blurb="masks at voxel resolution. The ceiling removed entirely, at 64x the decoder's "
               "output tensors; feasible only with a B300's memory, and expected to be the "
               "slowest arm by a wide margin."),
    dict(name="stride2_small", knobs={"mask_upscale": 8, "min_object_voxels": 64},
         blurb="stride 2 AND objects down to 64 voxels prompted for. The one arm that changes a "
               "data knob: the 512 floor was set because stride 4 pools 64 voxels into one cell, "
               "so at stride 2 the floor and the stride are the same question."),
    dict(name="feat64", knobs={"mask_feature_dim": 64},
         blurb="64 mask features instead of 32: the width of the fine feature volume every mask is "
               "read out of, and the number that multiplies every tensor after the expansion."),
    dict(name="refine4", knobs={"mask_refine_depth": 4},
         blurb="four refinement convolutions after the sub-pixel expansion instead of two -- the "
               "only layers that run AT mask resolution, with a receptive field across block "
               "seams of 9 cells instead of 5."),
    dict(name="deep4", knobs={"decoder_depth": 4},
         blurb="a four-layer two-way decoder instead of two: more attention between the prompt and "
               "the image before the mask is read out (the paper's data-engine model used three)."),
    dict(name="wide512", knobs={"prompt_dim": 512},
         blurb="a 512-wide neck and decoder instead of 256."),
    # ---- VERSION 4 (2026-09-13): the encoder's token against the neurite. Round 0 only, feat64
    # head, 3D axial RoPE. `arm1` and `arm3` both put a 64 nm token and a 16 nm mask cell on the
    # tissue; arm1 by reading at 4 nm (1 um window, 4096 tokens), arm3 by patch 8 at 8 nm (2 um
    # window, 32768 tokens). `arm2` is arm1 at twice the global batch. The 512-voxel object floor
    # is a physical size, so it is 4096 voxels at 4 nm.
    dict(name="arm1_4nm", knobs={"mask_feature_dim": 64, "min_object_voxels": 4096},
         nm=4, rope="vanilla", rounds=0,
         blurb="version 4 arm 1: the finetune volumes read at 4 nm (2x upsampled), so a token is "
               "64 nm and a mask cell 16 nm over a 1 um window; feat64 head; 3D axial RoPE."),
    dict(name="arm2_4nm_gb16", knobs={"mask_feature_dim": 64, "min_object_voxels": 4096},
         nm=4, rope="vanilla", batch=2, rounds=0,
         blurb="version 4 arm 2: arm 1 with two crops per rank, global batch 16 at the same LR."),
    dict(name="arm3_p8", knobs={"mask_feature_dim": 64},
         patch=8, rope="vanilla", rounds=0,
         blurb="version 4 arm 3: patch 8 at 8 nm -- the same 64 nm token and 16 nm mask cell as "
               "arm 1 (mask_upscale 4 at patch 8 is stride 2) over the full 2 um window, 32768 "
               "tokens per crop; the checkpoint's 16^3 patch kernel is pseudo-inverse resized to "
               "8^3 at load."),
    # The batch-size question on its own: version 3's geometry (8 nm, patch 16, 32 nm cells) with
    # 2 and 4 crops per rank, axial RoPE like the other version-4 arms, same LR.
    dict(name="arm4_8nm_gb16", knobs={"mask_feature_dim": 64}, rope="vanilla", batch=2, rounds=0,
         blurb="version 4 arm 4: version 3's geometry (8 nm, patch 16, 128 nm token, 32 nm cell) "
               "at two crops per rank, global batch 16, same LR; 3D axial RoPE."),
    # Arm 4 plus the three training changes taken from Archit et al. 2025 (Segment Anything for
    # Microscopy): twice the objects per crop (their most important hyperparameter), the
    # reference's positive+negative click pair per correction round, and the previous mask fed
    # back half the time. Everything else is arm 4, so arm 6 vs arm 4 is these three together.
    dict(name="arm6_8nm_gb16_musam",
         knobs={"mask_feature_dim": 64, "masks_per_sample": 32, "correction_pairs": True,
                "mask_prompt_prob": 0.5},
         rope="vanilla", batch=2, rounds=0,
         blurb="version 4 arm 6: arm 4 with three changes from Archit et al. 2025 -- 32 objects "
               "per crop (was 16), a foreground AND a background click per correction round, the "
               "previous mask fed back with probability 0.5 (was always)."),
    # Arm 6 with twice the objects again: does "more objects per crop" keep paying past 32?
    # Everything else identical to arm 6, so arm 7 vs arm 6 vs arm 4 is 64 vs 32 vs 16 objects
    # (the last also differing in the click recipe).
    dict(name="arm7_8nm_gb16_musam64",
         knobs={"mask_feature_dim": 64, "masks_per_sample": 64, "correction_pairs": True,
                "mask_prompt_prob": 0.5},
         rope="vanilla", batch=2, rounds=0,
         blurb="version 4 arm 7: arm 6 with 64 objects per crop instead of 32; otherwise "
               "identical (click pairs, mask fed back with probability 0.5, 8 nm, patch 16, "
               "global batch 16, 3D axial RoPE)."),
    # Arm 7 doubled again (2026-09-16 evening): does the per-crop object count keep paying past 64?
    # Identical to arm 7 except masks_per_sample 128, so arm 8 vs 7 vs 6 vs 4 is 128 vs 64 vs 32 vs
    # 16 objects. Arm 7 cost 1.4x arm 6 per step (18.2 vs 25.5 crops/s); expect ~1.3 s/step here.
    dict(name="arm8_8nm_gb16_musam128",
         knobs={"mask_feature_dim": 64, "masks_per_sample": 128, "correction_pairs": True,
                "mask_prompt_prob": 0.5},
         rope="vanilla", batch=2, rounds=0,
         blurb="version 4 arm 8: arm 7 with 128 objects per crop instead of 64; otherwise "
               "identical (click pairs, mask fed back with probability 0.5, 8 nm, patch 16, "
               "global batch 16, 3D axial RoPE)."),
    # Arm 8's recipe at half the crop edge (2026-09-21): 128^3 voxels = 1 um at 8 nm, 8^3 = 512
    # tokens per crop instead of 4096. A 1 um crop holds a few dozen objects of >= 512 voxels at
    # most (the 2 um windows held 7-119), so the 128 slots of arm 8 would mostly be padding: 32
    # slots here, arm 6's number. Eight crops per rank instead of two, so a step sees half of
    # arm 8's tissue (64 x 128^3 against 16 x 256^3 voxels) at the same 256 object decodes per
    # rank (8 x 32 against 2 x 128) and half the encoder tokens; 16 loader workers, as the other
    # arm above two crops per rank needed. RoPE normalises coordinates by the runtime grid, so
    # this model must be labelled and evaluated with 128-voxel windows (the `crop128` data
    # configs), never 256.
    dict(name="arm9_8nm_gb64_musam32_c128",
         knobs={"mask_feature_dim": 64, "masks_per_sample": 32, "correction_pairs": True,
                "mask_prompt_prob": 0.5},
         rope="vanilla", batch=8, rounds=0, crop=128, workers=16,
         blurb="version 4 arm 9: the click-pair recipe of arms 6-8 on 128^3 crops (1 um, 512 "
               "tokens) with 32 object slots and eight crops per rank (global batch 64); 8 nm, "
               "patch 16, 3D axial RoPE, mask fed back with probability 0.5."),
    # Arm 8's recipe at the largest cubic crop every GT volume can supply (2026-09-21): 352^3 voxels
    # = 2.8 um at 8 nm, 22^3 = 10648 tokens per crop (2.6x arm 8's 4096). kasthuri15 is 768 x 768 x
    # 362 voxels at 8 nm (100 sections of 29 nm), so 352 is the last multiple of the patch that fits;
    # 512 would drop it (TOO_SMALL_FOR_CROP). One crop per rank (global batch 8): the mask grid is
    # 88^3, so 128 object slots cost 1.3x arm 8's decoder cells per rank (87M vs 67M), two crops
    # would cost 2.6x. RoPE normalises by the runtime grid: label and evaluate with 352-voxel windows.
    dict(name="arm10_8nm_gb8_musam128_c352",
         knobs={"mask_feature_dim": 64, "masks_per_sample": 128, "correction_pairs": True,
                "mask_prompt_prob": 0.5},
         rope="vanilla", batch=1, rounds=0, crop=352, workers=16,
         blurb="version 4 arm 10: arm 8's recipe (128 objects per crop, click pairs, mask fed back "
               "with probability 0.5) on 352^3 crops (2.8 um, 10648 tokens), the largest cube "
               "kasthuri15 can supply, at one crop per rank (global batch 8); 8 nm, patch 16, "
               "3D axial RoPE."),
    dict(name="arm5_8nm_gb32", knobs={"mask_feature_dim": 64}, rope="vanilla", batch=4, rounds=0,
         workers=16,
         blurb="version 4 arm 5: as arm 4 at four crops per rank, global batch 32. 16 dataloader "
               "workers per rank instead of 8: at 8 the loader capped the node at ~28 crops/s, the "
               "same as arm 4, and the GPUs waited 15% of every step (measured at launch)."),
]

AUGMENT = '''[augment]
# BANIS' defaults, byte-identical to every stage of lmd_ssl_v1, so augmentation is not a confound
# between the affinity arms and these. For the data-engine rounds this is also the "noise" the
# student is trained under while the teacher labelled clean volumes (the NoisyStudent recipe); SAM
# training adds its own on top -- jittered boxes, random click positions, correction clicks drawn
# from the error region.
rotate = "inplane"
drop_slice_prob = 0.05
shift_slice_prob = 0.05
shift_magnitude = 10
intensity = true
mul_intensity = 0.1
add_intensity = 0.1
noise_scale = 0.5
'''

TRAINER = '''[trainer]
max_steps = %(max_steps)d
batch_size = %(batch)d            # per rank; x %(dp_shard)d ranks -> global batch %(global_batch)d
lr = %(lr)s
warmup_steps = %(warmup)d
min_lr_ratio = %(min_lr_ratio)s
weight_decay = 0.05
grad_clip_norm = 1.0
layerwise_lr_decay = 1.0  # 1.0 for every arm: superposition RoPE's depth gate is a zero-init scalar
patch_embed_lr_mult = 1.0
lr_schedule = "linear"
precision = "bf16"
# Measured on this algorithm at this scale (experiments/promptable_seg_v1/RESULTS.md): the step is
# host-bound -- 48 small decoder passes per step -- and compilation is the one setting that moved
# it, 1.66x on B300. Requires the SDPA kernels selected in [model] and [algorithm].
compile = true
activation_checkpointing = true   # the head's expansion to mask resolution, where its memory goes
log_every = 100
val_every = 2500
checkpoint_every = 12500
num_workers = %(workers)d
seed = 0

[parallelism]
dp_replicate = 1
dp_shard = %(dp_shard)d              # one full B300 node
tp = 1
'''


def head_block(knobs: dict[str, object], patch: int = 16, nm: int = 8) -> str:
    settings = dict(BASE_HEAD)
    settings.update(knobs)
    notes = dict(HEAD_NOTES)
    stride = patch // int(settings["mask_upscale"])
    notes["mask_upscale"] = (f"masks come out at patch_size / mask_upscale voxels; "
                             f"{settings['mask_upscale']} -> stride {stride} = {stride * nm} nm")
    lines = ['[algorithm]', 'name = "promptable_seg"']
    for key, value in settings.items():
        literal = f'"{value}"' if isinstance(value, str) else str(value).lower() \
            if isinstance(value, bool) else str(value)
        marker = "   # <-- THIS ARM'S KNOB" if key in knobs else ""
        lines.append(f"{key} = {literal}{marker}  # {notes[key]}")
    return "\n".join(lines) + "\n"


def split_path(which: str, nm: int, crop: int = 256) -> str:
    """The data config of one split at one lattice and crop: lmd_ssl_v1's own at 8 nm / 256, a
    generated copy with `resolutions` replaced at 4 nm, or with `patch_size` replaced for another
    crop (`data/` beside this file)."""
    if nm == 8 and crop == 256:
        return f"{SPLITS}/lmd_{which}_singlescale.yaml"
    if crop != 256:
        assert nm == 8, "a non-256 crop is only generated at 8 nm"
        return f"experiments/sam_lmd_v1/data/lmd_{which}_singlescale_crop{crop}.yaml"
    return f"experiments/sam_lmd_v1/data/lmd_{which}_singlescale_{nm}nm.yaml"


# Volumes that cannot supply a crop of the given edge at 8 nm (measured 2026-09-21 from the stores'
# boxes and voxel sizes): kasthuri15 is 768 x 768 x 362 voxels (100 sections of 29 nm); every other
# GT volume exceeds 1000 voxels on each axis. An arm at that crop trains and validates without them.
TOO_SMALL_FOR_CROP: dict[int, tuple[str, ...]] = {
    512: ("kasthuri15_ac3", "kasthuri15_ac4"),
}


def drop_volume(text: str, name: str) -> str:
    """Remove one `- name: <name>` entry, with its indented keys, from a split YAML's volume list."""
    pattern = re.compile(rf"^- name: {re.escape(name)}\n(?:  .*\n)*", re.MULTILINE)
    new, n = pattern.subn("", text)
    assert n == 1, f"volume {name} not found exactly once"
    return new


def split_copies() -> dict[str, str]:
    """The generated split configs: the 8 nm / 256 YAMLs with `resolutions` rewritten (4 nm) or
    `patch_size` rewritten (the crops the arms use), nothing else."""
    files = {}
    for which in ("finetune", "val"):
        text = (HERE.parents[1] / SPLITS / f"lmd_{which}_singlescale.yaml").read_text()
        old = "resolutions:\n- - 8.0\n  - 8.0\n  - 8.0\n"
        assert old in text, f"{which}: resolutions block not found"
        new = ("# GENERATED by experiments/sam_lmd_v1/make_configs.py from " + SPLITS +
               f"/lmd_{which}_singlescale.yaml:\n# the same volumes, boxes and weights read at "
               "4 nm (the stores' 8 nm level 0, upsampled 2x), so a 256-voxel\n# window is 1 um "
               "and an "
               "encoder token 64 nm. Edit the generator, not this.\n"
               "resolutions:\n- - 4.0\n  - 4.0\n  - 4.0\n")
        files[f"data/lmd_{which}_singlescale_4nm.yaml"] = text.replace(old, new, 1)
        old_patch = "patch_size:\n- 256\n- 256\n- 256\n"
        assert old_patch in text, f"{which}: patch_size block not found"
        for crop in sorted({int(arm.get("crop", 256)) for arm in ARMS} - {256}):
            dropped = [v for v in TOO_SMALL_FOR_CROP.get(crop, ()) if f"- name: {v}\n" in text]
            remaining = text.count("\n- name: ") - len(dropped)
            note = ("" if not dropped else
                    "# " + ", ".join(dropped) + f" dropped: too small for a {crop}-voxel crop at "
                    f"8 nm (TOO_SMALL_FOR_CROP); the {remaining} remaining volumes are re-weighted "
                    "equally.\n")
            new_patch = ("# GENERATED by experiments/sam_lmd_v1/make_configs.py from " + SPLITS +
                         f"/lmd_{which}_singlescale.yaml:\n# the same volumes, boxes, weights and "
                         f"8 nm lattice, read in {crop}-voxel crops ({crop * 8 / 1000:g} um) "
                         "instead of 256. A model\n# trained on these crops must also be labelled "
                         "and evaluated with them: RoPE normalises coordinates by\n# the runtime "
                         "grid. Edit the generator, not this.\n" + note +
                         f"patch_size:\n- {crop}\n- {crop}\n- {crop}\n")
            copy = text.replace(old_patch, new_patch, 1)
            for v in dropped:
                copy = drop_volume(copy, v)
            if dropped:
                assert copy.count("  weight: 0.25\n") == remaining
                copy = copy.replace("  weight: 0.25\n", f"  weight: {1 / remaining:.6f}\n")
            files[f"data/lmd_{which}_singlescale_crop{crop}.yaml"] = copy
    return files


def data_blocks(round_index: int, nm: int = 8, crop: int = 256) -> str:
    if round_index == 0:
        source = f'config_path = "{split_path("finetune", nm, crop)}"'
        note = ("# The four ground-truth finetune volumes of lmd_ssl_v1 (kasthuri15_ac3, zebrafish "
                f"quadcube1,\n# liconn_mouse_dg, hemibrain_ellipsoid_body), equally weighted, "
                f"{nm} nm, patch {crop}.")
    else:
        source = 'config_path = "ROUND_CONFIG"'
        note = ("# The round's mixture -- the same four ground-truth volumes plus one entry per "
                "pseudo-labelled\n# block -- written by make_round_config.py after the labelling "
                "job, and substituted by submit.sh.")
    return f'''[data]
{note}
name = "miao_volumes"
{source}
samples_per_epoch = {SAMPLES_PER_EPOCH}
# FALSE, deliberately, against this repo's usual setting. With deferral on, the geometric
# augmentation runs on the device AFTER the dataloader worker has already drawn this strategy's
# prompts from the un-augmented labels (`PromptTargets` is a sample transform), so every click
# and box would be misregistered with the rotated image. In the workers, augmentation runs first
# and prompt sampling second, which is the only correct order. Deferral was measured as
# throughput-neutral on this algorithm anyway (promptable_seg_v1/RESULTS.md).
defer_image_ops = false

[val_data]
# The four held-out volumes, identical to arms 1/2's validation set. NOT a model selector: every
# round is scored at its final step, as arms 1/2 were, so nothing here leaks into the reported
# number.
name = "miao_volumes"
config_path = "{split_path("val", nm, crop)}"
samples_per_epoch = {VAL_SAMPLES}
defer_image_ops = false
'''


def render(arm: dict, round_index: int) -> str:
    name = arm["name"]
    patch = arm.get("patch", 16)
    rope = arm.get("rope", "superposition")
    nm = arm.get("nm", 8)
    batch = arm.get("batch", BATCH_PER_RANK)
    workers = arm.get("workers", WORKERS)
    crop = arm.get("crop", 256)
    model = MODEL % dict(patch=patch, crop=crop, grid=crop // patch, tokens=(crop // patch) ** 3,
                         rope=rope, rope_note=ROPE_NOTES[rope])
    if round_index == 0:
        steps, lr, init = R0_STEPS, LR_R0, INIT_LVD
        what = (f"round 0 -- ground truth only. From the released DINOv3 LVD-1689M checkpoint, "
                f"{R0_STEPS // 1000}k steps at lmd_ssl_v1 stage B's schedule. This is the arm-2 "
                f"analogue, and the teacher of round 1.")
    else:
        steps, lr, init = ROUND_STEPS, LR_ROUND, INIT_WARM
        what = (f"round {round_index} -- data engine. Warm-started from round {round_index - 1}'s "
                f"full model and trained on ground truth plus the blocks it pseudo-labelled, "
                f"{ROUND_STEPS // 1000}k steps at stage C's peak LR.")
    return f"""# sam_lmd_v1 arm `{name}`, {what}
#
# Arm: {arm['blurb']}
#
# GENERATED by experiments/sam_lmd_v1/make_configs.py -- edit that, not this.
experiment_name = "{PREFIX}{name}_r{round_index}"

{model}
{init}
{head_block(arm['knobs'], patch, nm)}
{data_blocks(round_index, nm, crop)}
{TRAINER % dict(max_steps=steps, lr=lr, warmup=WARMUP, min_lr_ratio=MIN_LR_RATIO,
                batch=batch, dp_shard=DP_SHARD, global_batch=batch * DP_SHARD,
                workers=workers)}
{AUGMENT}"""


def outputs() -> dict[str, str]:
    files = dict(split_copies())
    for arm in ARMS:
        for round_index in range(1 + arm.get("rounds", 2)):
            files[f"{arm['name']}_r{round_index}.toml"] = render(arm, round_index)
    return files


def main(argv: list[str]) -> int:
    check = "--check" in argv
    stale = []
    for name, text in outputs().items():
        if name.endswith(".toml"):
            parsed = tomllib.loads(text)         # a file that does not parse is a bug here
            assert parsed["algorithm"]["name"] == "promptable_seg"
        path = HERE / name
        path.parent.mkdir(exist_ok=True)
        if check:
            if not path.is_file() or path.read_text() != text:
                stale.append(name)
            continue
        path.write_text(text)
        print(f"wrote {name}")
    if check:
        if stale:
            print(f"{len(stale)} config(s) differ from their generator: {stale}", file=sys.stderr)
            return 1
        print(f"{len(outputs())} configs match their generator")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
