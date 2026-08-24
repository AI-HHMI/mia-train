#!/usr/bin/env python3
"""Emit the eight run configs of lmd_ssl_v1 from one rule set.

Three arms differing in how the encoder is initialised, then ONE shared two-stage finetune. Writing
eight near-identical TOMLs by hand is how "the arms share a protocol" quietly stops being true, so
they are generated; an arm's identity is its entry in ARMS and nothing else.

    python experiments/lmd_ssl_v1/make_configs.py

See README.md for the design and its caveats.
"""
from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REL = "experiments/lmd_ssl_v1"
LMD = "/groups/miaai/miaai/lmd-v0.0.1/configs"

SSL_STEPS = 100_000
FT_STEPS = 50_000            # per stage; two stages
WARMUP = 3_000
LR_SSL = 3.0e-4
LR_INTERP = 3.0e-4
LR_SUBPIXEL = 1.0e-4         # lower peak rather than a shared schedule: the sub-pixel stage
                             # continues an encoder that has already converged under a decaying
                             # LR, and restarting at 3e-4 would undo that. Keeping the scheduler
                             # itself single-stage was the explicit preference.
MIN_LR_RATIO = 0.001

DINOV3_LVD = ("/groups/miaai/miaai/pretrained_models/dinov3/"
              "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

# ViT-L/16 geometry, shared by arms 1 and 2. `rope` is the only difference between them beyond
# initialisation, and it is forced by it: a 2D checkpoint can only be inflated under superposition.
DINOV3 = """[model]
name = "dinov3_vit3d"
img_size = 256            # 16^3 = 4096 tokens at patch 16
patch_size = 16
in_chans = 1
embed_dim = 1024          # ViT-L/16, 303M parameters
depth = 24
num_heads = 16            # head_dim 64; 64 %% 6 = 4, so axial 3D RoPE leaves 4 channels unrotated
n_storage_tokens = 4
layerscale_init = 1.0e-05 # NOT optional: the released weights were trained with LayerScale
mask_k_bias = true
pos_embed_rope_dtype = "fp32"
pos_embed_rope_type = "%(rope)s"
drop_path_rate = %(drop_path)s
"""

# MuViT sized to match ViT-L: 24 x (4 + 2*4) x 1024^2 ~= 302M, against DINOv3's 303M. Left at its
# defaults (512-dim, 12 deep) it would be ~40M, and a loss to arm 1 would be uninterpretable --
# model size, not objective or scale ladder.
MUVIT = """[model]
name = "muvit3d"
levels = [1, 2, 4]        # 1:2:4, matching the data config's 8/16/32 nm ladder
img_size = [256, 256, 256]
patch_size = [16, 16, 16]
in_channels = 1
embed_dim = 1024
depth = 24
num_heads = 16
mlp_ratio = 4.0           # ViT-L's ratio, not MuViT's default 2.0; see the parameter count above
# Explicit rather than "auto": auto silently falls back to SDPA when FlashAttention-4 is
# unavailable, and this arm's 3-level sequence is 12288 tokens, where attention dominates. A silent
# fallback would make the arm slower with no signal that it happened.
attention_backend = "flash4"
"""

AUGMENT = """[augment]
# BANIS' defaults, identical across all three arms so augmentation is not a confound.
#
# ARM 3 CAVEAT: `src/data/augment.py` transforms only `img` and `label`, never `bbox`, and MuViT
# takes its world coordinates from `bbox`. Two of these settings were checked against that:
#
#   rotate="inplane"  SAFE. Measured: the coordinate grid maps onto itself under an x<->y swap and
#                     under both flips, because miao's boxes are concentric and img_size is cubic,
#                     so the half-extents are equal on x and y. Augmentation also applies ONE
#                     transform to the whole (L, C, X, Y, Z) tensor, so every level turns together,
#                     and world_coords subtracts the crop centre first. A global rotation is a
#                     relabelling of a frame all levels share.
#
#   shift_slice       NOT geometrically clean for MuViT. It displaces content by a fixed number of
#                     VOXELS, and one voxel is a different physical distance at each level: a
#                     10-voxel shift is 10 world units at level 0, 20 at level 1, 40 at level 2.
#                     The levels therefore move by different physical amounts and stop being
#                     registered with each other -- exactly what the coordinates exist to prevent.
#                     Kept anyway, because holding augmentation identical across the three arms
#                     matters more than removing a small, realistic distortion from one of them,
#                     and dropping it for arm 3 alone would swap a known confound for an unknown
#                     one. FIRST THING TO ABLATE if arm 3 underperforms.
#
#   drop_slice        Fine: it removes data rather than moving it, so the coordinates still
#                     describe where the remaining data is.
rotate = "inplane"
drop_slice_prob = 0.05
shift_slice_prob = 0.05
shift_magnitude = 10
intensity = true
mul_intensity = 0.1
add_intensity = 0.1
noise_scale = 0.5
"""

TRAINER = """[trainer]
max_steps = %(max_steps)d
batch_size = 1            # per rank; 8 ranks -> global batch 8
lr = %(lr)s
warmup_steps = %(warmup)d
min_lr_ratio = %(min_lr_ratio)s
weight_decay = 0.05
grad_clip_norm = 1.0
layerwise_lr_decay = 1.0  # the stem must move: 0.9 x 0.2 once put patch_embed 696x below BANIS' LR
patch_embed_lr_mult = 1.0
lr_schedule = "linear"
activation_checkpointing = false
precision = "bf16"
log_every = 100
val_every = %(val_every)d
checkpoint_every = %(ckpt_every)d
num_workers = 6
seed = 0

[parallelism]
dp_replicate = 1
dp_shard = 8              # one full node per stage
tp = 1
"""

ARMS = [
    dict(
        n=1, tag="dinov3_simmim", model=DINOV3, rope="vanilla", scale="singlescale",
        ssl="simmim",
        blurb="DINOv3 ViT-L from random init, pretrained with SimMIM, then finetuned.",
        why_rope="No 2D checkpoint to stay compatible with, so the true axial 3D RoPE is "
                 "available and is what a 3D model should use.",
    ),
    dict(
        n=2, tag="dinov3_lvd", model=DINOV3, rope="superposition", scale="singlescale",
        ssl=None,
        blurb="DINOv3 ViT-L from the released LVD-1689M checkpoint, no SSL. The baseline.",
        why_rope="Forced by the initialisation: the released weights are 2D, and superposition is "
                 "the variant that inflates. NOTE its depth term sits behind a zero-initialised "
                 "scalar, so layerwise_lr_decay and patch_embed_lr_mult must stay at 1.0 or "
                 "z-position never learns and the model is effectively 2D.",
    ),
    dict(
        n=3, tag="muvit_mae", model=MUVIT, rope=None, scale="multiscale",
        ssl="muvit_mae",
        blurb="MuViT (multi-scale) from random init, pretrained with MAE, then finetuned.",
        why_rope="Not a config choice: MuViT's rotary embedding is intrinsic and driven by each "
                 "token's world coordinate, which is the mechanism that lets it relate levels.",
    ),
]

SSL_ALGO = {
    "simmim": """[algorithm]
name = "simmim"
mask_ratio = 0.6
mask_granularity = 2      # hide 2x2x2 patch blocks, i.e. 32 voxels across
norm_pix_loss = false
""",
    "muvit_mae": """[algorithm]
name = "muvit_mae"
mask_ratio = 0.75
dirichlet_alpha = 0.5     # how unevenly the mask budget is split across levels
decoder_embed_dim = 512
decoder_depth = 4
decoder_num_heads = 8
norm_pix_loss = true
""",
}

AFFINITY = """[algorithm]
name = "affinity_seg"
long_range = 10
decoder = "%(decoder)s"
%(decoder_opts)s split_disconnected = true
"""

INTERP_OPTS = "decoder_hidden_dim = 64\n"
SUBPIXEL_OPTS = ("decoder_hidden_dim = 256\n"
                 "decoder_readout_dim = 16\n"
                 "decoder_refine_depth = 2\n")


def data_block(path: str, samples: int = 1000) -> str:
    return f'[data]\nname = "miao_volumes"\nconfig_path = "{path}"\nsamples_per_epoch = {samples}\n'


def val_block(path: str) -> str:
    return f'[val_data]\nname = "miao_volumes"\nconfig_path = "{path}"\nsamples_per_epoch = 32\n'


def write(name: str, text: str) -> None:
    (HERE / name).write_text(text)
    print(f"wrote {name}")


def main() -> None:
    for arm in ARMS:
        n, tag, scale = arm["n"], arm["tag"], arm["scale"]
        model = arm["model"] % {"rope": arm["rope"], "drop_path": "0.1"} if arm["model"] is DINOV3 \
            else arm["model"]

        # ---- stage a: SSL pretraining (arms 1 and 3 only) ----------------------------------
        if arm["ssl"]:
            pre = (f"{LMD}/lmd_pretraining_{scale}_min1Gvox_noeval_patch256_sizeexp0.4.yaml")
            write(f"{n}a_{tag}_pretrain.toml",
                  f"""# lmd_ssl_v1 arm {n} stage A: {arm['ssl']} pretraining. {arm['blurb']}
#
# Pretrains on the 91-volume corpus with the 8 evaluation volumes HELD OUT, so this arm never sees
# them unlabelled. Arm 2 has no pretraining at all, so leaving them in would flatter arms 1 and 3
# against the baseline for a reason that has nothing to do with the objective.
#
# RoPE: {arm['why_rope']}
experiment_name = "lmd1__{n}a_{tag}_pretrain"

{model}
{SSL_ALGO[arm['ssl']]}
{data_block(pre)}
{val_block(pre)}
{TRAINER % dict(max_steps=SSL_STEPS, lr=LR_SSL, warmup=WARMUP, min_lr_ratio=MIN_LR_RATIO,
                val_every=5000, ckpt_every=10000)}
{AUGMENT}""")

        # ---- stages b and c: the shared two-stage finetune ---------------------------------
        for letter, decoder, opts, lr, prev in (
            ("b", "interpolate", INTERP_OPTS, LR_INTERP, "ssl"),
            ("c", "subpixel", SUBPIXEL_OPTS, LR_SUBPIXEL, "interp"),
        ):
            if prev == "ssl":
                if arm["ssl"]:
                    init = (f'# This arm\'s own SSL encoder, from stage A.\npath = "PREV_CHECKPOINT"\n'
                            f'prefix = "model."\nstrict = true\n')
                    init_note = "the encoder this arm pretrained in stage A"
                else:
                    init = (f'path = "{DINOV3_LVD}"\ninflate_2d_to_3d = true\n'
                            f'skip = ["rope_embed."]    # a 2D checkpoint says nothing about z\n'
                            f'strict = true\n')
                    init_note = "the released 2D DINOv3 checkpoint, inflated along z"
            else:
                init = ('# This arm\'s own interpolating-head encoder, from stage B.\n'
                        'path = "PREV_CHECKPOINT"\nprefix = "model."\nstrict = true\n')
                init_note = "this arm's stage-B encoder; the sub-pixel head starts fresh"

            stage_doc = (
                "Stage B trains the interpolating head. Stage C swaps in the sub-pixel head, warm-\n"
                "# started from B's encoder: the sub-pixel head's output layer is zero-initialised, so\n"
                "# at step 0 no gradient reaches the encoder at all and the early signal is ~1e-5\n"
                "# against the interpolating decoder's ~8.6. From a cold encoder that starves; from a\n"
                "# trained one it is fine, which is why the two stages exist."
                if letter == "c" else
                "Stage B of the shared two-stage finetune: interpolating head first."
            )
            lr_doc = (
                f"# Peak LR {lr:g} rather than stage B's {LR_INTERP:g}. Stage C continues an encoder that has\n"
                f"# already converged under a decaying schedule, and restarting at the full peak would undo\n"
                f"# that. A lower peak is the simple way to get the same effect without a stateful,\n"
                f"# multi-stage learning-rate scheduler."
                if letter == "c" else
                f"# Peak LR {lr:g}, linear to min_lr_ratio, {WARMUP} warmup steps -- identical across all\n"
                f"# three arms, because the finetune protocol is the thing being held constant."
            )

            write(f"{n}{letter}_{tag}_finetune_{decoder}.toml",
                  f"""# lmd_ssl_v1 arm {n} stage {letter.upper()}: finetune, {decoder} head.
#
# {stage_doc}
#
# Initialised from {init_note}.
#
{lr_doc}
#
# Trains on 4 of the 8 instance-segmentation eval volumes and validates on the other 4, one from
# each modality pair (see make_splits.py). Every volume keeps the bounding box measured from its
# label's non-zero extent.
experiment_name = "lmd1__{n}{letter}_{tag}_ft_{decoder}"

{model}
[init]
{init}
{AFFINITY % dict(decoder=decoder, decoder_opts=opts)}
{data_block(f"{REL}/lmd_finetune_{scale}.yaml")}
{val_block(f"{REL}/lmd_val_{scale}.yaml")}
{TRAINER % dict(max_steps=FT_STEPS, lr=lr, warmup=WARMUP, min_lr_ratio=MIN_LR_RATIO,
                val_every=2500, ckpt_every=5000)}
{AUGMENT}""")


if __name__ == "__main__":
    main()
