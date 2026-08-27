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

# Global batch for the SSL stages, and the single most consequential number in this experiment.
#
# The first run used 8 (one node, batch 1/rank), inherited from NISB *supervised* finetuning where
# it works -- arm 2 trains normally on it. Masked reconstruction does not: measured at fixed
# geometry, SimMIM at batch 84 crosses the trivial crop-mean floor at ~step 900 and keeps widening
# the margin to +0.0137 by step 4000 (encoder token-std 0.037 -> 0.156), while at batch 8-16 it
# never beats a single scalar. MuViT-MAE on the same corpus collapses outright below batch 32.
#
# 16 per rank x 8 ranks (ONE H200 node) = 128. Memory measured on a real H200 node with FSDP at
# arm 1's geometry: per-rank 16 peaks at 80.0 GiB of 139.8 (57%), per-rank 24 still fits, so this
# needs neither activation checkpointing nor gradient accumulation. Arm 3 is the more expensive
# shape -- measured on an H100, per-rank 16 does NOT fit in 80 GiB without checkpointing -- but it
# has ~1.7x the headroom on an H200 at the same batch.
#
# One node, not the two that global batch 256 needed: a single-node job takes an ordinary queue
# instead of a `*_parallel` one, needs no c10d rendezvous across the IB fabric, and schedules far
# sooner. Global 128 is still 16x the batch-8 regime that stalled, and well past the batch-16 ->
# batch-32 cliff measured for MuViT-MAE.
SSL_BATCH_PER_RANK = 16
SSL_DP_SHARD = 8
# The finetune stages stay at 8, unchanged, so arms 1-3 share one protocol and remain comparable
# to arm 2 -- which trained fine there, supervised.
FT_BATCH_PER_RANK = 1
FT_DP_SHARD = 8

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
# On by default here, unlike the model's own `use_fa4 = False`. Note the key differs from MuViT's
# `attention_backend = "flash4"` -- the two model families have separate switches with no shared
# base class, which is how arm 1 ran its first attempts on SDPA while arm 3 used FA4 and nothing
# reported the asymmetry. Unlike `attention_backend = "auto"`, this raises if FA4 is unusable
# rather than falling back in silence.
use_fa4 = true
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
batch_size = %(batch_size)d            # per rank; x %(dp_shard)d ranks -> global batch %(global_batch)d
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
num_workers = %(workers)d
seed = 0

[parallelism]
dp_replicate = 1
dp_shard = %(dp_shard)d              # %(nodes)s
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


# Measured on a full node at arm 1's geometry, 8 ranks through mia-train's real build_dataloader:
# `samples_per_epoch` is the dominant loader knob, and 1000 was pathological. At global batch 128 it
# makes an epoch SEVEN STEPS long, and every iterator restart drains the prefetch queue -- 82.1
# samples/s against 123.2 for the same 6 workers at 100k. Raising workers on top of that is worth a
# further ~6% (131.1 at 11), and nothing beyond 11 helps: 16/22/32 measured 113.2/129.5/127.3, flat
# inside a ~8% noise floor even at 264 processes on 96 cores, because the workers block on NFS
# rather than on CPU. `persistent_workers` and `prefetch_factor` both measured as no-ops.
SSL_SAMPLES_PER_EPOCH = 100_000
SSL_WORKERS = 11
# The finetune stages keep 1000/6: at global batch 8 an epoch is 125 steps, nowhere near the
# pathology, and arm 2 has already run under exactly these values.
FT_WORKERS = 6


def data_block(path: str, samples: int = 1000) -> str:
    return f'[data]\nname = "miao_volumes"\nconfig_path = "{path}"\nsamples_per_epoch = {samples}\n'


def val_block(path: str, global_batch: int) -> str:
    """Validation set, sized so the loader cannot come back empty.

    `Trainer` builds the val loader with the SAME per-rank `batch_size` as training and
    `drop_last=True`, and `DistributedSampler` hands each rank `samples_per_epoch / ranks`. If that
    quotient is below the per-rank batch, every rank drops its only partial batch, `validate()`
    averages over zero batches and returns `{}` -- so the run logs a bare `[val] step N` with no
    metrics and TensorBoard shows no `val/*` series at all. It is silent: training is unaffected and
    nothing errors, the validation curve simply never appears.

    Raising the SSL arms to global batch 256 walked straight into this: 32 samples over 16 ranks is
    2 per rank against a per-rank batch of 16. Sizing to at least the global batch guarantees one
    full batch per rank. The finetune stages stay at 32 (global batch 8 there), which keeps them
    byte-identical to the configs arm 2 has already run under.
    """
    samples = max(32, global_batch)
    return (f'[val_data]\nname = "miao_volumes"\nconfig_path = "{path}"\n'
            f'samples_per_epoch = {samples}\n')


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
{data_block(pre, SSL_SAMPLES_PER_EPOCH)}
{val_block(pre, SSL_BATCH_PER_RANK * SSL_DP_SHARD)}
{TRAINER % dict(max_steps=SSL_STEPS, lr=LR_SSL, warmup=WARMUP, min_lr_ratio=MIN_LR_RATIO, workers=SSL_WORKERS,
                val_every=5000, ckpt_every=10000,
                batch_size=SSL_BATCH_PER_RANK, dp_shard=SSL_DP_SHARD,
                global_batch=SSL_BATCH_PER_RANK * SSL_DP_SHARD,
                nodes="one full H200 node")}
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
{val_block(f"{REL}/lmd_val_{scale}.yaml", FT_BATCH_PER_RANK * FT_DP_SHARD)}
{TRAINER % dict(max_steps=FT_STEPS, lr=lr, warmup=WARMUP, min_lr_ratio=MIN_LR_RATIO, workers=FT_WORKERS,
                val_every=2500, ckpt_every=5000,
                batch_size=FT_BATCH_PER_RANK, dp_shard=FT_DP_SHARD,
                global_batch=FT_BATCH_PER_RANK * FT_DP_SHARD,
                nodes="one full node per stage")}
{AUGMENT}""")


if __name__ == "__main__":
    main()
