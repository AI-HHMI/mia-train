#!/usr/bin/env python3
"""Emit the six run configs of the data-scaling experiment from one template.

The claim this experiment rests on is that its three arms differ in *nothing* but how many
labelled cubes they train on. Hand-maintaining six near-identical TOMLs is how that claim quietly
stops being true, so the files are generated: every knob below is written once, and an arm's
identity is the two substitutions in ARMS. Regenerate with

    python experiments/data_scaling/make_configs.py

and `git diff` to see what a change actually did.
"""
import pathlib

HERE = pathlib.Path(__file__).parent
REL = "experiments/data_scaling"

# The 5 base cubes, taken in seed order so the arms are nested subsets (see the data YAMLs).
ARMS = [(1, "one"), (3, "three"), (5, "five")]

MODEL = """[model]
name = "dinov3_vit3d"
img_size = 256            # 16^3 = 4096 tokens at patch 16
patch_size = 16           # fixed by the checkpoint: the in-plane kernel transfers as is
in_chans = 1
embed_dim = 1024          # ViT-L/16, 303M parameters
depth = 24
num_heads = 16
n_storage_tokens = 4      # the released ViTs carry 4 registers
layerscale_init = 1.0e-05 # NOT optional -- the released weights were trained with LayerScale
mask_k_bias = true        # ditto: the checkpoint carries a masked key bias per block
pos_embed_rope_dtype = "fp32"
pos_embed_rope_type = "superposition"
drop_path_rate = 0.1      # as in pseudo_labeling; held equal across arms, so not a confound
"""

# Everything the two stages share. `max_steps`/`checkpoint_every` are the only trainer knobs that
# differ between them, and they are formatted in.
TRAINER = """[trainer]
max_steps = {max_steps}
batch_size = 1            # per rank; 8 ranks -> global batch 8
lr = 3.0e-4
warmup_steps = 2000       # 500 collapsed both pseudo-labelling arms onto the trivial predictor
min_lr_ratio = 0.001
weight_decay = 0.05
grad_clip_norm = 1.0
layerwise_lr_decay = 1.0  # the stem must move: 0.9 x 0.2 put patch_embed 696x below BANIS' LR
patch_embed_lr_mult = 1.0
lr_schedule = "linear"
activation_checkpointing = false
precision = "bf16"
log_every = 100
val_every = 5000
checkpoint_every = {checkpoint_every}
num_workers = 6           # as in the 8-rank runs this recipe was measured on
seed = 0

[parallelism]
dp_replicate = 1
dp_shard = 8              # one full node per arm
tp = 1

[augment]
# BANIS' own defaults, as used by the pseudo-labelling arms. "inplane" draws from the 16
# transforms that never exchange z with x or y; the full 48-transform group is refused against
# 9x9x20 nm voxels, since exchanging axes of unequal size produces shapes that do not occur in
# the specimen. This matters more here than usual -- with one cube, augmentation is most of what
# stands between the run and memorising 8.2e9 crop centres of a single volume -- so it is held
# identical across the arms rather than tuned per arm.
rotate = "inplane"
drop_slice_prob = 0.05
shift_slice_prob = 0.05
shift_magnitude = 10
intensity = true
mul_intensity = 0.1
add_intensity = 0.1
noise_scale = 0.5
"""

VAL = """[val_data]
# The same held-out cube every run in this line of work has been scored on. Untouched by the arm:
# only the *training* set changes, so val is directly comparable across arms and back to
# banis_parity / subpixel_decoder / pseudo_labeling.
name = "miao_volumes"
config_path = "configs/data/nisb_base.yaml"
patch_size = [256, 256, 256]
samples_per_epoch = 32
resolutions = [[9, 9, 20]]
output_axes = "lcxyz"

[[val_data.volumes]]
name = "NISB base val seed100"
path = "/groups/miaai/miaai/lmd-v0.0.1/dev/nisb/base/val/seed100.zarr"
image_key = "raw"
label_key = "labels/public_gt-cell-nisb"
zarr_version = "zarr3"
weight = 1.0
normalize = true
"""

HEAD_A = """# Data scaling on NISB base, arm "{n} cube{s}", stage A: interpolating decoder, 200k steps.
#
# Does NISB `base` still carry information at its fifth cube, or has it saturated below that? The
# benchmark's own baselines show 100 cubes doing no better than 5, which says the *upper* end is
# flat; this asks where the flat part starts. Pure supervised finetuning from the released DINOv3
# weights -- no SSL, no pseudo-labels -- so the only thing separating the arms is {n} cube{s} of
# labels against 3 and 5.
#
# Stage A is the recipe that produced `banis_parity__finetune_256_long`, at 200k rather than 300k
# because 200k is the checkpoint that lineage actually went on to use. It exists to give stage B a
# trained encoder: a sub-pixel head has never once trained from a cold encoder here -- three
# attempts all collapsed onto "everything is connected" (init_comparison/README.md) -- so the two
# stages are a requirement of the architecture, not a schedule choice.
experiment_name = "ds__{n}cube_interp"

{model}
[init]
# The released 2D DINOv3 ViT-L/16, inflated along z. This is the "pretrained checkpoint" baseline
# the whole comparison is against.
path = "/groups/miaai/miaai/pretrained_models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
inflate_2d_to_3d = true
skip = ["rope_embed."]    # a 2D checkpoint says nothing about a third axis
strict = true

[algorithm]
name = "affinity_seg"
long_range = 10
decoder = "interpolate"
decoder_hidden_dim = 64
split_disconnected = true

[data]
name = "miao_volumes"
config_path = "{rel}/nisb_base_{n}cube_256.yaml"
samples_per_epoch = 1000

{val}
{trainer}"""

HEAD_B = """# Data scaling on NISB base, arm "{n} cube{s}", stage B: sub-pixel decoder, 100k steps.
#
# Warm-starts from *this arm's own* stage A, so the arm is a closed chain and the {n}-cube number
# is never contaminated by an encoder that saw more cubes. `prefix = "model."` keeps the encoder
# and drops stage A's interpolating head: `BaseAlgorithm` registers the backbone as `model` (and
# `self.encoder` is the same object, so torch stores it once under the name it saw first), while
# the checkpoint's `decoder.*` keys do not start with the prefix and are filtered out before the
# unused-key check -- which is what leaves the new sub-pixel head at its initialisation.
#
# No `skip`: unlike stage A this predecessor is already 3D, so the RoPE buffers match.
experiment_name = "ds__{n}cube_subpixel"

{model}
[init]
# Resolved inside the job by submit.sh -- at submission time stage A's run directory does not
# exist yet.
path = "PREV_CHECKPOINT"
prefix = "model."
strict = true

[algorithm]
name = "affinity_seg"
long_range = 10
decoder = "subpixel"
# Width on the patch grid, where positions are 4096x cheaper than voxels -- so this can be wide.
decoder_hidden_dim = 256
# Width at voxel resolution. This is the number that costs: every tensor after the expansion is
# this many channels over the whole crop.
decoder_readout_dim = 16
# Two 3x3x3 convolutions -> a 5-voxel receptive field across the seam between two tokens' blocks,
# which are otherwise decoded independently.
decoder_refine_depth = 2
split_disconnected = true

[data]
name = "miao_volumes"
config_path = "{rel}/nisb_base_{n}cube_256.yaml"
samples_per_epoch = 1000

{val}
{trainer}"""

for n, _word in ARMS:
    s = "" if n == 1 else "s"
    for tag, head, steps, ckpt in (
        ("a_interp", HEAD_A, 200_000, 20_000),   # 10 ckpts x 4.8 GB
        ("b_subpixel", HEAD_B, 100_000, 10_000),  # 10 ckpts x 4.8 GB
    ):
        text = head.format(
            n=n, s=s, rel=REL, model=MODEL, val=VAL,
            trainer=TRAINER.format(max_steps=steps, checkpoint_every=ckpt),
        )
        (HERE / f"{n}{tag}.toml").write_text(text)
        print(f"wrote {n}{tag}.toml")
