from __future__ import annotations

from dataclasses import dataclass

# The rotation subgroups `[augment].rotate` selects. Imported rather than restated so the
# config cannot offer a value the augmentation does not implement.
from data.augment import ROTATIONS

PRECISIONS = ("fp32", "bf16")

# How the learning rate falls from its post-warmup peak to `min_lr_ratio`. Both reach the same
# endpoints at the same steps and differ only in the path between them: cosine spends longer near
# the peak and decays fastest in the middle, linear spends more of the run at a lower rate.
LR_SCHEDULES = ("linear", "cosine")


@dataclass(frozen=True)
class InitConfig:
    """Where a run's *model* weights start, when not from scratch.

    Separate from resuming, which `[trainer]` and the checkpoint manager handle: resuming
    continues one run and restores the optimizer and step too, while this begins a new run from
    weights trained elsewhere -- an earlier pretraining run, or a released checkpoint -- and takes
    nothing but the model.
    """

    path: str = ""
    prefix: str = ""
    inflate_2d_to_3d: bool = False
    skip: tuple[str, ...] = ()
    strict: bool = True
    allow_unused: bool = False
    # What the checkpoint is loaded *into*.
    #
    # `"model"` (the default, and what every run before this used) loads the bare encoder, before
    # the algorithm wraps it, so a strategy's own parameters -- a masked-autoencoding decoder, an
    # affinity head -- keep the initialisation they were built with. That is the right default:
    # the usual reason to set `[init]` is to start from a pretrained *encoder* and learn a new
    # head, and silently inheriting a head trained for a different objective would be worse than
    # useless.
    #
    # `"algorithm"` loads into the algorithm instead, so those parameters come from the checkpoint
    # too. Use it to genuinely continue a previous run's model -- not merely its encoder -- in a
    # new run with a fresh optimizer and step counter, which `--resume` cannot do because it
    # restores both. A pseudo-labelling round that warm-starts from the model that produced its
    # labels is the motivating case: dropping that model's head would throw away trained weights
    # for no reason and hand the student a random head to rediscover.
    #
    # The two need different `prefix` values, because they sit at different depths: an algorithm
    # checkpoint stores the encoder under `model.` *within* the algorithm, so `target = "model"`
    # wants `prefix = "model."` to reach past it while `target = "algorithm"` wants `prefix = ""`.
    # An algorithm that registers one module under two names (`affinity_seg` exposes its encoder
    # as both `model` and `encoder`) also stores it twice, so the duplicate needs `skip`.
    target: str = "model"
    # Fold any LoRA adapter in the checkpoint into the base weight it adapts, and drop it, before
    # matching against this model. What it is for: a stage that adapted an encoder through LoRA
    # produces a checkpoint carrying `.lora_a`/`.lora_b`/`.lora_scaling` beside every adapted
    # `weight`. Merging turns that into a plain checkpoint of the model it is *equivalent to* --
    # which is what lets the next stage either (a) load it into a model that knows nothing about
    # LoRA, or (b) load it into a freshly adapted model, where the previous stage's adaptation has
    # become part of the frozen prior and the new adapter starts from zero again. (b) is the
    # difference between chaining adapters and re-basing them, and it is one key rather than two
    # configs because the checkpoint states its own scaling.
    merge_lora: bool = False

    def __post_init__(self) -> None:
        if self.target not in ("model", "algorithm"):
            raise ValueError(
                f"[init].target must be 'model' or 'algorithm', got {self.target!r}"
            )
        if not self.path and (
            self.prefix
            or self.inflate_2d_to_3d
            or self.skip
            or self.allow_unused
            or self.merge_lora
            or self.target != "model"
        ):
            raise ValueError(
                "[init] sets loading options but no 'path', so nothing would be loaded and the "
                "run would silently start from scratch"
            )
        # TOML gives arrays as lists; freeze so the config stays hashable and immutable.
        object.__setattr__(self, "skip", tuple(self.skip))


@dataclass(frozen=True)
class LoRAConfig:
    """Adapt a pretrained encoder through low-rank deltas instead of updating it outright.

    Off by default (`rank = 0`), like every other mechanism here that changes what a run trains.

    What this is for: a training stage whose job is to *adapt* a pretrained backbone rather than
    replace it. The base weights are frozen and each targeted projection gains a rank-`r` delta
    initialised to exactly zero, so the run starts computing precisely what the pretrained model
    did and can only move within an `r`-dimensional subspace per layer. It costs almost nothing in
    memory (the saving is optimizer state and weight gradients, ~450 MB per rank on a ViT-L at
    dp_shard 8) and *nothing* in compute: backward still traverses the whole stack to reach the
    adapters. Reach for it for the inductive bias, not for the budget.

    Which parameters stay fully trainable is the other half of the setting. The three `train_*`
    switches below cover the parts of a 2D-pretrained encoder that genuinely start wrong or cost
    nothing to open; a model may additionally *require* a parameter to keep training, through
    `BaseModel.lora_required_trainable`, and no config can override that.
    """

    # 0 disables. `scaling = alpha / rank` multiplies the delta, so the two numbers are not
    # independent -- which is why `alpha` has no default. A default would mean that raising `rank`
    # to buy capacity silently *halved* the scale of the delta, and nothing in the run would say so.
    rank: int = 0
    alpha: float = 0.0
    # Group names from `BaseModel.lora_target_groups()`; a name the model does not offer is an
    # error naming the menu. Attention only, which is where LoRA was shown to pay best; add "mlp"
    # to roughly triple the adapter's parameter count.
    targets: tuple[str, ...] = ("attn_qkv", "attn_proj")

    # The input stem -- `engine.optimizer.is_stem`, the same partition the layerwise learning rate
    # and the frozen warm-up use. An inflated RGB kernel now reading single-channel EM is the one
    # layer with no pretrained answer at all, and a rank-`r` correction to a wrong stem is not the
    # tool for it. 4.2M params on a ViT-L/16 at patch 16.
    train_stem: bool = True
    # LayerNorm gains and biases, plus LayerScale gammas: 0.15M params on a ViT-L, and the
    # parameters that absorb a change in input statistics most directly. Opening them is what
    # essentially every PEFT recipe does.
    train_norms: bool = True
    # CLS, storage and mask tokens -- 6.1K params. The mask token is what a masked-image objective
    # substitutes for what it hid, so an SSL stage that froze it would be adapting to a fixed
    # stand-in it cannot shape.
    train_tokens: bool = True

    def enabled(self) -> bool:
        return self.rank > 0

    def __post_init__(self) -> None:
        # TOML gives arrays as lists; freeze so the config stays hashable and immutable.
        object.__setattr__(self, "targets", tuple(self.targets))
        if self.rank < 0:
            raise ValueError(f"[lora].rank must be >= 0, got {self.rank}; 0 disables")
        if not self.enabled():
            if self.alpha or not self.targets:
                raise ValueError(
                    "[lora] sets options but leaves rank at 0, so no adapter would be built and "
                    "the run would silently train the whole encoder. Set rank, or drop the section."
                )
            return
        if self.alpha <= 0.0:
            raise ValueError(
                f"[lora].alpha must be > 0 when rank is set, got {self.alpha}. The delta is scaled "
                f"by alpha/rank, so state it explicitly: with rank = {self.rank}, alpha = "
                f"{2 * self.rank} gives the commonly used scaling of 2."
            )
        if not self.targets:
            raise ValueError(
                "[lora].targets is empty, so the adapter would attach to nothing while the encoder "
                "sat frozen. Name at least one group from the model's lora_target_groups()."
            )
        duplicates = sorted({name for name in self.targets if self.targets.count(name) > 1})
        if duplicates:
            raise ValueError(f"[lora].targets repeats {duplicates}")


@dataclass(frozen=True)
class AugmentConfig:
    """Training-data augmentation. Empty by default, so a run augments only when it says so.

    Applied to the training dataset alone -- the engine never wraps `[val_data]`. Validation has
    to measure the model on the data as it is, and an augmented validation set silently changes
    what every number in the run means, which is not something a config key should be able to do
    by accident.

    Defaults are off rather than the reference recipe's values: turning augmentation on is a
    decision about the experiment, and a run that inherited it from a default would be hard to
    tell apart from one that asked for it.
    """

    # Axis-aligned rotations and flips. "none", "inplane" or "full" (see `data.augment.ROTATIONS`).
    #
    # This was previously the dataset's business -- miao's per-volume `aug_rot` key -- and moved
    # here when the operations did, so that one section describes a run's whole augmentation
    # rather than half of it. Two consequences worth knowing: it is now one setting for every
    # volume in a run rather than per volume, and a data config still carrying `aug_rot` will be
    # rejected by miao rather than silently ignored.
    #
    # "full" draws from all 48 signed permutations of the axes and needs cubic voxels. "inplane"
    # excludes the sectioning axis, leaving 16, which is what anisotropic serial-section EM needs:
    # at 9x9x20 nm an exchange of z with x would relabel a 20 nm neighbour relationship as a 9 nm
    # one and produce object shapes that do not occur in the data. miao asserts the condition each
    # requires against the sample's own `pixel_size`, so a mismatch fails rather than trains.
    rotate: str = "none"

    drop_slice_prob: float = 0.0
    shift_slice_prob: float = 0.0
    shift_magnitude: int = 10
    intensity: bool = False
    mul_intensity: float = 0.1
    add_intensity: float = 0.1
    noise_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.rotate not in ROTATIONS:
            raise ValueError(f"[augment].rotate must be one of {ROTATIONS}, got {self.rotate!r}")

    def enabled(self) -> bool:
        return bool(
            self.rotate != "none"
            or self.drop_slice_prob > 0.0
            or (self.shift_slice_prob > 0.0 and self.shift_magnitude > 0)
            or self.intensity
            or self.noise_scale > 0.0
        )


@dataclass(frozen=True)
class TrainerConfig:
    """Hyperparameters for one training run, normally populated from a .toml config."""

    max_steps: int
    batch_size: int
    lr: float = 1e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    grad_clip_norm: float | None = 1.0
    # Shape of the post-warmup decay; see LR_SCHEDULES. Governs the *learning rate* only --
    # `final_weight_decay` interpolates on its own curve, since the two schedules answer different
    # questions and coupling them would silently change one when the other is set.
    #
    # Runs recorded before this setting existed used cosine, which was then the only option; their
    # `resolved_config.json` therefore has no `lr_schedule` key at all rather than saying "cosine".
    lr_schedule: str = "linear"

    # Per-parameter learning-rate shaping. The two lr knobs default to "off"; the weight-decay
    # exemption below does not -- see its own note.
    #
    # `layerwise_lr_decay` scales the learning rate by `decay ** (depth_from_the_top)`, so early
    # blocks move less than late ones. That is what makes fine-tuning a pretrained backbone
    # stable: the general features near the input are worth preserving, while the layers nearest
    # the objective have the most to unlearn. 1.0 disables it.
    layerwise_lr_decay: float = 1.0
    # The patch embedding is the one layer that has to re-learn a new input statistic (a different
    # modality, or an inflated 2D kernel), yet it also sits deepest in the layerwise schedule.
    # This scales it separately; upstream DINOv3 uses 0.2.
    patch_embed_lr_mult: float = 1.0
    # Weight decay is scheduled like the learning rate, from `weight_decay` to this. None holds it
    # constant. DINOv3 *raises* it over training (0.04 -> 0.4), tightening the representation as
    # the teacher stops moving.
    final_weight_decay: float | None = None
    # Weight decay shrinks a weight toward zero, which regularizes a matrix whose norm controls
    # how much signal it passes. A normalization gain and a bias have no such reading: decaying a
    # LayerNorm gain just pulls the layer's output toward zero and works against the normalization
    # the layer exists to provide. Exempting them is what essentially every transformer recipe
    # does, DINOv3's included, which is why it is the default here rather than an opt-in.
    zero_weight_decay_on_norm_and_bias: bool = True
    # The same exemption for a LoRA adapter's two factors, and it is on by default for a reason
    # worth stating. `wd_scale` decides by *rank*, and `lora_a`/`lora_b` are both 2-D, so without
    # this they receive the full `weight_decay`. Decaying `lora_b` -- which starts at zero --
    # shrinks the *delta*, not a weight matrix, which is an implicit pull of the adapted model back
    # toward the pretrained weights it started from. That may well be desirable, but it is a
    # regularizer toward θ₀ rather than toward the origin, its strength would be set by a key named
    # for something else, and a run would carry it with nothing to show it. Set this false to opt
    # into that anchor deliberately.
    zero_weight_decay_on_lora: bool = True

    # Recompute the model's and algorithm's declared regions during backward instead of storing
    # their activations: roughly 30% more compute for most of the activation memory back. What
    # gets recomputed is each architecture's own answer (`checkpointable_modules`); this only
    # says whether to honour it. Off by default, since it is a cost with no benefit until memory
    # is actually the binding constraint -- which for volumetric crops it becomes abruptly, as
    # activation memory grows with the cube of the crop size.
    activation_checkpointing: bool = False

    # Peak dense throughput of one GPU, in TFLOP/s, as the denominator for MFU. None looks the
    # device up in `utils.hardware_flops`; set it explicitly for a GPU that table does not know,
    # or to override it. Use the dense figure, not the 2:4-sparsity one vendors headline with --
    # nothing here prunes weights, and the sparse number would halve every reported MFU.
    peak_tflops: float | None = None
    # Measure a step's FLOPs once at startup and report `mfu`, `tflops_per_s` and `samples_per_s`.
    # The probe costs one extra forward/backward per run, whose gradients are discarded.
    measure_mfu: bool = True

    # Write a torch.profiler trace of a few steps to `<run>/profile/`. Off by default: this is a
    # diagnostic to reach for when `mfu` says a run is slow, not something every run should carry.
    # See `engine.profiler` for what the trace does and does not record.
    profile: bool = False
    # Which steps to trace, counted from this process's first step rather than from the absolute
    # training step, so a resumed job still profiles. The default skips the window where cuDNN is
    # still choosing algorithms and the allocator is still growing, neither of which recurs.
    profile_start_step: int = 50
    profile_steps: int = 6
    # Every rank, rather than rank 0 alone. Costs a trace per rank; buys the one thing a single
    # rank cannot show, which is whether the ranks are balanced -- see `profiler.should_profile`.
    profile_all_ranks: bool = False
    # Record allocator activity alongside the timeline. Useful for deciding whether activation
    # checkpointing would pay, and separable from the timing question, so it is its own switch.
    profile_memory: bool = False

    # Hold the encoder fixed for this many steps while the patch embedding and the algorithm's own
    # head train against it, then unfreeze and train jointly. 0 disables it. The point is to spare
    # a pretrained encoder the gradients of a randomly initialised head: adapting a 2D checkpoint
    # to volumes, the patch embedding and the head are the two parts that start wrong, and the
    # backbone is the part worth protecting until they are not.
    #
    # Only the *model's* parameters are frozen, and only those outside its input stem -- the
    # algorithm's head sits outside the model and keeps training, which is the whole point.
    # Note this does not make the phase cheap: the stem is below the blocks, so backward still
    # traverses the whole stack to reach it. What it saves is the weight gradients and the
    # optimizer state, not the pass.
    freeze_backbone_steps: int = 0
    # Linear ramp on the learning rate when the backbone joins, so it does not take a full-size
    # step on its first one. Applies to every group, not just the newly unfrozen ones: the head has
    # been training for `freeze_backbone_steps` already, and briefly slowing it too is a smaller
    # distortion than running two learning-rate schedules at once.
    unfreeze_warmup_steps: int = 1000

    precision: str = "fp32"
    log_every: int = 10
    checkpoint_every: int = 0
    val_every: int = 0
    num_workers: int = 0
    # Keep the dataloader's worker processes alive across epochs instead of forking a new set each
    # time the loader is exhausted. Off by default, which is the opposite of what the obvious
    # reasoning suggests, so the measurement is worth recording.
    #
    # Respawning does cost something: each worker re-imports torch and re-opens every zarr store,
    # measured at ~1.9 s per epoch boundary, or ~4 ms amortized over the 125 steps an epoch lasts
    # here. Keeping them alive is worse, by far. A volumetric sample is large -- 134 MB, and the
    # affinity task's components pass allocates a 256^3 int32 array per call on top -- and in a
    # process that never restarts, that never gets fully reclaimed. Over 400 steps on 8 ranks:
    #
    #   persistent = false   epoch 0: 241 ms/step   after: 266 ms/step    3 of 28 windows stalled
    #   persistent = true    epoch 0: 232 ms/step   after: 334 ms/step   20 of 28 windows stalled
    #
    # A 100k-step run is ~800 epochs, so essentially all of it lives in the second column: 4 ms
    # saved against 68 ms lost. The stall is invisible from rank 0 -- it shows up on whichever
    # rank drew the slow sample, which is what `data_wait_frac_max` exists to surface.
    persistent_workers: bool = False
    # How many batches each worker runs ahead. Left at torch's default, and raising it is a trap
    # worth naming: at prefetch 6 the same runs measured 213 ms/step through epoch 0 -- the best
    # of any setting, since a deeper queue really does absorb a slow crop -- and then 445 ms/step
    # afterwards, the worst of any setting, because three times as many 134 MB samples in flight
    # reach the memory pressure above three times faster. Depth is not the constraint here.
    prefetch_factor: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < max_steps ({self.max_steps})"
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError(f"grad_clip_norm must be > 0 or None, got {self.grad_clip_norm}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}"
            )
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if self.peak_tflops is not None and self.peak_tflops <= 0.0:
            raise ValueError(f"peak_tflops must be > 0 or None, got {self.peak_tflops}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.prefetch_factor < 1:
            raise ValueError(
                f"prefetch_factor must be >= 1, got {self.prefetch_factor}; it is ignored when "
                "num_workers = 0, since there is no worker to run ahead"
            )
        if self.profile_start_step < 0:
            raise ValueError(
                f"profile_start_step must be >= 0, got {self.profile_start_step}"
            )
        if self.profile_steps < 1:
            raise ValueError(
                f"profile_steps must be >= 1, got {self.profile_steps}; set profile = false to "
                "disable profiling instead"
            )
        if self.freeze_backbone_steps < 0:
            raise ValueError(
                f"freeze_backbone_steps must be >= 0, got {self.freeze_backbone_steps}; 0 disables"
            )
        if self.freeze_backbone_steps >= self.max_steps:
            raise ValueError(
                f"freeze_backbone_steps ({self.freeze_backbone_steps}) must be < max_steps "
                f"({self.max_steps}), or the backbone would never train"
            )
        if self.unfreeze_warmup_steps < 0:
            raise ValueError(
                f"unfreeze_warmup_steps must be >= 0, got {self.unfreeze_warmup_steps}"
            )
        if not 0.0 < self.layerwise_lr_decay <= 1.0:
            raise ValueError(
                f"layerwise_lr_decay must be in (0, 1], got {self.layerwise_lr_decay}; it is a "
                "per-layer multiplier, and a value above 1 would give the earliest layers the "
                "largest learning rate"
            )
        if self.patch_embed_lr_mult <= 0.0:
            raise ValueError(
                f"patch_embed_lr_mult must be > 0, got {self.patch_embed_lr_mult}; use a small "
                "value to slow the patch embedding, not zero to freeze it"
            )
        if self.final_weight_decay is not None and self.final_weight_decay < 0.0:
            raise ValueError(
                f"final_weight_decay must be >= 0 or None, got {self.final_weight_decay}"
            )
