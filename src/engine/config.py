from __future__ import annotations

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        if not self.path and (
            self.prefix or self.inflate_2d_to_3d or self.skip or self.allow_unused
        ):
            raise ValueError(
                "[init] sets loading options but no 'path', so nothing would be loaded and the "
                "run would silently start from scratch"
            )
        # TOML gives arrays as lists; freeze so the config stays hashable and immutable.
        object.__setattr__(self, "skip", tuple(self.skip))


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

    drop_slice_prob: float = 0.0
    shift_slice_prob: float = 0.0
    shift_magnitude: int = 10
    intensity: bool = False
    mul_intensity: float = 0.1
    add_intensity: float = 0.1
    noise_scale: float = 0.0

    def enabled(self) -> bool:
        return bool(
            self.drop_slice_prob > 0.0
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

    # Recompute the model's and algorithm's declared regions during backward instead of storing
    # their activations: roughly 30% more compute for most of the activation memory back. What
    # gets recomputed is each architecture's own answer (`checkpointable_modules`); this only
    # says whether to honour it. Off by default, since it is a cost with no benefit until memory
    # is actually the binding constraint -- which for volumetric crops it becomes abruptly, as
    # activation memory grows with the cube of the crop size.
    activation_checkpointing: bool = False

    precision: str = "fp32"
    log_every: int = 10
    checkpoint_every: int = 0
    val_every: int = 0
    num_workers: int = 0
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
