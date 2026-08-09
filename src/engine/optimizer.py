from __future__ import annotations

import math
import re
from typing import Any

import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .config import LR_SCHEDULES, TrainerConfig

# Every model here names its transformer stack `blocks`, so a parameter's depth can be read off
# its name. `tests/unit/test_engine_optimizer.py` pins that convention across the registry rather
# than trusting it, so a future model that names its stack something else fails loudly instead of
# silently getting a flat learning rate.
_BLOCK_INDEX = re.compile(r"(?:^|\.)blocks\.(\d+)\.")

# The stem: everything the first block reads. Named rather than inferred because the models here
# disagree -- the ViTs call it `patch_embed` and hold learned tokens beside it, while `muvit3d`
# projects each scale level through `patch_proj` and adds a `level_embed`.
# `tests/unit/test_engine_optimizer.py` builds every registered model and checks its stem is
# recognised, so a model whose stem is named something else fails loudly instead of silently
# receiving the *largest* learning rate in the network.
_STEM_MARKERS = ("patch_embed", "pos_embed", "patch_proj", "level_embed")

# Algorithms register their model first (`BaseAlgorithm.__init__`), so a backbone parameter is
# always reached through this prefix once an algorithm wraps it. Depth only means something inside
# the backbone: a masked-autoencoding decoder's own mask token is not the encoder's stem, however
# similarly it is named.
_BACKBONE_PREFIX = "model."


def _block_count(model: nn.Module) -> int:
    """How many transformer blocks deep the model is, read from its parameter names."""
    depths = [
        int(match.group(1))
        for name, _ in model.named_parameters()
        if (match := _BLOCK_INDEX.search(name))
    ]
    return max(depths) + 1 if depths else 0


def parameter_depth(name: str, n_blocks: int, backbone_scoped: bool = False) -> int:
    """Where a parameter sits in the stack: 0 at the stem, `n_blocks + 1` past the last block.

    The stem is everything the first block reads, and takes the deepest discount. Anything that is
    neither a block nor the stem -- a final norm, a projection head, an algorithm's own decoder --
    sits above the stack and is not discounted at all.

    `backbone_scoped` says the caller is looking at a whole algorithm rather than a bare model, in
    which case only parameters under the backbone prefix can have a depth. Without it, a
    strategy's freshly-initialised decoder token would be mistaken for the encoder's stem and
    trained at the slowest rate in the network.
    """
    if backbone_scoped and not name.startswith(_BACKBONE_PREFIX):
        return n_blocks + 1
    match = _BLOCK_INDEX.search(name)
    if match is not None:
        return int(match.group(1)) + 1
    is_stem = any(marker in name for marker in _STEM_MARKERS)
    if is_stem or name.endswith("_token") or "tokens" in name:
        return 0
    return n_blocks + 1


def lr_scale(
    name: str, n_blocks: int, config: TrainerConfig, backbone_scoped: bool = False
) -> float:
    """The fixed per-parameter learning-rate multiplier, before any schedule."""
    if config.layerwise_lr_decay >= 1.0 and config.patch_embed_lr_mult == 1.0:
        return 1.0
    depth = parameter_depth(name, n_blocks, backbone_scoped)
    scale = config.layerwise_lr_decay ** (n_blocks + 1 - depth)
    if any(marker in name for marker in ("patch_embed", "patch_proj")):
        scale *= config.patch_embed_lr_mult
    return scale


def wd_scale(name: str, parameter: nn.Parameter, config: TrainerConfig) -> float:
    """The fixed per-parameter weight-decay multiplier: 1, or 0 for things that should not decay.

    Decided by *rank*, not by name. A parameter with fewer than two dimensions has no matrix norm
    for decay to regularize -- it is a bias, a normalization gain, a LayerScale gamma, or a set of
    rotary frequencies -- and shrinking it just drags the layer toward the identity. Rank is also
    the only rule that survives the naming here: `muvit3d` holds a LayerNorm inside an
    `nn.Sequential`, so its gain is called `patch_proj.0.1.weight` with no "norm" anywhere in it,
    and `ViT3D`'s learned rotary frequencies are `blocks.0.rotary.inv_freqs.0`.
    """
    if not config.zero_weight_decay_on_norm_and_bias:
        return 1.0
    return 0.0 if parameter.ndim < 2 else 1.0


def build_param_groups(model: nn.Module, config: TrainerConfig) -> list[dict[str, Any]]:
    """Bucket parameters by their (lr, wd) multipliers -> optimizer param groups.

    Parameters sharing both multipliers go in one group, so a 40-block model produces a few dozen
    groups rather than one per tensor. `lr` is baked in per group, which is what makes `LambdaLR`
    apply the schedule on top of the layerwise scaling for free -- it multiplies each group's own
    `initial_lr`.
    """
    n_blocks = _block_count(model)
    # An algorithm reaches its encoder through `model.`; a bare encoder has no such prefix.
    backbone_scoped = any(
        name.startswith(_BACKBONE_PREFIX) for name, _ in model.named_parameters()
    )
    buckets: dict[tuple[float, float], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = (
            lr_scale(name, n_blocks, config, backbone_scoped),
            wd_scale(name, parameter, config),
        )
        group = buckets.setdefault(
            key,
            {
                "params": [],
                "lr": config.lr * key[0],
                "weight_decay": config.weight_decay * key[1],
                "lr_multiplier": key[0],
                "wd_multiplier": key[1],
            },
        )
        group["params"].append(parameter)
    if not buckets:
        raise ValueError("model has no parameters that require grad, so there is nothing to train")
    # Descending, so index 0 is the *undiscounted* group. Order has to be deterministic for a
    # checkpoint's optimizer state to reload, and putting the base learning rate first means
    # anything logging `param_groups[0]["lr"]` reports the number the config states rather than
    # whichever group happens to be most discounted.
    return [buckets[key] for key in sorted(buckets, reverse=True)]


def build_optimizer(model: nn.Module, config: TrainerConfig) -> Optimizer:
    return AdamW(
        build_param_groups(model, config),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


def decay_fraction(progress: float, schedule: str) -> float:
    """How much of the peak learning rate survives, 1.0 at `progress` 0 and 0.0 at 1.0.

    Only the path between those endpoints differs. Cosine leaves the rate near its peak for the
    first part of the run and does most of its decay in the middle; linear starts falling
    immediately, so a run spends more of its length at a lower rate. Which is better is a property
    of the run, not of the repo -- hence a setting rather than a fixed choice.
    """
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "linear":
        return 1.0 - progress
    raise ValueError(f"unknown lr_schedule {schedule!r}; expected one of {list(LR_SCHEDULES)}")


def lr_multiplier(step: int, config: TrainerConfig) -> float:
    """Linear warmup then `config.lr_schedule` decay to `min_lr_ratio`, as a multiple of base LR.

    `step` is 0-based, matching LambdaLR's `last_epoch`. Returns 1.0 at the end of warmup and
    exactly `min_lr_ratio` at `max_steps`, whichever decay shape is in use -- the shapes differ
    only in between.
    """
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
    remaining = decay_fraction(progress, config.lr_schedule)
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * remaining


def weight_decay_at(step: int, config: TrainerConfig) -> float:
    """Cosine from `weight_decay` to `final_weight_decay` over the run, in absolute terms.

    No warmup, unlike the learning rate: there is nothing to stabilise, and the endpoints are what
    the schedule is stated in.

    Deliberately *not* governed by `lr_schedule`. This interpolates between two stated endpoints
    because DINOv3 raises weight decay as the teacher settles; the learning-rate shape answers an
    unrelated question, and tying them would mean changing one setting silently moved the other.
    """
    if config.final_weight_decay is None:
        return config.weight_decay
    progress = min(1.0, max(0.0, step / max(1, config.max_steps)))
    span = config.final_weight_decay - config.weight_decay
    return config.weight_decay + span * (1.0 - math.cos(math.pi * progress)) / 2


class WarmupDecaySchedule(LambdaLR):
    """Drives the learning rate -- warmup then `config.lr_schedule` decay -- and the weight decay.

    `LambdaLR` alone only moves the learning rate. Weight decay is not a scalar it knows about, so
    it is written here -- on the same `step()` the trainer already calls, rather than as a second
    object the loop would have to remember to advance.
    """

    def __init__(self, optimizer: Optimizer, config: TrainerConfig) -> None:
        self._config = config
        super().__init__(optimizer, lr_lambda=lambda step: lr_multiplier(step, config))
        self._apply_weight_decay()

    def _apply_weight_decay(self) -> None:
        decay = weight_decay_at(self.last_epoch, self._config)
        for group in self.optimizer.param_groups:
            group["weight_decay"] = decay * group.get("wd_multiplier", 1.0)

    def step(self, epoch: int | None = None) -> None:
        super().step(epoch)
        self._apply_weight_decay()


def build_lr_scheduler(optimizer: Optimizer, config: TrainerConfig) -> LambdaLR:
    return WarmupDecaySchedule(optimizer, config)
