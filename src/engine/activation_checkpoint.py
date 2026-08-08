"""Trade compute for activation memory by recomputing regions of the forward pass in backward.

Backpropagation needs every intermediate a forward pass produced, so activation memory scales
with the *input*, not the parameter count. For a transformer that is linear in sequence length,
and for a dense head writing at voxel resolution it is linear in voxels — which is why a large 3D
crop is expensive in a way a large model is not. A 512-cube crop at patch 16 is 32768 tokens,
about 64x the activations of a 128-cube crop through the same 300M-parameter encoder, and it is
the activations that decide whether a run fits on a GPU.

Checkpointing keeps only a region's inputs and runs it a second time during backward to
regenerate the rest: roughly 30% more compute for most of the memory back.

Which regions are worth it is architecture knowledge, not engine knowledge — a transformer wants
its blocks, a dense decoder wants the half of itself that runs at full resolution — so models and
algorithms declare their own through `checkpointable_modules()` and the engine only decides
whether to honour it. That keeps `[trainer].activation_checkpointing` a single switch that means
the same thing everywhere, instead of a flag per architecture.

Applied *before* sharding: FSDP2 hooks a module's forward to all-gather its parameters, and
wrapping after that would put the recomputation outside the gather.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)

from algorithms.base import BaseAlgorithm
from models.base import BaseModel

logger = logging.getLogger(__name__)


def checkpointable_targets(root: nn.Module) -> list[nn.Module]:
    """Every submodule the tree under `root` declares worth recomputing.

    Collected by walking rather than asking the root, so an algorithm does not have to remember
    to forward its model's answer along with its own — `parallelize_algorithm` treats the two as
    one tree for sharding and this matches.
    """
    targets: list[nn.Module] = []
    seen: set[int] = set()
    for module in root.modules():
        if not isinstance(module, BaseModel | BaseAlgorithm):
            continue
        for target in module.checkpointable_modules():
            if id(target) not in seen:
                seen.add(id(target))
                targets.append(target)
    return targets


def apply_activation_checkpointing(root: nn.Module) -> int:
    """Wrap every declared region of `root` in place. Returns how many were wrapped.

    Nothing declared is an error rather than a silent no-op: the switch is turned on to make a run
    fit in memory, and a run that quietly ignored it would OOM with the setting apparently
    enabled, which is a long way to travel to find a missing method.
    """
    targets = checkpointable_targets(root)
    if not targets:
        raise ValueError(
            "activation_checkpointing is on, but nothing under "
            f"{type(root).__name__} declares checkpointable_modules(), so it would have no "
            "effect. Implement it on the model or algorithm (returning e.g. its transformer "
            "blocks), or turn the setting off."
        )

    wrapped = {id(target) for target in targets}
    _wrap_children(root, wrapped)
    logger.info(
        "activation checkpointing: recomputing %d module(s) in backward (%s)",
        len(targets),
        ", ".join(sorted({type(target).__name__ for target in targets})),
    )
    return len(targets)


def _wrap_children(module: nn.Module, wrapped: set[int]) -> None:
    """Replace matching children with checkpointed versions, depth first.

    Written out rather than using torch's own `apply_activation_checkpointing`, whose `check_fn`
    is consulted for every module in the tree: identity matching against a precomputed set says
    exactly which instances were asked for, where a predicate on type or name would also catch
    a block nested inside one already being recomputed and pay for the same region twice.
    """
    for name, child in list(module.named_children()):
        if id(child) in wrapped:
            setattr(module, name, checkpoint_wrapper(child, CheckpointImpl.NO_REENTRANT))
        else:
            _wrap_children(child, wrapped)
