from __future__ import annotations

import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._composable.replicate import replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import register_fsdp_forward_method
from torch.distributed.tensor.parallel import parallelize_module

from algorithms.base import BaseAlgorithm
from models.base import BaseModel

from .parallel_dims import ParallelDims


def apply_tensor_parallel(model: nn.Module, mesh: DeviceMesh, dims: ParallelDims) -> None:
    """Shard individual layers across the tensor-parallel mesh, per the model's own plan.

    Public, and separate from the sharding below, because of an ordering constraint that is
    otherwise invisible: **tensor parallelism must be applied before activation checkpointing.**
    `checkpoint_wrapper` re-parents a block under `_checkpoint_wrapped_module`, and
    `parallelize_module` resolves a plan's paths through `named_children()`, so a plan written
    against `blocks.0.attn.qkv` matches nothing once the block is wrapped -- and matching nothing
    is not an error there. The run would train, with every projection replicated, at tp times the
    memory the setting was chosen to avoid. `engine.trainer` therefore calls this first, then
    checkpoints, then shards.
    """
    if not dims.tp_enabled:
        return
    if any(isinstance(module, CheckpointWrapper) for module in model.modules()):
        raise ValueError(
            f"{type(model).__name__} is already wrapped for activation checkpointing, so a "
            "tensor-parallel plan's paths no longer resolve. Apply tensor parallelism first: "
            "engine.trainer does, and this is checked here because the alternative outcome is a "
            "run that trains fully replicated without saying so."
        )
    plan = model.tensor_parallel_plan() if isinstance(model, BaseModel) else None
    if plan is None:
        raise ValueError(
            f"{type(model).__name__} does not define a tensor_parallel_plan(), "
            f"but tp={dims.tp} was requested"
        )
    parallelize_module(model, mesh["tp"], plan)


def _shard(module: nn.Module, mesh: DeviceMesh, dims: ParallelDims) -> bool:
    """FSDP-shard `module` over the data-parallel mesh. False if no sharding was requested.

    The module's declared `fsdp_units` become units of their own inside it. FSDP2 all-gathers a
    unit's parameters for the whole of that unit's forward, so a single root unit leaves a sharded
    model materializing every parameter at once -- the optimizer state is sharded and nothing else
    is. Nested units are gathered and resharded block by block instead.
    """
    if dims.hsdp_enabled:
        dp_mesh = mesh["dp_replicate", "dp_shard"]
    elif dims.dp_shard > 1:
        dp_mesh = mesh["dp_shard"]
    else:
        return False

    for unit in _fsdp_units(module):
        fully_shard(unit, mesh=dp_mesh)
    fully_shard(module, mesh=dp_mesh)
    return True


def _fsdp_units(module: nn.Module) -> tuple[nn.Module, ...]:
    """What `module` itself declares, without walking into it.

    Deliberately not the tree walk `engine.activation_checkpoint` does: `parallelize_algorithm`
    shards the model and the algorithm as separate roots, and a walk from the algorithm would
    reach the model's blocks a second time after they were already made units.
    """
    return module.fsdp_units() if isinstance(module, BaseModel) else ()


def _entry_points(model: nn.Module) -> tuple[str, ...]:
    return model.extra_forward_methods() if isinstance(model, BaseModel) else ()


def parallelize_model(model: nn.Module, mesh: DeviceMesh, dims: ParallelDims) -> nn.Module:
    """Apply tensor parallelism, then sharding or replication, to a model in place.

    For a model driven only through its own `forward`. A training strategy that owns parameters
    of its own should go through `parallelize_algorithm` instead, which covers both.
    """
    apply_tensor_parallel(model, mesh, dims)
    return shard_model(model, mesh, dims)


def shard_model(model: nn.Module, mesh: DeviceMesh, dims: ParallelDims) -> nn.Module:
    """The sharding half of `parallelize_model`, for a caller that applied tensor parallelism."""
    entry_points = _entry_points(model)
    if _shard(model, mesh, dims):
        # FSDP2 all-gathers parameters around `forward` only, so a model driven through other
        # methods needs those wrapped too or they see sharded DTensors.
        for name in entry_points:
            register_fsdp_forward_method(model, name)
    elif dims.dp_replicate > 1:
        if entry_points:
            raise ValueError(
                f"{type(model).__name__} is also used through {entry_points} rather than only "
                "forward(), and `replicate` all-reduces gradients from forward hooks, so a "
                f"dp_replicate={dims.dp_replicate} run would silently skip the sync for those "
                "calls. Replicate the owning algorithm instead (parallelize_algorithm), whose "
                "forward encloses the whole step."
            )
        replicate(model, device_mesh=mesh["dp_replicate"])

    return model


def parallelize_algorithm(
    algorithm: BaseAlgorithm, mesh: DeviceMesh, dims: ParallelDims
) -> BaseAlgorithm:
    """Parallelize a whole training strategy: its model, plus any parameters it owns itself.

    An algorithm may hold parameters beside the model — MAE's decoder exists only for the
    pretraining objective — and torch's grad-norm clipping and optimizers refuse to mix sharded
    DTensors with plain tensors, so a sharded run has to cover both or it fails at the first
    `clip_grad_norm_`.

    Both halves at once, for a caller with nothing to do between them. `engine.trainer` has:
    activation checkpointing goes after `apply_tensor_parallel` and before `shard_algorithm`, so
    it calls the two directly.
    """
    apply_tensor_parallel(algorithm.model, mesh, dims)
    return shard_algorithm(algorithm, mesh, dims)


def shard_algorithm(
    algorithm: BaseAlgorithm, mesh: DeviceMesh, dims: ParallelDims
) -> BaseAlgorithm:
    """The sharding half of `parallelize_algorithm`.

    Under FSDP the model is sharded first as its own unit, which keeps it resharded between
    forward passes, and the algorithm becomes the outer unit holding whatever is left. Training
    reaches the outer unit through `BaseAlgorithm.forward`, which FSDP hooks natively;
    `validation_step` is registered because the engine calls it directly.

    Under plain replication the *algorithm* is wrapped and the model is not: DDP's all-reduce
    fires from `forward` hooks, and only the algorithm's forward encloses the entire step —
    including a model the strategy drives through methods other than its forward. Wrapping the
    model there would miss those parameters, and would miss them silently.
    """
    if _shard(algorithm.model, mesh, dims):
        for name in _entry_points(algorithm.model):
            register_fsdp_forward_method(algorithm.model, name)

    if _shard(algorithm, mesh, dims):
        register_fsdp_forward_method(algorithm, "validation_step")
    elif dims.dp_replicate > 1:
        replicate(algorithm, device_mesh=mesh["dp_replicate"])

    return algorithm
