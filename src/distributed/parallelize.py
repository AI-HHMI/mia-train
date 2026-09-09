from __future__ import annotations

import torch
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


# Fraction of device memory that one FSDP unit's gathered parameters may occupy when
# `fsdp_unit_budget_gb` is "auto". Deliberately small: the gathered weights sit alongside their
# gradients and the activations, and this only has to be loose enough that a model whose whole
# parameter set is a rounding error on the device stays a single unit.
_AUTO_UNIT_BUDGET_FRACTION = 0.05
_BYTES_PER_GIB = 1024**3


def _shard(module: nn.Module, mesh: DeviceMesh, dims: ParallelDims) -> bool:
    """FSDP-shard `module` over the data-parallel mesh. False if no sharding was requested.

    The module's declared `fsdp_units` are packed into groups, each its own FSDP unit inside the
    unit the module itself forms. FSDP2 all-gathers a unit's parameters for the whole of that
    unit's forward and reduce-scatters its gradients in one collective, so the grouping is a
    direct trade: fewer, larger units issue fewer collectives and hold more weights resident.

    Both ends of that trade are real, which is why this is a budget and not a boolean. One unit
    per block on ViT-L (303M parameters, 24 blocks) issues ~72 collectives a step against ~3, and
    measured 114.3 +/- 2.1 samples/s over four runs of the production SimMIM arm against
    123.8 +/- 0.3 without it -- 7.7% -- to save ~1.2 GB of resident weights on a 141 GB device.
    The same per-block split at 7B saves ~28 GB and is what makes the run fit. Packing to a byte
    budget gets both: a small model collapses to a single group, a large one keeps the split it
    needs.

    The cost only shows up when the host is contended, which is the other half of why it went
    unnoticed: the collectives are cheap on device but each carries host-side launch work, and
    `defer_image_ops=false` leaves the dataloader workers saturating the cores. The `defer=true`
    arm leaves them ~90% idle and measured no difference at all (166.7 against 166.8).
    """
    if dims.hsdp_enabled:
        dp_mesh = mesh["dp_replicate", "dp_shard"]
    elif dims.dp_shard > 1:
        dp_mesh = mesh["dp_shard"]
    else:
        return False

    # Bottom-up, as FSDP2 requires: a group claims the parameters not already claimed by a group
    # made from a submodule, so the root call last picks up whatever the groups left.
    for group in _fsdp_unit_groups(module, dims, dp_mesh.device_type):
        # The list form makes the whole group ONE unit -- one all-gather, one reduce-scatter --
        # without a container module to hold it, so the module tree and therefore every
        # state_dict key is unchanged. A per-module loop here would instead make len(group)
        # units and defeat the point.
        fully_shard(list(group), mesh=dp_mesh)
    fully_shard(module, mesh=dp_mesh)
    return True


def _fsdp_units(module: nn.Module) -> tuple[nn.Module, ...]:
    """What `module` itself declares, without walking into it.

    Deliberately not the tree walk `engine.activation_checkpoint` does: `parallelize_algorithm`
    shards the model and the algorithm as separate roots, and a walk from the algorithm would
    reach the model's blocks a second time after they were already made units.
    """
    return module.fsdp_units() if isinstance(module, BaseModel) else ()


def unit_budget_bytes(dims: ParallelDims, device_type: str = "cpu") -> int:
    """`fsdp_unit_budget_gb` in bytes, resolving "auto" against a mesh of `device_type`.

    Public so a caller can report the budget it will be sharded under.

    `device_type` comes from the mesh rather than from `torch.cuda.is_available()`, because
    `torch.cuda.get_device_properties` *initialises a CUDA context* and this is reached from
    `_shard`, which on the Gloo path runs inside each forked rank. A CPU mesh has no business
    creating a CUDA context per rank -- on a node whose GPUs are in exclusive-process mode that
    is a device contended for no reason -- so a non-CUDA mesh does not touch CUDA at all, not
    even to ask whether it exists. (`tests/distributed` does fail with
    `CUDA-capable device(s) is/are busy or unavailable` in a one-GPU job, but master fails it
    identically, so that is the job's GPU count and not this.)

    For a non-CUDA mesh there is no device memory to take a fraction of, so this falls back to a
    figure larger than any model such a test builds: the fallback degenerates to the
    single-root-unit case rather than splitting a test model at some arbitrary point, and it is
    deliberately close to what a real device yields (5% of an H200's 141 GB is ~7 GiB) so CPU and
    GPU runs do not group differently for small models.
    """
    budget = dims.fsdp_unit_budget_gb
    if budget != "auto":
        return int(float(budget) * _BYTES_PER_GIB)
    if device_type == "cuda" and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        return int(total * _AUTO_UNIT_BUDGET_FRACTION)
    return 8 * _BYTES_PER_GIB


def _fsdp_unit_groups(
    module: nn.Module, dims: ParallelDims, device_type: str = "cpu"
) -> tuple[tuple[nn.Module, ...], ...]:
    """The declared units packed into consecutive groups, each under the byte budget.

    Consecutive rather than best-fit: FSDP2 prefetches the next unit while the current one
    computes, and that only overlaps if a unit's members run together. Packing blocks 0-5 into
    one group preserves that; packing 0, 7 and 19 into one would gather weights long before two
    of them are needed and stall on the third.

    A unit larger than the budget on its own becomes its own group, because there is nothing
    smaller to split it into -- the budget bounds what this can choose, not what the model
    declared.
    """
    units = _fsdp_units(module)
    if not units:
        return ()
    budget = unit_budget_bytes(dims, device_type)

    groups: list[tuple[nn.Module, ...]] = []
    current: list[nn.Module] = []
    current_bytes = 0
    for unit in units:
        unit_bytes = sum(p.numel() * p.element_size() for p in unit.parameters())
        if current and current_bytes + unit_bytes > budget:
            groups.append(tuple(current))
            current, current_bytes = [], 0
        current.append(unit)
        current_bytes += unit_bytes
    if current:
        groups.append(tuple(current))

    # One group holding every declared unit is the same set of parameters the root call would
    # have claimed anyway, so making it a nested unit buys a redundant collective. Drop it and
    # let the root be the only unit -- which is exactly the pre-budget behaviour for a model
    # small enough not to need splitting.
    if len(groups) == 1:
        return ()
    return tuple(groups)


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
