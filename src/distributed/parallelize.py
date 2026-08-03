from __future__ import annotations

import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module

from models.base import BaseModel

from .parallel_dims import ParallelDims


def parallelize_model(model: nn.Module, mesh: DeviceMesh, dims: ParallelDims) -> nn.Module:
    """Apply tensor parallelism, then sharding/replication, to `model` in place.

    Sharding and replication work on any `nn.Module` — which is all `BaseAlgorithm` promises to
    hold — while tensor parallelism additionally needs the `BaseModel.tensor_parallel_plan()`
    contract, so a module without one is rejected here rather than deeper inside torch.
    """
    if dims.tp_enabled:
        plan = model.tensor_parallel_plan() if isinstance(model, BaseModel) else None
        if plan is None:
            raise ValueError(
                f"{type(model).__name__} does not define a tensor_parallel_plan(), "
                f"but tp={dims.tp} was requested"
            )
        parallelize_module(model, mesh["tp"], plan)

    if dims.hsdp_enabled:
        fully_shard(model, mesh=mesh["dp_replicate", "dp_shard"])
    elif dims.dp_shard > 1:
        fully_shard(model, mesh=mesh["dp_shard"])
    elif dims.dp_replicate > 1:
        replicate(model, device_mesh=mesh["dp_replicate"])

    return model
