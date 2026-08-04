from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel

from distributed.parallel_dims import ParallelDims
from distributed.parallelize import parallelize_model
from models.base import BaseModel


class _ToyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(8, 16)
        self.w2 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x))

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0

    def tensor_parallel_plan(self) -> dict[str, ParallelStyle]:
        return {"w1": ColwiseParallel(), "w2": RowwiseParallel()}


class _MethodEntryModel(BaseModel):
    """Used through `encode` rather than `forward`, the way MAE drives a ViT3D."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0

    def extra_forward_methods(self) -> tuple[str, ...]:
        return ("encode",)


class _NoPlanModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0


def _has_grad_after_forward_backward(model: nn.Module, param_name: str) -> bool:
    x = torch.randn(2, 8)
    loss = model(x).sum()
    loss.backward()
    return dict(model.named_parameters())[param_name].grad is not None


def _ddp_worker(rank: int, world_size: int) -> bool:
    dims = ParallelDims(dp_replicate=world_size)
    mesh = dims.build_mesh("cpu")
    model = _ToyModel()
    parallelize_model(model, mesh, dims)
    return _has_grad_after_forward_backward(model, "w1.weight")


def _fsdp_worker(rank: int, world_size: int) -> bool:
    dims = ParallelDims(dp_shard=world_size)
    mesh = dims.build_mesh("cpu")
    model = _ToyModel()
    parallelize_model(model, mesh, dims)
    ok = _has_grad_after_forward_backward(model, "w1.weight")
    return ok and isinstance(model.w1.weight, DTensor)


def _hsdp_worker(rank: int, world_size: int) -> bool:
    dims = ParallelDims(dp_replicate=2, dp_shard=2)
    mesh = dims.build_mesh("cpu")
    model = _ToyModel()
    parallelize_model(model, mesh, dims)
    return _has_grad_after_forward_backward(model, "w1.weight")


def _tp_plus_fsdp_worker(rank: int, world_size: int) -> bool:
    dims = ParallelDims(dp_shard=2, tp=2)
    mesh = dims.build_mesh("cpu")
    model = _ToyModel()
    parallelize_model(model, mesh, dims)
    return _has_grad_after_forward_backward(model, "w1.weight")


def _declared_forward_method_worker(rank: int, world_size: int) -> bool:
    dims = ParallelDims(dp_shard=world_size)
    mesh = dims.build_mesh("cpu")
    model = _MethodEntryModel()
    parallelize_model(model, mesh, dims)
    assert isinstance(model.proj.weight, DTensor)  # sharded before anything is called
    model.encode(torch.randn(2, 8)).sum().backward()  # never goes through forward()
    return model.proj.weight.grad is not None


def _replicate_rejects_method_entry_worker(rank: int, world_size: int) -> bool:
    dims = ParallelDims(dp_replicate=world_size)
    mesh = dims.build_mesh("cpu")
    try:
        parallelize_model(_MethodEntryModel(), mesh, dims)
    except ValueError as error:
        return "dp_replicate" in str(error)
    return False


def _tp_without_plan_raises_worker(rank: int, world_size: int) -> bool:
    dims = ParallelDims(tp=world_size)
    mesh = dims.build_mesh("cpu")
    model = _NoPlanModel()
    try:
        parallelize_model(model, mesh, dims)
    except ValueError:
        return True
    return False


@pytest.mark.cpu_dist
def test_ddp_replicates_and_produces_gradients(run_distributed):
    assert all(run_distributed(_ddp_worker, world_size=2))


@pytest.mark.cpu_dist
def test_fsdp_shards_parameters(run_distributed):
    assert all(run_distributed(_fsdp_worker, world_size=2))


@pytest.mark.cpu_dist
def test_hsdp_shards_and_replicates(run_distributed):
    assert all(run_distributed(_hsdp_worker, world_size=4))


@pytest.mark.cpu_dist
def test_tp_plus_fsdp_composes(run_distributed):
    assert all(run_distributed(_tp_plus_fsdp_worker, world_size=4))


@pytest.mark.cpu_dist
def test_fsdp_wraps_the_forward_methods_a_model_declares(run_distributed):
    # FSDP2 all-gathers around `forward` only. A model an algorithm calls into by another name
    # would otherwise still hold sharded DTensors, and the op would reject the plain input.
    assert all(run_distributed(_declared_forward_method_worker, world_size=2))


@pytest.mark.cpu_dist
def test_replicate_rejects_a_model_used_outside_forward(run_distributed):
    # `replicate` syncs gradients from forward hooks, so it cannot cover such calls; failing
    # loudly beats silently averaging nothing.
    assert all(run_distributed(_replicate_rejects_method_entry_worker, world_size=2))


@pytest.mark.cpu_dist
def test_tp_without_plan_raises(run_distributed):
    assert all(run_distributed(_tp_without_plan_raises_worker, world_size=2))
