"""How `fsdp_unit_budget_gb` turns a model's declared FSDP units into groups.

Unit-level and CPU-only on purpose: `_fsdp_unit_groups` needs a module and a `ParallelDims`, not a
mesh or a device, and the packing is the part with a decision in it. The collective count it
produces is what the 7.7% throughput difference on the production SimMIM arm came down to, so it
is worth pinning at this level rather than only through an 8-rank run.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from distributed.parallel_dims import ParallelDims
from distributed.parallelize import _fsdp_unit_groups, unit_budget_bytes
from models.base import BaseModel

_MIB = 1024**2


class _BlockModel(BaseModel):
    """`n_blocks` blocks of a known parameter size, declared as this model's FSDP units."""

    def __init__(self, n_blocks: int, floats_per_block: int, declare: bool = True) -> None:
        super().__init__()
        # One float32 parameter vector per block, so a block's byte size is exactly
        # 4 * floats_per_block and the arithmetic under test is checkable by hand.
        self.blocks = nn.ModuleList(
            nn.Linear(floats_per_block, 1, bias=False) for _ in range(n_blocks)
        )
        self._declare = declare

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0

    def fsdp_units(self) -> tuple[nn.Module, ...]:
        return tuple(self.blocks) if self._declare else ()


def _block_bytes(model: _BlockModel) -> int:
    return sum(p.numel() * p.element_size() for p in model.blocks[0].parameters())


@pytest.mark.unit
def test_auto_is_the_default_and_a_positive_budget_is_accepted():
    assert ParallelDims().fsdp_unit_budget_gb == "auto"
    assert ParallelDims(fsdp_unit_budget_gb=4.0).fsdp_unit_budget_gb == 4.0


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, 0.0, "nope", "", True, False])
def test_a_budget_that_is_not_auto_or_positive_is_rejected(bad: Any):
    # `True` is in here because bool subclasses int, so `True > 0` would otherwise be read as a
    # 1 GiB budget rather than a configuration mistake.
    with pytest.raises(ValueError, match="fsdp_unit_budget_gb"):
        ParallelDims(fsdp_unit_budget_gb=bad)


@pytest.mark.unit
def test_a_model_declaring_no_units_gets_no_groups():
    model = _BlockModel(n_blocks=4, floats_per_block=8, declare=False)
    assert _fsdp_unit_groups(model, ParallelDims(dp_shard=2)) == ()


@pytest.mark.unit
def test_units_that_all_fit_the_budget_collapse_to_no_nested_group():
    """One group holding every unit claims the same parameters the root would, so it is dropped.

    This is the case that matters for ViT-L: 24 blocks of 1.2 GB total against a budget of several
    GB is one group, and one group is a redundant collective in front of the root's.
    """
    model = _BlockModel(n_blocks=24, floats_per_block=1024)
    dims = ParallelDims(dp_shard=8, fsdp_unit_budget_gb=1.0)
    assert _fsdp_unit_groups(model, dims) == ()


@pytest.mark.unit
def test_units_are_packed_into_consecutive_groups_under_the_budget():
    model = _BlockModel(n_blocks=12, floats_per_block=256 * 1024)  # 1 MiB per block
    per_block = _block_bytes(model)
    assert per_block == _MIB
    # 4 MiB budget over 1 MiB blocks: groups of 4, so 3 groups.
    dims = ParallelDims(dp_shard=8, fsdp_unit_budget_gb=4 / 1024)
    groups = _fsdp_unit_groups(model, dims)

    assert [len(g) for g in groups] == [4, 4, 4]
    for group in groups:
        assert sum(p.numel() * p.element_size() for m in group for p in m.parameters()) <= 4 * _MIB


@pytest.mark.unit
def test_grouping_preserves_declared_order_and_partitions_every_unit():
    """Consecutive, exhaustive and non-overlapping -- FSDP2 prefetch depends on the first, and a
    dropped or duplicated unit would silently change which parameters get sharded."""
    model = _BlockModel(n_blocks=10, floats_per_block=256 * 1024)
    dims = ParallelDims(dp_shard=8, fsdp_unit_budget_gb=3 / 1024)
    groups = _fsdp_unit_groups(model, dims)

    flattened = [m for group in groups for m in group]
    assert flattened == list(model.blocks)
    assert len({id(m) for m in flattened}) == len(flattened)


@pytest.mark.unit
def test_a_unit_larger_than_the_budget_becomes_its_own_group():
    """The budget bounds what the packing may choose, not what the model declared: there is
    nothing smaller than one declared unit to split into."""
    model = _BlockModel(n_blocks=3, floats_per_block=256 * 1024)  # 1 MiB per block
    dims = ParallelDims(dp_shard=8, fsdp_unit_budget_gb=0.25 / 1024)  # 256 KiB, under one block
    groups = _fsdp_unit_groups(model, dims)

    assert [len(g) for g in groups] == [1, 1, 1]


@pytest.mark.unit
def test_an_explicit_budget_is_taken_literally_in_gib():
    assert unit_budget_bytes(ParallelDims(fsdp_unit_budget_gb=2.0)) == 2 * 1024**3
    assert unit_budget_bytes(ParallelDims(fsdp_unit_budget_gb=0.5)) == 512 * _MIB


@pytest.mark.unit
def test_auto_resolves_to_a_positive_budget_with_or_without_a_device():
    # The CPU fallback exists so the grouping is testable without a GPU; either way the budget has
    # to be a usable positive number rather than 0, which would put every unit in its own group.
    assert unit_budget_bytes(ParallelDims()) > 0
    assert unit_budget_bytes(ParallelDims(), "cpu") > 0


@pytest.mark.unit
def test_a_cpu_mesh_never_initialises_cuda(monkeypatch: pytest.MonkeyPatch):
    """Regression: an "auto" budget must not probe CUDA when the mesh is a CPU mesh.

    `_shard` runs inside each forked rank on the Gloo path, and `get_device_properties`
    initialises a CUDA context, so probing it there has every CPU-only rank claim a device it
    will never use -- pointless on any node and actively contended where GPUs are in
    exclusive-process mode.
    """
    def _boom(*args: Any, **kwargs: Any):
        raise AssertionError("CPU mesh must not touch CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    monkeypatch.setattr(torch.cuda, "get_device_properties", _boom)
    monkeypatch.setattr(torch.cuda, "current_device", _boom)

    assert unit_budget_bytes(ParallelDims(), "cpu") > 0
    model = _BlockModel(n_blocks=6, floats_per_block=256 * 1024)
    # Grouping must work end to end on a CPU mesh without any CUDA call.
    assert _fsdp_unit_groups(model, ParallelDims(dp_shard=2), "cpu") == ()
    dims = ParallelDims(dp_shard=2, fsdp_unit_budget_gb=2 / 1024)
    assert [len(g) for g in _fsdp_unit_groups(model, dims, "cpu")] == [2, 2, 2]


@pytest.mark.unit
def test_a_tighter_budget_never_produces_fewer_groups():
    """Monotonicity is the property that makes the knob predictable."""
    model = _BlockModel(n_blocks=16, floats_per_block=256 * 1024)
    counts = [
        len(_fsdp_unit_groups(model, ParallelDims(dp_shard=8, fsdp_unit_budget_gb=gib / 1024)))
        for gib in (16, 8, 4, 2, 1)
    ]
    assert counts == sorted(counts), counts
