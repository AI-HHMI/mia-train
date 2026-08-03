from __future__ import annotations

import pytest

from distributed.parallel_dims import ParallelDims


@pytest.mark.unit
def test_defaults_are_single_rank():
    dims = ParallelDims()
    assert dims.world_size == 1
    assert not dims.dp_enabled
    assert not dims.hsdp_enabled
    assert not dims.tp_enabled


@pytest.mark.unit
def test_world_size_is_product_of_dims():
    dims = ParallelDims(dp_replicate=2, dp_shard=3, tp=4)
    assert dims.world_size == 24


@pytest.mark.unit
def test_hsdp_enabled_requires_both_replicate_and_shard():
    assert ParallelDims(dp_replicate=2, dp_shard=2).hsdp_enabled
    assert not ParallelDims(dp_replicate=2, dp_shard=1).hsdp_enabled
    assert not ParallelDims(dp_replicate=1, dp_shard=2).hsdp_enabled


@pytest.mark.unit
def test_dp_enabled_true_for_replicate_only_or_shard_only():
    assert ParallelDims(dp_replicate=2).dp_enabled
    assert ParallelDims(dp_shard=2).dp_enabled
    assert not ParallelDims().dp_enabled


@pytest.mark.unit
def test_tp_enabled():
    assert ParallelDims(tp=2).tp_enabled
    assert not ParallelDims(tp=1).tp_enabled


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"dp_replicate": 0},
        {"dp_shard": 0},
        {"tp": 0},
        {"dp_replicate": -1},
    ],
)
def test_rejects_non_positive_dims(kwargs):
    with pytest.raises(ValueError):
        ParallelDims(**kwargs)
