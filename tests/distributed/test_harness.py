import pytest
import torch
import torch.distributed as dist


def _all_reduce_sum(rank: int, world_size: int) -> float:
    tensor = torch.tensor([float(rank + 1)])
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


@pytest.mark.cpu_dist
def test_all_reduce_sum(run_distributed):
    world_size = 2
    results = run_distributed(_all_reduce_sum, world_size=world_size)
    expected = float(sum(rank + 1 for rank in range(world_size)))
    assert results == [expected] * world_size
