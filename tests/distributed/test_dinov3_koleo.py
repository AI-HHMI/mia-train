"""DINOv3's KoLeo regulariser across ranks.

At one sample per rank -- this repo's usual 3D setting -- a rank's own share of the batch holds no
neighbour for its embedding, so the term exists only over the gathered, data-parallel batch.
"""

from __future__ import annotations

import pytest
import torch

from algorithms.dinov3.losses import KoLeoLoss

WIDTH = 16


def _embeddings(world_size: int) -> torch.Tensor:
    return torch.randn(world_size, WIDTH, generator=torch.Generator().manual_seed(0))


def _koleo_worker(rank: int, world_size: int) -> tuple[float, torch.Tensor]:
    local = _embeddings(world_size)[rank : rank + 1].clone().requires_grad_(True)
    loss = KoLeoLoss()(local)
    loss.backward()
    assert local.grad is not None
    return float(loss), local.grad.clone()


@pytest.mark.cpu_dist
def test_koleo_at_one_sample_per_rank_is_the_whole_batch_term(run_distributed):
    world_size = 4
    results = run_distributed(_koleo_worker, world_size=world_size)

    everyone = _embeddings(world_size).requires_grad_(True)
    expected = KoLeoLoss()(everyone)
    expected.backward()
    assert everyone.grad is not None

    losses = torch.tensor([loss for loss, _ in results])
    assert losses.mean().item() == pytest.approx(expected.item(), rel=1e-5)
    # A rank's loss is its own sample's term, and the gather's backward also hands it the pushes
    # its sample receives as other samples' nearest neighbour: the gradient of the sum of all the
    # ranks' terms, which is world_size times that of the one-process mean.
    for rank, (_, grad) in enumerate(results):
        assert grad.abs().sum() > 0, f"rank {rank}: KoLeo carries no gradient"
        assert torch.allclose(grad[0], world_size * everyone.grad[rank], atol=1e-5)
