"""`semantic_seg`'s Dice term is pooled over the global batch, not over each rank's share of it.

Two ranks with one crop each must report the Dice loss one process reports for both crops, and the
average of their gradients -- what data-parallel training applies -- must be that loss's gradient.
Per-rank Dice would fail both: a class missing from one rank's crop would cost that rank a full 1.0.
The two crops have equal numbers of supervised voxels, so the per-rank cross-entropy means average
to the single process's mean and the whole loss can be compared, not only its Dice half.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from algorithms.semantic_seg import SemanticSegmentation
from models.vit import ViT3D

CROP, PATCH, CLASSES, IGNORE = 32, 8, 4, 255


def _algorithm() -> SemanticSegmentation:
    torch.manual_seed(0)
    encoder = ViT3D(
        img_size=(CROP,) * 3, patch_size=(PATCH,) * 3, in_channels=1, embed_dim=32, depth=1,
        num_heads=4,
    )
    return SemanticSegmentation(
        encoder, input_axes="lcxyz", num_classes=CLASSES, decoder="subpixel", ignore_index=IGNORE,
        decoder_hidden_dim=8, decoder_readout_dim=4, decoder_zero_init_output=False,
        dice_weight=1.0,
    )


def _batch() -> dict[str, torch.Tensor]:
    """Two crops; class 3 is labelled in the first only, so per-rank Dice would differ."""
    generator = torch.Generator().manual_seed(1)
    labels = torch.randint(0, 3, (2, 1, CROP, CROP, CROP), generator=generator)
    labels[0, :, 4:12, 4:12, 4:12] = 3
    outside = torch.ones_like(labels, dtype=torch.bool)
    outside[..., 2:30, 3:29, 1:31] = False
    labels[outside] = IGNORE
    images = torch.rand(2, 1, 1, CROP, CROP, CROP, generator=generator, dtype=torch.float64)
    return {"img": images, "label": labels}


def _step(algorithm: SemanticSegmentation, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    reported = algorithm.training_step(batch)
    reported["loss"].backward()
    # NumPy, not tensors: a tensor sent from a worker travels through shared memory that is gone
    # once the worker exits, before the parent has read it.
    grads = {
        name: parameter.grad.detach().numpy().copy()
        for name, parameter in algorithm.named_parameters()
        if parameter.grad is not None
    }
    return {"dice_loss": float(reported["dice_loss"]), "grads": grads}


def _one_crop_per_rank(rank: int, world_size: int) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    batch = {name: value[rank : rank + 1] for name, value in _batch().items()}
    return _step(_algorithm(), batch)


@pytest.mark.cpu_dist
def test_two_ranks_pool_the_dice_terms_like_one_process(run_distributed):
    ranks = run_distributed(_one_crop_per_rank, world_size=2)

    torch.set_default_dtype(torch.float64)
    try:
        single = _step(_algorithm(), _batch())
    finally:
        torch.set_default_dtype(torch.float32)

    for result in ranks:
        assert result["dice_loss"] == pytest.approx(single["dice_loss"], rel=1e-12)
    for name, expected in single["grads"].items():
        averaged = (ranks[0]["grads"][name] + ranks[1]["grads"][name]) / 2
        torch.testing.assert_close(
            torch.from_numpy(averaged), torch.from_numpy(expected), rtol=1e-10, atol=1e-12, msg=name
        )
