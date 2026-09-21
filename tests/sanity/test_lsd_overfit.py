"""The descriptor head learns: on one fixed crop whose image shows the objects, `loss_lsd` falls.

A convergence check rather than a unit test, because the *dynamics* are what is being asked
about. The descriptor loss is a squared error on a sigmoid, whose gradient is at most a quarter of
the affinity BCE's per element, and on a crop whose image is pure noise it sits still while the
affinity loss falls -- which looks like a broken head and is not one. Here the image is the
objects' interiors with their boundaries dark, so the target is learnable from what the network
sees, and the loss has to come down.
"""

from __future__ import annotations

import pytest
import torch

from algorithms.affinity_seg import AffinitySegmentation
from models.vit import ViT3D

CROP = 16
PIXEL_SIZE = (9.0, 9.0, 20.0)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    coarse = torch.randint(0, 4, (1, CROP // 4, CROP // 4, CROP // 4))
    labels = coarse
    for axis in (1, 2, 3):
        labels = labels.repeat_interleave(4, dim=axis)
    # Interiors bright, every face between two labels dark, a little noise.
    image = (labels > 0).float()
    for axis in (1, 2, 3):
        extent = labels.shape[axis]
        same = labels.narrow(axis, 1, extent - 1) == labels.narrow(axis, 0, extent - 1)
        interior = torch.ones_like(labels, dtype=torch.bool)
        interior.narrow(axis, 1, extent - 1).logical_and_(same)
        image = image * interior.float()
    image = image + 0.1 * torch.randn_like(image)
    return {
        "img": image[:, None, None],
        "label": labels[:, None],
        "pixel_size": torch.tensor(PIXEL_SIZE).expand(1, 1, 3).clone(),
    }


@pytest.mark.slow
def test_descriptor_loss_falls_on_an_informative_crop():
    torch.manual_seed(0)
    encoder = ViT3D(
        img_size=(CROP,) * 3, patch_size=(8, 8, 8), in_channels=1, embed_dim=32, depth=1,
        num_heads=4,
    )
    algorithm = AffinitySegmentation(
        encoder, input_axes="lcxyz", decoder_hidden_dim=32, long_range=4,
        split_disconnected=False, lsd_sigma=20.0,
    )
    batch = _batch()
    optimizer = torch.optim.Adam(algorithm.parameters(), lr=1e-2)
    first = last = None
    for _ in range(150):
        metrics = algorithm.training_step(batch)
        last = metrics["loss_lsd"].item()
        first = last if first is None else first
        optimizer.zero_grad()
        metrics["loss"].backward()
        optimizer.step()
    assert first is not None and last is not None
    assert last < 0.7 * first, (first, last)
