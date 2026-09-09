"""Can a promptable segmenter learn to return the object a click points at?

The end-to-end check that the strategy is wired: prompt -> mask -> loss -> gradient -> better mask.
Every unit test around it verifies one link; only this one verifies that the chain closes. If it
fails, no measurement on real data means anything, because the model would be learning from a
signal that never reaches the thing being prompted.

Two objects, one crop, the image showing the labels so a three-layer encoder can solve it at all.
Measured across three seeds: IoU 0.99-1.00 after 150 steps, in about nine seconds. The assertion is
0.85, comfortably clear of that, so a failure means the chain broke rather than that the run was
unlucky.
"""

from __future__ import annotations

import pytest
import torch

import components  # noqa: F401  (populates the registries as a real run would)
from algorithms.registry import AlgorithmRegistry
from models.registry import ModelRegistry

STEPS = 150


@pytest.mark.slow
def test_promptable_segmentation_overfits_a_single_batch():
    torch.manual_seed(0)
    model = ModelRegistry.build(
        "vit3d", img_size=(24,) * 3, patch_size=(8,) * 3, embed_dim=96, depth=3, num_heads=4
    )
    algorithm = AlgorithmRegistry.build(
        "promptable_seg", model, None,
        input_axes="lcxyz", prompt_dim=48, decoder_heads=4, mask_upscale=4,
        mask_feature_dim=16, masks_per_sample=2, min_object_voxels=64, rounds=1,
    )

    labels = torch.zeros(1, 1, 24, 24, 24, dtype=torch.int64)
    labels[0, 0, 2:12, 2:12, 2:12] = 11
    labels[0, 0, 14:22, 14:22, 6:18] = 22
    image = (labels > 0).float().unsqueeze(1) + 0.1 * torch.randn(1, 1, 1, 24, 24, 24)
    batch = {"img": image, "label": labels}

    optimizer = torch.optim.AdamW(algorithm.parameters(), lr=1e-3)
    first = None
    metrics: dict[str, torch.Tensor] = {}
    for _ in range(STEPS):
        metrics = algorithm.training_step(batch)
        first = metrics["loss"].item() if first is None else first
        optimizer.zero_grad(set_to_none=True)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(algorithm.parameters(), 1.0)
        optimizer.step()

    assert first is not None
    assert metrics["loss"].item() < first / 10
    assert metrics["first_iou"].item() > 0.85
    # The head that ranks candidate masks has to be right too, or automatic mask generation would
    # be choosing between them at random.
    assert metrics["first_iou_error"].item() < 0.2
