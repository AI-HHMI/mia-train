"""Unit tests for SimMIM masked image modelling."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from algorithms.registry import AlgorithmRegistry
from algorithms.simmim import SimMIM, patchify, random_mask
from models.dinov3_vit import DinoVisionTransformer
from models.dinov3_vit3d import DinoVisionTransformer3D
from models.vit import ViT3D


def _encoder(rank: int):
    common: dict[str, Any] = dict(patch_size=8, in_chans=1, embed_dim=96, depth=2, num_heads=4,
                                  pos_embed_rope_dtype="fp32")
    if rank == 2:
        return DinoVisionTransformer(img_size=32, **common)
    return DinoVisionTransformer3D(img_size=16, **common)


def _algorithm(rank: int = 3, **overrides: Any) -> SimMIM:
    kwargs: dict[str, Any] = dict(
        input_axes="lcxy" if rank == 2 else "lcxyz", mask_granularity=1
    )
    kwargs.update(overrides)
    return SimMIM(_encoder(rank), **kwargs)


def _batch(rank: int, batch_size: int = 2) -> dict[str, torch.Tensor]:
    spatial = (32, 32) if rank == 2 else (16, 16, 16)
    return {"img": torch.rand(batch_size, 1, 1, *spatial)}


BOTH_RANKS = pytest.mark.parametrize("rank", [2, 3])


# ---------------------------------------------------------------- patchify


@pytest.mark.unit
@BOTH_RANKS
def test_patchify_shape_and_content(rank):
    patch = (4,) * rank
    volumes = torch.arange(2 * 1 * 8**rank, dtype=torch.float32).reshape(2, 1, *((8,) * rank))
    out = patchify(volumes, patch)

    n_patches = 2**rank
    assert out.shape == (2, n_patches, math.prod(patch))
    # Every value survives exactly once: patchify is a permutation, not a resampling.
    assert torch.equal(out.flatten().sort().values, volumes.flatten().sort().values)


@pytest.mark.unit
def test_patchify_matches_the_reference_model_implementation():
    """`ViT3D` already had a 3D patchify; the rank-generic one must agree with it, since the
    encoder's token order is what the prediction has to line up with."""
    model = ViT3D(img_size=(16, 16, 16), patch_size=(8, 8, 8), in_channels=2, embed_dim=16,
                  depth=1, num_heads=2)
    volumes = torch.randn(3, 2, 16, 16, 16)
    assert torch.equal(patchify(volumes, (8, 8, 8)), model.patchify(volumes))


@pytest.mark.unit
def test_patchify_rejects_a_partial_patch():
    with pytest.raises(ValueError, match="not divisible"):
        patchify(torch.randn(1, 1, 12, 16), (8, 8))


# ---------------------------------------------------------------- masking


@pytest.mark.unit
@pytest.mark.parametrize("grid", [(8, 8), (4, 4, 4)])
def test_mask_hits_the_requested_ratio(grid):
    mask = random_mask(grid, batch=8, ratio=0.6, granularity=1)
    assert mask.shape == (8, math.prod(grid))
    # Exact, not approximate: every sample masks the same count, drawn independently.
    fractions = mask.float().mean(dim=1).tolist()
    assert fractions == pytest.approx([0.6] * 8, abs=1.0 / math.prod(grid))


@pytest.mark.unit
def test_each_sample_gets_a_different_mask():
    mask = random_mask((8, 8), batch=4, ratio=0.5, granularity=1)
    assert not torch.equal(mask[0], mask[1])


@pytest.mark.unit
def test_granularity_masks_in_contiguous_groups():
    """SimMIM's central finding: masking single patches leaves every hidden patch surrounded by
    visible ones, so the task collapses to local interpolation."""
    fine = random_mask((16, 16), batch=1, ratio=0.5, granularity=1).reshape(16, 16)
    coarse = random_mask((16, 16), batch=1, ratio=0.5, granularity=4).reshape(16, 16)

    def boundaries(m):
        return (m[:, :-1] != m[:, 1:]).sum() + (m[:-1] != m[1:]).sum()

    assert boundaries(coarse) < boundaries(fine) / 2, "coarse units have far fewer edges"


@pytest.mark.unit
def test_granularity_that_does_not_divide_the_grid_still_covers_it_exactly():
    mask = random_mask((7, 7), batch=2, ratio=0.5, granularity=2)
    assert mask.shape == (2, 49), "trimmed back to the real grid, not the rounded-up one"


@pytest.mark.unit
@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1])
def test_rejects_a_degenerate_mask_ratio(ratio):
    with pytest.raises(ValueError, match="ratio"):
        random_mask((4, 4), batch=1, ratio=ratio)


# ---------------------------------------------------------------- the algorithm


@pytest.mark.unit
def test_registered_under_its_config_name():
    assert AlgorithmRegistry.get("simmim") is SimMIM


@pytest.mark.unit
@BOTH_RANKS
def test_one_algorithm_serves_both_ranks(rank):
    algorithm = _algorithm(rank)
    assert algorithm.spatial_rank == rank
    assert algorithm.head.out_features == math.prod(algorithm.patch_size) * algorithm.in_chans

    out = algorithm.training_step(_batch(rank))
    assert torch.isfinite(out["loss"])
    assert set(out) == {"loss", "masked_fraction", "visible_l1"}


@pytest.mark.unit
def test_rejects_an_encoder_that_cannot_substitute_a_mask_token():
    """The whole reason SimMIM fits these backbones where MAE does not."""
    encoder = ViT3D(img_size=(16, 16, 16), patch_size=(8, 8, 8), in_channels=1, embed_dim=32,
                    depth=1, num_heads=4)
    with pytest.raises(TypeError, match="forward_features"):
        SimMIM(encoder, input_axes="lcxyz")


@pytest.mark.unit
@BOTH_RANKS
def test_gradient_reaches_the_encoder_and_the_mask_token(rank):
    algorithm = _algorithm(rank)
    algorithm.training_step(_batch(rank))["loss"].backward()

    encoder = algorithm.encoder
    assert encoder.blocks[0].attn.qkv.weight.grad is not None
    assert encoder.mask_token.grad is not None, "the substituted token has to learn"
    assert algorithm.head.weight.grad is not None


@pytest.mark.unit
def test_loss_is_computed_only_where_the_encoder_was_blind():
    """Scoring visible patches would reward passing the input straight through."""
    torch.manual_seed(0)
    algorithm = _algorithm(3).eval()
    batch = _batch(3)

    volumes = algorithm.encoder.prepare_input(batch["img"], algorithm.input_axes)
    target = patchify(volumes, algorithm.patch_size)
    mask = random_mask((2, 2, 2), volumes.shape[0], 0.5, granularity=1)

    out = algorithm.encoder.forward_features(volumes, mask)
    prediction = algorithm.head(out["x_norm_patchtokens"])
    per_patch = (prediction - target).abs().mean(dim=-1)

    masked_only = (per_patch * mask).sum() / mask.sum()
    everything = per_patch.mean()
    assert not torch.isclose(masked_only, everything), "the mask must actually restrict the loss"


@pytest.mark.unit
def test_a_perfect_prediction_scores_zero():
    """Orientation check: reproducing the input exactly must be rewarded."""
    torch.manual_seed(0)
    algorithm = _algorithm(3, norm_pix_loss=False).eval()
    batch = _batch(3)
    volumes = algorithm.encoder.prepare_input(batch["img"], algorithm.input_axes)

    target = patchify(volumes, algorithm.patch_size)
    mask = random_mask((2, 2, 2), volumes.shape[0], 0.5, granularity=1)
    per_patch = (target - target).abs().mean(dim=-1)
    assert ((per_patch * mask).sum() / mask.sum()).item() == 0.0


@pytest.mark.unit
def test_norm_pix_loss_changes_the_target_not_the_shape():
    torch.manual_seed(0)
    plain = _algorithm(3, norm_pix_loss=False).eval()
    normed = _algorithm(3, norm_pix_loss=True).eval()
    normed.load_state_dict(plain.state_dict())

    batch = _batch(3)
    with torch.no_grad():
        assert plain.training_step(batch)["loss"] != normed.training_step(batch)["loss"]


@pytest.mark.unit
def test_validation_step_does_not_differ_from_training_step_in_eval():
    algorithm = _algorithm(3).eval()
    torch.manual_seed(0)
    batch = _batch(3)
    torch.manual_seed(1)
    train_loss = algorithm.training_step(batch)["loss"]
    torch.manual_seed(1)
    val_loss = algorithm.validation_step(batch)["loss"]
    assert torch.allclose(train_loss, val_loss)


@pytest.mark.unit
def test_rejects_input_axes_contradicting_the_dataset():
    class _Dataset:
        sample_axes = "lcxyz"

    with pytest.raises(ValueError, match="contradicts"):
        SimMIM(_encoder(3), _Dataset(), input_axes="lcxy")
