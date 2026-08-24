"""`MuViT3D.patch_features` returns the finest level's tokens, so a dense head can drive it.

`BaseModel` declines `patch_features` for multi-scale encoders by default. MuViT3D overrides that
with a specific contract -- the finest level's tokens and grid -- and these pin the parts of it that
would otherwise fail silently: the wrong slice of the sequence is still the right shape, and a head
fed coarse tokens would train without complaint on a 4x-too-large receptive field.
"""

from __future__ import annotations

import pytest
import torch

from models.muvit import MuViT3D


def _model(levels=(1, 2, 4), img=(32, 32, 32), patch=(8, 8, 8), dim=32, heads=4, depth=2):
    torch.manual_seed(0)
    return MuViT3D(
        levels=levels, img_size=img, patch_size=patch,
        embed_dim=dim, depth=depth, num_heads=heads,
    )


def _sample(model, batch=2):
    torch.manual_seed(1)
    return torch.randn(batch, model.num_levels, model.in_channels, *model.img_size)


def test_returns_one_level_of_tokens_on_the_finest_grid():
    model = _model()
    tokens, grid = model.patch_features(_sample(model))
    assert grid == model.grid_size
    assert tokens.shape == (2, model.patches_per_level, model.embed_dim)
    # Not the whole joint sequence: that is the mistake this contract exists to prevent.
    assert tokens.shape[1] * model.num_levels == model.num_patches


def test_tokens_are_the_finest_level_not_some_other_slice():
    """The finest level is the first block of the sequence; assert that, don't assume it."""
    model = _model()
    x = _sample(model)
    with torch.no_grad():
        joint_tokens, coords = model.embed(x)
        joint = model.encode(joint_tokens, coords)
        got, _ = model.patch_features(x)
    torch.testing.assert_close(got, joint[:, : model.patches_per_level])
    # And it is genuinely a different tensor from the coarse blocks, so the slice is load-bearing.
    coarse = joint[:, model.patches_per_level : 2 * model.patches_per_level]
    assert not torch.allclose(got, coarse)


def test_coarser_levels_still_influence_the_output():
    """Joint attention is the point: perturbing only a coarse level must move the fine tokens.

    If this ever fails, the coarse levels are being carried but not attended over, and the
    multi-scale model is doing single-scale work at three times the cost.
    """
    model = _model()
    x = _sample(model)
    perturbed = x.clone()
    perturbed[:, 1] += 5.0  # touch only the middle level
    with torch.no_grad():
        base, _ = model.patch_features(x)
        moved, _ = model.patch_features(perturbed)
    assert not torch.allclose(base, moved, atol=1e-5)


def test_single_level_model_returns_the_whole_sequence():
    model = _model(levels=(1,))
    tokens, grid = model.patch_features(_sample(model))
    assert tokens.shape[1] == model.num_patches == model.patches_per_level
    assert grid == model.grid_size


def test_explicit_bbox_changes_the_result():
    """`bbox` defaults to concentric crops; a dataset with off-centre levels must be able to say so
    and have it matter, since wrong coordinates are otherwise undetectable."""
    model = _model()
    x = _sample(model)
    shifted = model.default_bbox(x.shape[0], x.device).clone()
    shifted[:, 1] += 17.0  # move the middle level off centre
    with torch.no_grad():
        default, _ = model.patch_features(x)
        explicit, _ = model.patch_features(x, shifted)
    assert not torch.allclose(default, explicit, atol=1e-5)


def test_rejects_a_sample_with_the_wrong_number_of_levels():
    model = _model()
    wrong = torch.randn(2, model.num_levels - 1, model.in_channels, *model.img_size)
    with pytest.raises(ValueError, match="expected input"):
        model.patch_features(wrong)
