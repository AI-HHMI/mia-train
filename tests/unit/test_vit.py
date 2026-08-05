from __future__ import annotations

from typing import Any

import pytest
import torch

from models.base import BaseModel
from models.registry import ModelRegistry
from models.vit import ViT3D


def _tiny(**overrides: Any) -> ViT3D:
    # Annotated because a heterogeneous dict literal infers `dict[str, object]`, which cannot be
    # unpacked into ViT3D's typed signature.
    kwargs: dict[str, Any] = dict(
        img_size=(16, 16, 16), patch_size=(8, 8, 8), in_channels=1,
        embed_dim=32, depth=2, num_heads=4,
    )
    kwargs.update(overrides)
    return ViT3D(**kwargs)


@pytest.mark.unit
def test_registered_under_its_config_name():
    assert ModelRegistry.get("vit3d") is ViT3D


@pytest.mark.unit
def test_patch_grid_and_counts():
    model = _tiny()
    assert model.grid_size == (2, 2, 2)
    assert model.num_patches == 8
    assert model.patch_volume == 512  # 8*8*8*1


@pytest.mark.unit
def test_rejects_indivisible_image_size():
    with pytest.raises(ValueError, match="divisible"):
        _tiny(img_size=(20, 16, 16))


@pytest.mark.unit
def test_rejects_non_3d_sizes():
    with pytest.raises(ValueError, match="3D"):
        _tiny(patch_size=(8, 8))


@pytest.mark.unit
def test_embed_and_forward_shapes():
    model = _tiny()
    x = torch.randn(2, 1, 16, 16, 16)
    assert model.embed(x).shape == (2, 8, 32)
    assert model(x).shape == (2, 32)


@pytest.mark.unit
def test_embed_rejects_wrong_input_shape():
    model = _tiny()
    with pytest.raises(ValueError, match="expected input"):
        model.embed(torch.randn(2, 1, 8, 8, 8))


@pytest.mark.unit
def test_encoder_accepts_a_token_subset():
    # MAE feeds only the visible tokens, so the encoder must not assume a full grid.
    model = _tiny()
    tokens = model.embed(torch.randn(2, 1, 16, 16, 16))
    assert model.encode(tokens[:, :3]).shape == (2, 3, 32)


@pytest.mark.unit
def test_multichannel_input_changes_patch_volume():
    model = _tiny(in_channels=3)
    assert model.patch_volume == 512 * 3
    assert model.embed(torch.randn(2, 3, 16, 16, 16)).shape == (2, 8, 32)


@pytest.mark.unit
def test_flops_and_parameter_counts_are_positive():
    model = _tiny()
    assert model.flops((1, 16, 16, 16)) > 0
    assert model.num_parameters() > 0
    assert model.num_parameters(trainable_only=True) == model.num_parameters()


@pytest.mark.unit
def test_prepare_input_squeezes_the_single_scale_level():
    model = _tiny()
    volumes = model.prepare_input(torch.randn(2, 1, 16, 16, 16), "lzyx")
    assert volumes.shape == (2, 1, 16, 16, 16)  # level dropped, singleton channel added


@pytest.mark.unit
def test_prepare_input_keeps_a_declared_channel_axis():
    # miao's real config uses output_axes "lcxyz".
    model = _tiny()
    volumes = model.prepare_input(torch.randn(2, 1, 1, 16, 16, 16), "lcxyz")
    assert volumes.shape == (2, 1, 16, 16, 16)


@pytest.mark.unit
def test_prepare_input_finds_the_level_axis_even_when_it_is_not_first():
    model = _tiny()
    volumes = model.prepare_input(torch.randn(2, 1, 1, 16, 16, 16), "clzyx")
    assert volumes.shape == (2, 1, 16, 16, 16)


@pytest.mark.unit
def test_prepare_input_output_feeds_embed_directly():
    # The whole point of prepare_input: its result must be what embed accepts.
    model = _tiny()
    volumes = model.prepare_input(torch.randn(2, 1, 1, 16, 16, 16), "lcxyz")
    assert model.embed(volumes).shape == (2, 8, 32)


@pytest.mark.unit
def test_prepare_input_rejects_more_than_one_scale_level():
    # A plain ViT has one patch grid at one resolution. Silently folding 3 levels into the batch
    # would redefine batch_size; treating them as channels would assert a pixel correspondence
    # that miao's levels do not have.
    model = _tiny()
    with pytest.raises(ValueError, match="single-scale"):
        model.prepare_input(torch.randn(2, 3, 16, 16, 16), "lzyx")


@pytest.mark.unit
def test_prepare_input_rejects_axes_without_a_level_axis():
    model = _tiny()
    with pytest.raises(ValueError, match="must contain 'l'"):
        model.prepare_input(torch.randn(2, 16, 16, 16), "zyx")


@pytest.mark.unit
def test_prepare_input_rejects_axes_a_3d_encoder_cannot_read():
    # A trailing channel would put a spatial axis where the channel belongs.
    model = _tiny()
    with pytest.raises(ValueError, match="3D encoder"):
        model.prepare_input(torch.randn(2, 1, 16, 16, 16, 1), "lzyxc")


@pytest.mark.unit
def test_prepare_input_rejects_a_batch_rank_that_contradicts_the_axes():
    model = _tiny()
    with pytest.raises(ValueError, match="-D batch"):
        model.prepare_input(torch.randn(2, 1, 1, 16, 16, 16), "lzyx")


@pytest.mark.unit
def test_base_model_declines_to_guess_an_input_layout():
    # BaseModel cannot know an architecture's input contract, so it must refuse rather than
    # invent one. _DummyModel here deliberately does not override prepare_input.
    class _NoPrepare(BaseModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def flops(self, input_shape: tuple[int, ...]) -> int:
            return 0

    with pytest.raises(NotImplementedError, match="prepare_input"):
        _NoPrepare().prepare_input(torch.randn(1, 1, 4, 4, 4), "lzyx")


@pytest.mark.unit
def test_blocks_use_the_configured_attention_backend():
    model = _tiny(attention_backend="sdpa")
    assert model.attention_backend == "sdpa"
    assert all(block.attn.backend == "sdpa" for block in model.blocks)


@pytest.mark.unit
def test_attention_backend_defaults_to_auto():
    assert _tiny().attention_backend == "auto"
    assert all(block.attn.backend == "auto" for block in _tiny().blocks)


@pytest.mark.unit
def test_blocks_hold_our_own_attention_module():
    from models.attention import SelfAttention

    assert all(isinstance(block.attn, SelfAttention) for block in _tiny().blocks)
