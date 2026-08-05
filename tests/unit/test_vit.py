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
    tokens, coords = model.embed(x)
    assert tokens.shape == (2, 8, 32)
    assert coords.shape == (2, 8, 3), "every token carries its patch coordinate"
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
    tokens, coords = model.embed(torch.randn(2, 1, 16, 16, 16))
    assert model.encode(tokens[:, :3], coords[:, :3]).shape == (2, 3, 32)


@pytest.mark.unit
def test_multichannel_input_changes_patch_volume():
    model = _tiny(in_channels=3)
    assert model.patch_volume == 512 * 3
    assert model.embed(torch.randn(2, 3, 16, 16, 16))[0].shape == (2, 8, 32)


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
    tokens, coords = model.embed(volumes)
    assert tokens.shape == (2, 8, 32)
    assert model.encode(tokens, coords).shape == (2, 8, 32)


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
    from layers.attention import SelfAttention

    assert all(isinstance(block.attn, SelfAttention) for block in _tiny().blocks)


# --------------------------------------------------------------------------------------------
# Position: axial rotary embeddings on patch coordinates, replacing the learned table.
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_has_no_learned_position_table():
    # A regression guard on the switch to rotary position: a leftover table would mean two position
    # mechanisms stacked, and would silently pin the model to one img_size.
    names = dict(_tiny().named_parameters())
    assert not any("pos_embed" in name for name in names)
    assert any("rotary.inv_freqs" in name for name in names)


@pytest.mark.unit
def test_patch_coordinates_are_grid_indices_in_token_order():
    # The ordering detail that could silently be wrong: coordinates must enumerate the grid in the
    # same order the convolution flattens its patches, last axis fastest.
    model = _tiny()
    coords = model.patch_coords(1, torch.device("cpu"))
    assert coords.shape == (1, 8, 3)
    assert torch.equal(coords[0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(coords[0, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(coords[0, -1], torch.tensor([1.0, 1.0, 1.0]))


@pytest.mark.unit
def test_coordinates_line_up_with_patch_content():
    # Ties coordinates to the actual pixels rather than to themselves: mark one patch of the volume,
    # find which token it became, and check that token's coordinate is the patch's grid position.
    # A transposed or misordered coordinate grid passes every shape test but fails this.
    model = _tiny()
    volumes = torch.zeros(1, 1, 16, 16, 16)
    volumes[0, 0, 8:16, 0:8, 8:16] = 1.0  # the patch at grid position (1, 0, 1)

    patches = model.patchify(volumes)
    marked = (patches.sum(dim=-1)[0] > 0).nonzero().flatten()
    assert marked.numel() == 1, "exactly one patch should be marked"

    coords = model.patch_coords(1, torch.device("cpu"))
    assert torch.equal(coords[0, marked[0]], torch.tensor([1.0, 0.0, 1.0]))


@pytest.mark.unit
def test_position_changes_the_encoding():
    # Proves rotary position is actually applied. Without it the encoder would be a set function and
    # identical tokens at different places would encode identically.
    torch.manual_seed(0)
    model = _tiny().eval()
    tokens = torch.randn(1, 6, 32)
    with torch.no_grad():
        here = model.encode(tokens, torch.zeros(1, 6, 3))
        there = model.encode(tokens, torch.arange(6.0).reshape(1, 6, 1).expand(1, 6, 3))
    assert not torch.allclose(here, there, atol=1e-4)


@pytest.mark.unit
def test_encoding_is_invariant_to_shifting_every_coordinate():
    # Rotary attention sees displacements, not absolute positions, so a uniform shift is a no-op.
    torch.manual_seed(0)
    model = _tiny().eval()
    tokens = torch.randn(1, 6, 32)
    coords = torch.randn(1, 6, 3)
    with torch.no_grad():
        here = model.encode(tokens, coords)
        there = model.encode(tokens, coords + torch.tensor([9.0, -3.0, 0.5]))
    assert torch.allclose(here, there, atol=1e-4)


@pytest.mark.unit
def test_position_comes_only_from_coordinates_not_sequence_order():
    # Permuting tokens and coordinates together must permute the output the same way. This holds
    # only if nothing in the encoder depends on a token's index in the sequence -- which is what
    # removing the learned table bought, and what lets MAE pass an arbitrary visible subset.
    torch.manual_seed(0)
    model = _tiny().eval()
    tokens, coords = model.embed(torch.randn(1, 1, 16, 16, 16))
    order = torch.randperm(tokens.shape[1])

    with torch.no_grad():
        straight = model.encode(tokens, coords)
        shuffled = model.encode(tokens[:, order], coords[:, order])
    assert torch.allclose(shuffled, straight[:, order], atol=1e-5)


@pytest.mark.unit
def test_encode_rejects_coordinates_that_do_not_match_the_tokens():
    model = _tiny()
    tokens, coords = model.embed(torch.randn(1, 1, 16, 16, 16))
    with pytest.raises(ValueError, match="every token needs a coordinate"):
        model.encode(tokens[:, :3], coords)


@pytest.mark.unit
def test_rejects_a_head_too_small_for_three_axis_rotary():
    # New constraint from axial rotary position: every spatial axis needs a channel pair, so
    # head_dim must be at least 6. Irrelevant at realistic widths (head_dim 64) but it does rule out
    # some very narrow toy configurations, so it fails at construction with the fix in the message.
    with pytest.raises(ValueError, match="head_dim 4 cannot cover 3 spatial axes"):
        _tiny(embed_dim=32, num_heads=8)
