"""Unit tests for the MuViT multi-resolution encoder."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from layers.common.blocks import TransformerBlock
from layers.common.rope import AxialRotaryEmbedding
from models.muvit import MuViT3D
from models.registry import ModelRegistry


def _tiny(**overrides: Any) -> MuViT3D:
    # Annotated because a heterogeneous dict literal infers `dict[str, object]`, which cannot be
    # unpacked into MuViT3D's typed signature.
    kwargs: dict[str, Any] = dict(
        levels=(1, 4),
        img_size=(16, 16, 16),
        patch_size=(8, 8, 8),
        in_channels=1,
        embed_dim=32,
        depth=2,
        num_heads=2,
        attention_backend="sdpa",
    )
    kwargs.update(overrides)
    return MuViT3D(**kwargs)


@pytest.mark.unit
def test_registered_under_its_name():
    assert ModelRegistry.build(
        "muvit3d", levels=(1, 4), img_size=(16, 16, 16), patch_size=(8, 8, 8), embed_dim=32,
        depth=1, num_heads=2,
    ).num_levels == 2


@pytest.mark.unit
def test_patch_grid_counts_every_level():
    model = _tiny()
    assert model.grid_size == (2, 2, 2)
    assert model.patches_per_level == 8
    assert model.num_patches == 16, "the joint sequence holds both levels"
    assert model.patch_volume == 512


@pytest.mark.unit
def test_embed_returns_tokens_and_coordinates():
    model = _tiny()
    tokens, coords = model.embed(torch.randn(2, 2, 1, 16, 16, 16))
    assert tokens.shape == (2, 16, 32)
    assert coords.shape == (2, 16, 3)


@pytest.mark.unit
def test_forward_pools_to_one_vector_per_sample():
    model = _tiny()
    assert model(torch.randn(2, 2, 1, 16, 16, 16)).shape == (2, 32)


@pytest.mark.unit
def test_encode_accepts_a_token_subset():
    # Masked autoencoding hands the encoder a visible subset, so the token count must be free --
    # but each surviving token has to keep its own coordinate.
    model = _tiny()
    tokens, coords = model.embed(torch.randn(2, 2, 1, 16, 16, 16))
    keep = torch.tensor([0, 3, 9])
    encoded = model.encode(tokens[:, keep], coords[:, keep])
    assert encoded.shape == (2, 3, 32)


@pytest.mark.unit
def test_encode_rejects_coordinates_that_do_not_match_the_tokens():
    # The failure this prevents is silent: masking the tokens but forgetting to mask the
    # coordinates alongside them would attach every token to the wrong place.
    model = _tiny()
    tokens, coords = model.embed(torch.randn(1, 2, 1, 16, 16, 16))
    with pytest.raises(ValueError, match="every token needs a coordinate"):
        model.encode(tokens[:, :4], coords)


# --------------------------------------------------------------------------------------------
# World coordinates: the part that makes this multi-resolution rather than multi-crop.
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_coarse_levels_span_proportionally_more_world():
    # Level l is l-times downsampled, so in the same pixel budget it must cover l times the scene.
    model = _tiny(levels=(1, 4))
    bbox = model.default_bbox(1, torch.device("cpu"))
    fine_extent = (bbox[0, 0, 1] - bbox[0, 0, 0])
    coarse_extent = (bbox[0, 1, 1] - bbox[0, 1, 0])
    assert torch.allclose(coarse_extent, 4 * fine_extent)


@pytest.mark.unit
def test_levels_share_a_centre_by_default():
    model = _tiny(levels=(1, 4, 16))
    bbox = model.default_bbox(1, torch.device("cpu"))
    centres = (bbox[0, :, 0, :] + bbox[0, :, 1, :]) / 2
    assert torch.allclose(centres, torch.zeros_like(centres)), "default crops are concentric"


@pytest.mark.unit
def test_every_level_is_centred_on_the_world_origin():
    # Follows from concentric boxes, and is what makes coordinates comparable across levels: the
    # mean patch position of each level lands on the same point.
    model = _tiny(levels=(1, 4))
    coords = model.world_coords(model.default_bbox(1, torch.device("cpu")))
    per_level = coords.reshape(1, 2, model.patches_per_level, 3)
    assert torch.allclose(per_level[0, 0].mean(0), per_level[0, 1].mean(0), atol=1e-5)


@pytest.mark.unit
def test_a_single_level_reproduces_plain_patch_coordinates():
    # With one level at scale 1 the world frame is just the pixel frame, so the corner patches sit
    # at the volume's own extremes. This anchors the coordinate convention to something checkable
    # by hand instead of only to itself.
    model = _tiny(levels=(1,), img_size=(16, 16, 16), patch_size=(8, 8, 8))
    coords = model.world_coords(model.default_bbox(1, torch.device("cpu")))
    half = 16 / 2 - 0.5
    assert torch.allclose(coords[0, 0], torch.full((3,), -half))
    assert torch.allclose(coords[0, -1], torch.full((3,), half))


@pytest.mark.unit
def test_absolute_position_in_the_volume_is_discarded():
    # Coordinates are reported relative to the finest level's centre. Datasets give absolute boxes
    # -- miao's are nanometres into the volume, measured in the hundreds of thousands -- and float32
    # would spend its precision on that offset instead of on the differences between neighbouring
    # patches. Rotary attention only ever sees differences, so removing the offset costs nothing.
    model = _tiny(levels=(1, 4))
    box = model.default_bbox(1, torch.device("cpu"))
    assert torch.allclose(
        model.world_coords(box), model.world_coords(box + 250_000.0), atol=1e-4
    )


@pytest.mark.unit
def test_off_centre_levels_keep_their_relative_displacement():
    # Recentring must not flatten the geometry between levels: miao's levels are *not* concentric
    # (measured: up to 240nm apart at a 33nm finest voxel, because each level's origin is floored to
    # its own voxel grid), and that offset is real information. Subtracting a per-level centre
    # instead of one per-sample reference would erase it.
    model = _tiny(levels=(1, 4))
    box = model.default_bbox(1, torch.device("cpu")).clone()
    box[:, 1, :, :] += 7.0  # nudge the coarse level off centre

    coords = model.world_coords(box).reshape(1, 2, model.patches_per_level, 3)
    offset = coords[0, 1].mean(0) - coords[0, 0].mean(0)
    assert torch.allclose(offset, torch.full((3,), 7.0), atol=1e-4)


@pytest.mark.unit
def test_the_finest_level_is_the_origin():
    model = _tiny(levels=(1, 4))
    coords = model.world_coords(model.default_bbox(2, torch.device("cpu")))
    finest = coords.reshape(2, 2, model.patches_per_level, 3)[:, 0]
    assert torch.allclose(finest.mean(dim=1), torch.zeros(2, 3), atol=1e-5)


@pytest.mark.unit
def test_output_is_invariant_to_sliding_the_world_frame():
    # Rotary embeddings position tokens by relative displacement, so translating every box must
    # leave the encoding untouched -- the model cares where patches are with respect to each other,
    # not where the crop sits in the source image.
    #
    # This also pins something no other test reaches: that attention rotates queries *and* keys.
    # Rotating only queries still yields a position-dependent model that trains, but its logits
    # depend on absolute position, and this test would catch it.
    model = _tiny(levels=(1, 4)).eval()
    volumes = torch.randn(2, 2, 1, 16, 16, 16)
    box = model.default_bbox(2, torch.device("cpu"))

    with torch.no_grad():
        here = model(volumes, box)
        there = model(volumes, box + 100.0)
    assert torch.allclose(here, there, atol=1e-4)


@pytest.mark.unit
def test_world_coords_rejects_a_malformed_box():
    model = _tiny(levels=(1, 4))
    with pytest.raises(ValueError, match=r"shape \(B, 2, 2, 3\)"):
        model.world_coords(torch.zeros(1, 3, 2, 3))


@pytest.mark.unit
def test_a_patch_and_its_coarse_counterpart_are_positioned_alike():
    # The architecture's central claim, tested end to end through the rotary embedding: patches
    # from different resolution levels that describe the same physical place receive the same
    # positional treatment. Here the centre of the fine level and the centre of the coarse level
    # coincide at the world origin, so their rotations must agree exactly.
    model = _tiny(levels=(1, 4))
    coords = model.world_coords(model.default_bbox(1, torch.device("cpu")))
    per_level = coords.reshape(1, 2, model.patches_per_level, 3)

    # Build a two-token sequence: one token per level, both placed at the shared centre.
    centre = torch.zeros(1, 2, 3)
    rope = AxialRotaryEmbedding(head_dim=16, spatial_rank=3)
    content = torch.randn(1, 1, 1, 16).expand(1, 2, 1, 16).contiguous()
    rotated = rope(centre)(content)
    assert torch.allclose(rotated[0, 0], rotated[0, 1], atol=1e-6)

    # And the two levels really do share that centre, rather than the test asserting it of itself.
    assert torch.allclose(per_level[0, 0].mean(0), per_level[0, 1].mean(0), atol=1e-5)


# --------------------------------------------------------------------------------------------
# prepare_input: the multi-scale counterpart of ViT3D's single-scale contract.
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_prepare_input_consumes_the_level_axis():
    model = _tiny(levels=(1, 4))
    prepared = model.prepare_input(torch.randn(3, 2, 16, 16, 16), "lzyx")
    assert prepared.shape == (3, 2, 1, 16, 16, 16), "channel axis added, level axis kept"


@pytest.mark.unit
def test_prepare_input_handles_a_declared_channel_axis():
    model = _tiny(levels=(1, 4))
    prepared = model.prepare_input(torch.randn(3, 2, 1, 16, 16, 16), "lcxyz")
    assert prepared.shape == (3, 2, 1, 16, 16, 16)


@pytest.mark.unit
def test_prepare_input_moves_a_trailing_level_axis_into_place():
    model = _tiny(levels=(1, 4))
    prepared = model.prepare_input(torch.randn(3, 1, 16, 16, 16, 2), "cxyzl")
    assert prepared.shape == (3, 2, 1, 16, 16, 16)


@pytest.mark.unit
def test_prepare_input_rejects_the_wrong_number_of_levels():
    # The per-level projections and level embeddings are indexed positionally, so a mismatch would
    # pair a level's data with another level's weights.
    model = _tiny(levels=(1, 4))
    with pytest.raises(ValueError, match="configured for 2 scale levels"):
        model.prepare_input(torch.randn(3, 3, 16, 16, 16), "lzyx")


@pytest.mark.unit
def test_prepare_input_requires_a_level_axis():
    model = _tiny()
    with pytest.raises(ValueError, match="must contain 'l'"):
        model.prepare_input(torch.randn(3, 1, 16, 16, 16), "czyx")


@pytest.mark.unit
def test_prepare_input_rejects_an_unreadable_axis_order():
    model = _tiny()
    with pytest.raises(ValueError, match="not usable by a 3D encoder"):
        model.prepare_input(torch.randn(3, 2, 16, 16, 16, 1), "lzyxc")


@pytest.mark.unit
def test_prepare_input_rejects_a_rank_mismatch():
    model = _tiny()
    with pytest.raises(ValueError, match="implies a 5-D batch"):
        model.prepare_input(torch.randn(3, 2, 1, 16, 16, 16), "lzyx")


@pytest.mark.unit
def test_prepare_input_output_feeds_embed():
    # The two halves of the contract have to meet: whatever prepare_input returns must be exactly
    # what embed accepts, or the model cannot be driven from a dataset batch at all.
    model = _tiny(levels=(1, 4))
    prepared = model.prepare_input(torch.randn(2, 2, 16, 16, 16), "lzyx")
    tokens, coords = model.embed(prepared)
    assert tokens.shape == (2, 16, 32)


# --------------------------------------------------------------------------------------------
# Per-level structure and configuration contracts.
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_each_level_gets_its_own_projection():
    # A pixel pattern means something different at 1x and 16x, so the projections must not be tied.
    model = _tiny(levels=(1, 4, 16))
    assert len(model.patch_proj) == 3
    first = model.patch_proj[0][0].weight
    assert all(first is not proj[0].weight for proj in list(model.patch_proj)[1:])


@pytest.mark.unit
def test_level_embeddings_distinguish_the_levels():
    # World coordinates cannot tell a fine patch from a coarse one at the same place; this can.
    model = _tiny(levels=(1, 4, 16))
    assert model.level_embed.shape == (3, 1, 32)
    assert model.level_embed.requires_grad


@pytest.mark.unit
def test_identical_volumes_at_different_levels_still_differ_as_tokens():
    # Feed the same pixels at every level and the tokens must not collapse. Note this holds via the
    # per-level projections alone -- it does not isolate the level embedding, which is what the
    # next test is for. Mutation testing is how that distinction surfaced: deleting the level
    # embedding leaves this test passing.
    torch.manual_seed(0)
    model = _tiny(levels=(1, 4))
    one_level = torch.randn(1, 1, 1, 16, 16, 16)
    tokens, _ = model.embed(one_level.expand(1, 2, 1, 16, 16, 16).contiguous())
    fine, coarse = tokens[:, :8], tokens[:, 8:]
    assert not torch.allclose(fine, coarse, atol=1e-4)


@pytest.mark.unit
def test_level_embedding_actually_reaches_the_tokens():
    # Isolates the level embedding by neutralising it: if it were never added, zeroing it would
    # change nothing. World coordinates cannot substitute for it, since a fine and a coarse patch
    # can occupy the same place.
    torch.manual_seed(0)
    model = _tiny(levels=(1, 4))
    volumes = torch.randn(1, 2, 1, 16, 16, 16)

    with torch.no_grad():
        before, _ = model.embed(volumes)
        model.level_embed.zero_()
        after, _ = model.embed(volumes)
    assert not torch.allclose(before, after, atol=1e-6)


@pytest.mark.unit
def test_every_transformer_layer_has_independent_rotary_parameters():
    # The paper gives each layer its own learnable frequencies, so a deep model can attend at
    # several distance scales. Sharing one module across layers would silently remove that.
    model = _tiny(depth=3)
    frequencies = [block.rotary.inv_freqs[0] for block in model.blocks]
    assert all(a is not b for a, b in zip(frequencies, frequencies[1:], strict=False))


@pytest.mark.unit
def test_gradients_reach_the_rotary_frequencies_and_every_level():
    model = _tiny(levels=(1, 4))
    model(torch.randn(2, 2, 1, 16, 16, 16)).pow(2).mean().backward()

    assert model.level_embed.grad is not None
    assert all(proj[0].weight.grad is not None for proj in model.patch_proj)
    assert all(block.rotary.inv_freqs[0].grad is not None for block in model.blocks)


@pytest.mark.unit
def test_patchify_is_invertible_in_layout():
    # Patchify defines the reconstruction target, so its layout has to be a pure regrouping of the
    # input: the same values, no loss and no duplication.
    model = _tiny(levels=(1, 4))
    volumes = torch.randn(2, 2, 1, 16, 16, 16)
    patches = model.patchify(volumes)
    assert patches.shape == (2, 16, 512)

    # A permutation check per (sample, level): the same multiset of values on both sides. Comparing
    # sums alone would pass even if patchify shuffled values between levels.
    regrouped = patches.reshape(2, 2, model.patches_per_level * model.patch_volume)
    flattened = volumes.reshape(2, 2, -1)
    assert torch.allclose(regrouped.sort(dim=-1).values, flattened.sort(dim=-1).values)


@pytest.mark.unit
def test_patchify_is_level_major_like_the_tokens():
    # Reconstruction predictions come out of `embed` level-major; if patchify disagreed, the loss
    # would compare a fine patch against a coarse target and still produce a plausible number.
    model = _tiny(levels=(1, 4))
    fine = torch.zeros(1, 1, 1, 16, 16, 16)
    coarse = torch.ones(1, 1, 1, 16, 16, 16)
    patches = model.patchify(torch.cat([fine, coarse], dim=1))
    assert (patches[0, : model.patches_per_level] == 0).all()
    assert (patches[0, model.patches_per_level :] == 1).all()


@pytest.mark.unit
def test_rejects_repeated_levels():
    with pytest.raises(ValueError, match="must be distinct"):
        _tiny(levels=(1, 1))


@pytest.mark.unit
def test_rejects_no_levels():
    with pytest.raises(ValueError, match="at least one scale level"):
        _tiny(levels=())


@pytest.mark.unit
def test_rejects_nonpositive_levels():
    with pytest.raises(ValueError, match="positive downsampling"):
        _tiny(levels=(1, 0))


@pytest.mark.unit
def test_rejects_a_patch_size_that_does_not_tile_the_volume():
    with pytest.raises(ValueError, match="must be divisible"):
        _tiny(img_size=(16, 16, 16), patch_size=(6, 6, 6))


@pytest.mark.unit
def test_rejects_a_non_3d_geometry():
    with pytest.raises(ValueError, match="must be 3D"):
        _tiny(img_size=(16, 16), patch_size=(8, 8))


@pytest.mark.unit
def test_embed_rejects_a_shape_it_was_not_configured_for():
    model = _tiny(levels=(1, 4))
    with pytest.raises(ValueError, match="expected input"):
        model.embed(torch.randn(2, 2, 1, 32, 32, 32))


@pytest.mark.unit
def test_attention_backend_reaches_every_block():
    model = _tiny(attention_backend="sdpa")
    assert model.attention_backend == "sdpa"
    assert all(block.attn.backend == "sdpa" for block in model.blocks)


@pytest.mark.unit
def test_extra_forward_methods_names_what_mae_calls():
    # FSDP2 all-gathers around `forward` only, so anything an algorithm calls directly must be
    # declared or its parameters stay sharded mid-call.
    assert set(_tiny().extra_forward_methods()) == {"embed", "encode"}


@pytest.mark.unit
def test_flops_grow_superlinearly_with_added_levels():
    # Attention is quadratic in the joint sequence, so a second level costs more than twice one
    # level. That cross-level term is exactly what the architecture is buying.
    one = _tiny(levels=(1,)).flops((1, 16, 16, 16))
    two = _tiny(levels=(1, 4)).flops((1, 16, 16, 16))
    assert two > 2 * one


@pytest.mark.unit
def test_uses_the_shared_transformer_block():
    # ViT3D and MuViT3D are both positioned by coordinate rotary attention, so they run the same
    # block. A separate near-identical copy per model would be two places to keep in step.
    block = TransformerBlock(32, 2, attention_backend="sdpa")
    x = torch.randn(2, 5, 32)
    assert block(x, torch.randn(2, 5, 3)).shape == x.shape
    assert all(isinstance(b, TransformerBlock) for b in _tiny().blocks)


@pytest.mark.unit
def test_paper_configuration_has_the_published_parameter_count():
    # The paper states ~25M parameters for its 12-layer encoder. Checking the transformer stack
    # against that catches a wrong feed-forward ratio or head layout -- the kind of mistake that
    # trains fine but is not the published architecture. The stack is measured rather than the
    # whole model because the per-level projections scale with patch volume and level count, which
    # differ between the paper's 2D setting and this 3D one.
    model = MuViT3D(
        levels=(1, 4, 16), img_size=(64, 64, 64), patch_size=(8, 8, 8),
        embed_dim=512, depth=12, num_heads=8, mlp_ratio=2.0,
    )
    stack = sum(p.numel() for p in model.blocks.parameters())
    assert 24e6 < stack < 27e6, f"transformer stack has {stack / 1e6:.1f}M parameters"
