"""Unit tests for the prompt encoder and the two-way decoder transformer."""

from __future__ import annotations

import pytest
import torch

from layers.common.prompt import (
    BACKGROUND,
    BOX_FAR,
    BOX_NEAR,
    FOREGROUND,
    PAD,
    PromptEncoder3D,
)
from layers.common.two_way import TwoWayTransformer

DIM = 32
GRID = (2, 3, 4)


def _labels() -> torch.Tensor:
    """One prompt of each shape: a click pair, a box plus a click, and padding only."""
    return torch.tensor(
        [
            [FOREGROUND, BACKGROUND, PAD, PAD],
            [BOX_NEAR, BOX_FAR, FOREGROUND, PAD],
            [PAD, PAD, PAD, PAD],
        ]
    )


@pytest.mark.unit
def test_padding_gets_the_not_a_point_token_at_the_origin():
    encoder = PromptEncoder3D(DIM)
    labels = _labels()
    coords = torch.rand(*labels.shape, 3) * 2 - 1
    tokens, positions, _ = encoder(coords, labels, GRID)

    padded = labels == PAD
    # The origin is the identity rotation, so a token with no location is not merely "somewhere":
    # it is at the centre of the crop, which is what the module promises.
    assert torch.all(positions[padded] == 0)
    torch.testing.assert_close(
        tokens[padded], encoder.not_a_point.weight.expand(int(padded.sum()), DIM)
    )
    # Everything else keeps the coordinate it was given, untouched.
    torch.testing.assert_close(positions[~padded], coords[~padded])


@pytest.mark.unit
def test_a_box_is_two_points_and_batches_with_them():
    encoder = PromptEncoder3D(DIM)
    labels = _labels()
    tokens, positions, _ = encoder(torch.zeros(*labels.shape, 3), labels, GRID)
    assert tokens.shape == (3, 4, DIM)
    # The four prompt types must be four distinguishable tokens, or the decoder cannot tell a
    # background click from a near box corner.
    used = encoder.point_embed.weight
    assert len({tuple(row.tolist()) for row in used}) == 4
    torch.testing.assert_close(tokens[1, 0], used[BOX_NEAR])
    torch.testing.assert_close(tokens[1, 1], used[BOX_FAR])


@pytest.mark.unit
def test_class_prompts_exist_only_when_asked_for():
    labels = _labels()
    coords = torch.zeros(*labels.shape, 3)

    plain = PromptEncoder3D(DIM, num_classes=0)
    assert plain.class_embed is None
    tokens, _, _ = plain(coords, labels, GRID)
    assert tokens.shape[1] == labels.shape[1], "no class token when none was configured"
    with pytest.raises(ValueError, match="num_classes=0"):
        plain(coords, labels, GRID, classes=torch.zeros(3, dtype=torch.long))

    semantic = PromptEncoder3D(DIM, num_classes=5)
    # Six rows, not five: the last is "no class named", so a purely geometric prompt still has a
    # token to occupy the slot and the batch stays rectangular.
    assert semantic.class_embed is not None
    assert semantic.class_embed.num_embeddings == 6
    tokens, positions, _ = semantic(coords, labels, GRID)
    assert tokens.shape[1] == labels.shape[1] + 1
    torch.testing.assert_close(tokens[:, -1], semantic.class_embed.weight[5].expand(3, DIM))
    assert torch.all(positions[:, -1] == 0), "a class names a what, not a where"

    named, _, _ = semantic(coords, labels, GRID, classes=torch.tensor([0, 2, 4]))
    torch.testing.assert_close(named[:, -1], semantic.class_embed.weight[torch.tensor([0, 2, 4])])


@pytest.mark.unit
def test_dense_embedding_is_learned_when_no_mask_is_given():
    encoder = PromptEncoder3D(DIM, mask_downscale=4)
    labels = _labels()
    _, _, dense = encoder(torch.zeros(*labels.shape, 3), labels, GRID)
    assert dense.shape == (3, DIM, *GRID)
    # One learned state broadcast everywhere, not a zero the decoder would have to tell apart from
    # a mask prompt that happens to be empty.
    torch.testing.assert_close(dense[0, :, 0, 0, 0], encoder.no_mask.weight[0])
    assert torch.all(dense[0] == dense[0, :, :1, :1, :1])


@pytest.mark.unit
def test_a_mask_prompt_is_taken_at_the_stride_the_decoder_emits():
    encoder = PromptEncoder3D(DIM, mask_downscale=4)
    labels = _labels()
    coords = torch.zeros(*labels.shape, 3)
    fine = tuple(extent * 4 for extent in GRID)
    _, _, dense = encoder(coords, labels, GRID, mask_input=torch.randn(3, 1, *fine))
    assert dense.shape == (3, DIM, *GRID)

    with pytest.raises(ValueError, match="mask prompt is"):
        encoder(coords, labels, GRID, mask_input=torch.randn(3, 1, *GRID))


@pytest.mark.unit
def test_mask_downscale_must_decompose_into_stride_two_convolutions():
    with pytest.raises(ValueError, match="power of two"):
        PromptEncoder3D(DIM, mask_downscale=3)
    # 1 is legal and means the mask prompt already sits on the token grid.
    flat = PromptEncoder3D(DIM, mask_downscale=1)
    _, _, dense = flat(
        torch.zeros(1, 1, 3), torch.tensor([[FOREGROUND]]), GRID,
        mask_input=torch.randn(1, 1, *GRID),
    )
    assert dense.shape == (1, DIM, *GRID)


@pytest.mark.unit
def test_two_way_transformer_shapes_and_broadcast_image_coordinates():
    torch.manual_seed(0)
    transformer = TwoWayTransformer(depth=2, dim=DIM, num_heads=4)
    prompts, tokens_n, image_n = 3, 5, 24
    tokens = torch.randn(prompts, tokens_n, DIM)
    image = torch.randn(prompts, image_n, DIM)
    token_coords = torch.rand(prompts, tokens_n, 3) * 2 - 1
    image_coords = torch.rand(1, image_n, 3) * 2 - 1

    out_tokens, out_image = transformer(tokens, image, token_coords, image_coords)
    assert out_tokens.shape == tokens.shape
    assert out_image.shape == image.shape

    # Broadcasting the shared image coordinates must be exactly materialising them, since that is
    # the only reason not to pay for `prompts` copies of one crop's rotation tables.
    expanded = transformer(tokens, image, token_coords, image_coords.expand(prompts, -1, -1))
    torch.testing.assert_close(out_tokens, expanded[0])
    torch.testing.assert_close(out_image, expanded[1])


@pytest.mark.unit
def test_the_decoder_sees_only_displacements():
    # The property the whole positional design rests on. Translate the prompt and the image
    # together and nothing may change: the decoder is told where a click is *relative to* the
    # patches, never where either sits in the volume. An additive position encoding -- the
    # reference's -- fails this outright.
    torch.manual_seed(0)
    transformer = TwoWayTransformer(depth=2, dim=DIM, num_heads=4, rope_min_period=0.1)
    tokens = torch.randn(2, 4, DIM)
    image = torch.randn(2, 16, DIM)
    token_coords = torch.rand(2, 4, 3) * 2 - 1
    image_coords = torch.rand(1, 16, 3) * 2 - 1
    shift = torch.tensor([0.13, -0.21, 0.05])

    here = transformer(tokens, image, token_coords, image_coords)
    there = transformer(tokens, image, token_coords + shift, image_coords + shift)
    torch.testing.assert_close(here[0], there[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(here[1], there[1], atol=1e-5, rtol=1e-4)


@pytest.mark.unit
def test_gradient_reaches_every_prompt_embedding_that_was_used():
    encoder = PromptEncoder3D(DIM, num_classes=3)
    transformer = TwoWayTransformer(depth=1, dim=DIM, num_heads=4)
    labels = _labels()
    coords = torch.rand(*labels.shape, 3) * 2 - 1
    tokens, positions, dense = encoder(coords, labels, GRID)
    image = dense.flatten(2).transpose(1, 2)
    image_coords = torch.rand(1, image.shape[1], 3) * 2 - 1

    transformer(tokens, image, positions, image_coords)[0].square().sum().backward()

    assert encoder.point_embed.weight.grad is not None
    used = torch.tensor([BACKGROUND, FOREGROUND, BOX_NEAR, BOX_FAR])
    assert torch.all(encoder.point_embed.weight.grad[used].abs().sum(-1) > 0)
    assert encoder.not_a_point.weight.grad.abs().sum() > 0
    assert encoder.no_mask.weight.grad.abs().sum() > 0, "the dense path must be differentiable too"
