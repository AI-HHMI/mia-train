"""Unit tests for axial rotary embeddings over continuous coordinates."""

from __future__ import annotations

import pytest
import torch

from layers.rope import AxialRotaryEmbedding, RotaryTables, split_rope_dims


def _attention_logits(
    query: torch.Tensor, key: torch.Tensor, rope: AxialRotaryEmbedding, coords: torch.Tensor
) -> torch.Tensor:
    """Rotate then take every query-key inner product -> (B, heads, N, N)."""
    tables = rope(coords)
    rotated_q, rotated_k = tables(query), tables(key)
    return torch.einsum("bnhd,bmhd->bhnm", rotated_q, rotated_k)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("head_dim", "rank", "expected"),
    [
        (64, 3, (24, 20, 20)),  # 64 does not divide by 3; the remainder goes to the first axis
        (48, 3, (16, 16, 16)),
        (64, 2, (32, 32)),
        (6, 3, (2, 2, 2)),  # the smallest split that gives every axis a pair
        (10, 3, (6, 2, 2)),
    ],
)
def test_split_uses_every_channel_in_even_pieces(head_dim, rank, expected):
    dims = split_rope_dims(head_dim, rank)
    assert dims == expected
    assert sum(dims) == head_dim, "a leftover channel would carry no position"
    assert all(width % 2 == 0 for width in dims), "rotation acts on pairs"


@pytest.mark.unit
def test_split_rejects_a_head_too_small_to_cover_the_axes():
    with pytest.raises(ValueError, match="at least 6"):
        split_rope_dims(4, 3)


@pytest.mark.unit
def test_rotation_preserves_shape_and_dtype():
    rope = AxialRotaryEmbedding(head_dim=16, spatial_rank=3)
    x = torch.randn(2, 5, 4, 16)
    rotated = rope(torch.randn(2, 5, 3))(x)
    assert rotated.shape == x.shape
    assert rotated.dtype == x.dtype


@pytest.mark.unit
def test_rotation_is_norm_preserving():
    # A rotation cannot change a vector's length. If this fails the "rotary" name is a lie and the
    # embedding is rescaling activations, which would show up as a training instability, not an
    # obvious error.
    rope = AxialRotaryEmbedding(head_dim=16, spatial_rank=3)
    x = torch.randn(2, 5, 4, 16)
    rotated = rope(torch.randn(2, 5, 3) * 10)(x)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


@pytest.mark.unit
def test_attention_logits_depend_only_on_relative_position():
    # The defining property of RoPE, and the reason it works on world coordinates at all: shifting
    # every coordinate by a constant must leave attention untouched. Without it, MuViT's absolute
    # world coordinates would make the model sensitive to where a crop sits in the source image
    # rather than to how its patches relate to each other.
    torch.manual_seed(0)
    rope = AxialRotaryEmbedding(head_dim=16, spatial_rank=3)
    query, key = torch.randn(1, 6, 2, 16), torch.randn(1, 6, 2, 16)
    coords = torch.randn(1, 6, 3) * 20

    here = _attention_logits(query, key, rope, coords)
    shifted = _attention_logits(query, key, rope, coords + torch.tensor([3.5, -7.0, 0.25]))
    assert torch.allclose(here, shifted, atol=1e-4)


@pytest.mark.unit
def test_identical_coordinates_give_identical_rotations():
    # What lets MuViT fuse scales: two patches at the same physical place are positioned
    # identically, whichever resolution level produced them.
    rope = AxialRotaryEmbedding(head_dim=12, spatial_rank=3)
    coords = torch.zeros(1, 2, 3)
    coords[0, 0] = torch.tensor([4.0, -2.0, 9.0])
    coords[0, 1] = torch.tensor([4.0, -2.0, 9.0])

    x = torch.randn(1, 1, 2, 12).expand(1, 2, 2, 12).contiguous()
    rotated = rope(coords)(x)
    assert torch.allclose(rotated[0, 0], rotated[0, 1], atol=1e-6)


@pytest.mark.unit
def test_different_coordinates_give_different_rotations():
    # The converse of the test above: without this, "identical coordinates agree" would also be
    # satisfied by an embedding that ignores position entirely.
    rope = AxialRotaryEmbedding(head_dim=12, spatial_rank=3)
    coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    x = torch.randn(1, 1, 2, 12).expand(1, 2, 2, 12).contiguous()

    rotated = rope(coords)(x)
    assert not torch.allclose(rotated[0, 0], rotated[0, 1], atol=1e-3)


@pytest.mark.unit
def test_each_axis_rotates_its_own_slice_only():
    # Axis separability: moving along one axis must not disturb the channels allocated to another.
    # Concatenating all axes' angles and rotating the head as a single block passes the relative
    # position test but fails this one, because it mixes axes at the halfway split.
    rope = AxialRotaryEmbedding(head_dim=12, spatial_rank=3)
    first, second, third = rope.axis_dims
    x = torch.randn(1, 1, 1, 12)

    base = rope(torch.zeros(1, 1, 3))(x)
    moved = rope(torch.tensor([[[5.0, 0.0, 0.0]]]))(x)

    assert not torch.allclose(base[..., :first], moved[..., :first], atol=1e-3)
    untouched = slice(first, first + second + third)
    assert torch.allclose(base[..., untouched], moved[..., untouched], atol=1e-6)


@pytest.mark.unit
def test_zero_coordinates_leave_the_input_alone():
    # Angle zero is the identity rotation, which makes the origin a meaningful reference point.
    rope = AxialRotaryEmbedding(head_dim=16, spatial_rank=3)
    x = torch.randn(2, 3, 4, 16)
    assert torch.allclose(rope(torch.zeros(2, 3, 3))(x), x, atol=1e-6)


@pytest.mark.unit
def test_frequencies_are_learnable():
    # MuViT gives every layer its own learnable frequencies; if they were buffers the model would
    # silently train with a fixed schedule.
    rope = AxialRotaryEmbedding(head_dim=16, spatial_rank=3)
    assert len(list(rope.parameters())) == 3
    rope(torch.randn(1, 4, 3))(torch.randn(1, 4, 2, 16)).pow(2).mean().backward()
    assert all(param.grad is not None for param in rope.inv_freqs)


@pytest.mark.unit
def test_frequencies_start_as_a_geometric_progression():
    rope = AxialRotaryEmbedding(head_dim=12, spatial_rank=3, base=100.0)
    width = rope.axis_dims[0]
    expected = 1.0 / (100.0 ** (torch.arange(0, width, 2).float() / width))
    assert torch.allclose(rope.inv_freqs[0], expected)


@pytest.mark.unit
def test_channels_beyond_the_axis_split_pass_through():
    # `AxialRotaryEmbedding` always allocates the whole head, so this is checked on `RotaryTables`
    # directly -- it is the type that defines the convention, and the paper allows the per-axis
    # allocations to sum to less than the head dimension. Unallocated channels ride along
    # unrotated rather than being dropped.
    angles = torch.randn(1, 2, 1, 2)
    tables = RotaryTables(((angles.cos(), angles.sin()),), axis_dims=(4,))

    x = torch.randn(1, 2, 3, 10)
    rotated = tables(x)
    assert rotated.shape == x.shape
    assert torch.allclose(rotated[..., 4:], x[..., 4:], atol=1e-6)
    assert not torch.allclose(rotated[..., :4], x[..., :4], atol=1e-3)


@pytest.mark.unit
def test_rejects_coordinates_of_the_wrong_rank():
    rope = AxialRotaryEmbedding(head_dim=16, spatial_rank=3)
    with pytest.raises(ValueError, match="3 components"):
        rope(torch.randn(1, 4, 2))


@pytest.mark.unit
def test_rejects_a_degenerate_base():
    with pytest.raises(ValueError, match="greater than 1"):
        AxialRotaryEmbedding(head_dim=16, spatial_rank=3, base=1.0)
