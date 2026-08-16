"""Unit tests for the dense prediction heads: interpolating and sub-pixel."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from layers.common.dense_heads import SubPixelHead, VoxelHead

IN_DIM = 8
PATCH = (4, 4, 4)
GRID = (3, 2, 5)
OUT = 6


def _head(**overrides) -> SubPixelHead:
    kwargs = dict(in_dim=IN_DIM, patch_size=PATCH, out_channels=OUT, hidden=16, readout=4)
    kwargs.update(overrides)
    torch.manual_seed(0)
    return SubPixelHead(**kwargs)  # type: ignore[arg-type]


def _size() -> tuple[int, ...]:
    return tuple(g * p for g, p in zip(GRID, PATCH, strict=True))


@pytest.mark.unit
def test_expands_the_grid_by_the_patch_size() -> None:
    head = _head()
    out = head(torch.randn(2, IN_DIM, *GRID), _size())
    assert out.shape == (2, OUT, *_size())


@pytest.mark.unit
def test_anisotropic_patches_expand_per_axis() -> None:
    """A patch size that differs per axis must not be collapsed to a single scale factor."""
    patch = (2, 4, 8)
    head = _head(patch_size=patch)
    size = tuple(g * p for g, p in zip(GRID, patch, strict=True))
    out = head(torch.randn(1, IN_DIM, *GRID), size)
    assert out.shape == (1, OUT, *size)


@pytest.mark.unit
def test_a_crop_that_is_not_whole_patches_is_rejected() -> None:
    head = _head()
    wrong = tuple(s + 1 for s in _size())
    with pytest.raises(ValueError, match="whole number of patches"):
        head(torch.randn(1, IN_DIM, *GRID), wrong)


@pytest.mark.unit
def test_zero_initialised_output_starts_constant() -> None:
    """A warm encoder must not be hit with gradients from a random dense head."""
    head = _head()
    out = head(torch.randn(2, IN_DIM, *GRID), _size())
    assert torch.equal(out, torch.zeros_like(out))

    head.set_output_bias(1.6)
    out = head(torch.randn(2, IN_DIM, *GRID), _size())
    assert torch.allclose(out, torch.full_like(out, 1.6))


@pytest.mark.unit
def test_each_token_writes_to_its_own_block_at_the_right_place() -> None:
    """The one test that catches a transposed fold.

    A permutation that swaps two spatial axes produces a head that trains perfectly well and a
    segmentation that is silently wrong -- the same class of error as feeding this task labels in
    z,y,x order. So drive a single token and assert the response lands at that token's block, on
    a deliberately non-cubic grid where an x/z swap cannot alias into a valid index.
    """
    head = _head(refine_depth=1)
    # Undo the zero-init: with it the head answers zero everywhere and proves nothing.
    torch.nn.init.normal_(head.out.weight, std=1.0)

    for token in [(0, 0, 0), (2, 1, 4), (1, 0, 3)]:
        x = torch.zeros(1, IN_DIM, *GRID)
        x[(0, slice(None), *token)] = 1.0
        # The projection's bias reaches every position, so the difference against an all-zero
        # input isolates what this one token contributed.
        response = head(x, _size()) - head(torch.zeros_like(x), _size())

        energy = response.abs().sum(dim=1)[0]
        block = tuple(slice(t * p, (t + 1) * p) for t, p in zip(token, PATCH, strict=True))
        inside = energy[block].sum()
        assert inside > 0, f"token {token} produced no response in its own block"

        # `refine_depth=1` is one 3x3x3 convolution, so energy may bleed one voxel past the face
        # but cannot reach a non-adjacent block.
        outside = energy.sum() - inside
        reachable = torch.zeros_like(energy, dtype=torch.bool)
        halo = tuple(
            slice(max(t * p - 1, 0), (t + 1) * p + 1) for t, p in zip(token, PATCH, strict=True)
        )
        reachable[halo] = True
        assert energy[~reachable].sum() == 0, f"token {token} wrote outside its block and halo"
        assert inside > outside, f"token {token} put more energy in the halo than in its block"


@pytest.mark.unit
def test_blocks_are_decoded_independently() -> None:
    """Without refinement the head is exactly block-diagonal: one token cannot reach another."""
    head = _head(refine_depth=1)
    head.refine = torch.nn.Identity()  # type: ignore[assignment]
    torch.nn.init.normal_(head.out.weight, std=1.0)

    x = torch.zeros(1, IN_DIM, *GRID)
    x[0, :, 1, 1, 1] = 1.0
    response = head(x, _size()) - head(torch.zeros_like(x), _size())

    energy = response.detach().abs().sum(dim=1)[0]
    block = tuple(slice(p, 2 * p) for p in PATCH)
    assert energy[block].sum().item() > 0
    assert energy.sum().item() == pytest.approx(energy[block].sum().item(), rel=1e-5)


@pytest.mark.unit
def test_refinement_must_exist() -> None:
    with pytest.raises(ValueError, match="refine_depth"):
        _head(refine_depth=0)


@pytest.mark.unit
@pytest.mark.parametrize("patch_size", [(4,), (4, 4, 4, 4)])
def test_rejects_a_rank_with_no_convolution(patch_size) -> None:
    """2 and 3 are supported because torch has Conv2d/Conv3d; 1 and 4 have no counterpart."""
    with pytest.raises(ValueError, match="2 or 3 entries"):
        _head(patch_size=patch_size)


# ---------------------------------------------------------------- shared interface


@pytest.mark.unit
def test_both_heads_take_the_same_call():
    """`forward(x, size)` on both is what lets an algorithm offer the choice as configuration."""
    x = torch.randn(1, IN_DIM, *GRID)
    size = _size()
    interpolating = VoxelHead(
        torch.nn.Conv3d(IN_DIM, OUT, kernel_size=3, padding=1), mode="trilinear"
    )
    assert interpolating(x, size).shape == _head()(x, size).shape


@pytest.mark.unit
def test_voxel_head_interpolates_to_any_size():
    """The counterpart property to the sub-pixel head's divisibility requirement."""
    head = VoxelHead(torch.nn.Conv3d(IN_DIM, OUT, kernel_size=1), mode="trilinear")
    odd = (13, 7, 21)
    assert head(torch.randn(1, IN_DIM, *GRID), odd).shape == (1, OUT, *odd)


@pytest.mark.unit
def test_subpixel_head_serves_two_dimensional_data():
    """Rank follows `patch_size`, so the 2D semantic-segmentation path is reachable too."""
    head = SubPixelHead(IN_DIM, (4, 4), OUT, hidden=16, readout=4)
    out = head(torch.randn(2, IN_DIM, 3, 5), (12, 20))
    assert out.shape == (2, OUT, 12, 20)


@pytest.mark.unit
def test_voxel_head_uses_the_mode_it_was_built_with():
    head = VoxelHead(torch.nn.Conv2d(IN_DIM, OUT, kernel_size=1), mode="bilinear")
    assert head(torch.randn(1, IN_DIM, 3, 5), (12, 20)).shape == (1, OUT, 12, 20)


@pytest.mark.unit
def test_output_can_start_non_zero_so_a_cold_encoder_gets_gradient() -> None:
    """The zero init closes the head, which stalls an encoder that still has to learn.

    With `out.weight` at zero the gradient reaching everything upstream is zero as well, so a
    cold encoder trains on nothing until the head opens. Measured on real runs as ~1-2k wasted
    steps, so the choice has to be available per run.
    """
    x = torch.randn(1, IN_DIM, *GRID, requires_grad=False)

    closed = _head(refine_depth=1)                      # zero_init_output defaults to True
    closed(x, _size()).sum().backward()
    assert closed.project.weight.grad is not None
    assert torch.equal(closed.project.weight.grad, torch.zeros_like(closed.project.weight.grad)), (
        "with a zeroed output convolution nothing upstream should receive gradient"
    )

    open_head = _head(refine_depth=1, zero_init_output=False)
    open_head(x, _size()).sum().backward()
    assert open_head.project.weight.grad.abs().sum() > 0, (
        "with zero_init_output=False the encoder-facing projection must receive gradient on the "
        "first step"
    )


# ------------------------------------------------- the sub-pixel expansion, as a matmul


@pytest.mark.unit
@pytest.mark.parametrize("patch_size", [(4, 4, 4), (2, 4, 8), (8, 8), (3, 5)])
def test_matmul_expansion_equals_the_transposed_convolution(patch_size):
    """`_expand_tokens` must agree with the `ConvTranspose3d` it replaces, elementwise.

    This is the test the class docstring asks for. The expansion was rewritten from a transposed
    convolution to a per-token matmul plus a fold because cuDNN evaluates the convolution form at
    ~0.3% of peak at kernel == stride == 16, and the fold is a permutation -- one that swapped two
    spatial axes would train perfectly well and segment silently wrong. Comparing against the
    convolution itself is what makes that failure loud.

    Anisotropic and non-power-of-two patches are included deliberately: a transposed axis is
    invisible when every axis has the same extent, which is exactly the shape the affinity configs
    use.
    """
    rank = len(patch_size)
    grid = (3, 2, 4)[:rank]
    head = SubPixelHead(in_dim=6, patch_size=patch_size, out_channels=2, hidden=5, readout=3)

    # Random rather than the initialised values: the default init is small and symmetric enough
    # that a wrong permutation could pass on it.
    torch.manual_seed(0)
    with torch.no_grad():
        head.expand.weight.normal_()
        head.expand.bias.normal_()
    head = head.double()
    x = torch.randn(2, 5, *grid, dtype=torch.float64)

    assert torch.allclose(head._expand_tokens(x), head.expand(x), rtol=1e-10, atol=1e-12)


@pytest.mark.unit
def test_matmul_expansion_places_each_token_in_its_own_block():
    """A single non-zero token must light up exactly its own block and nothing else.

    The elementwise test above would also pass if both implementations shared a wrong convention.
    This pins the convention itself against the definition: with kernel == stride, input position
    (i, j, k) owns output block [i*p : (i+1)*p, ...] and no other voxel.
    """
    patch = (2, 3, 4)
    head = SubPixelHead(in_dim=4, patch_size=patch, out_channels=2, hidden=4, readout=1)
    with torch.no_grad():
        head.expand.weight.fill_(1.0)
        head.expand.bias.zero_()

    x = torch.zeros(1, 4, 2, 2, 2)
    x[0, :, 1, 0, 1] = 1.0  # one token, all hidden channels
    out = head._expand_tokens(x)

    block = out[0, 0, 1 * patch[0]:2 * patch[0], 0:patch[1], 1 * patch[2]:2 * patch[2]]
    assert torch.all(block != 0), "the owning block must be written"
    total = (out != 0).sum()
    assert int(total) == block.numel(), "no voxel outside the owning block may be touched"


@pytest.mark.unit
def test_expansion_parameters_keep_their_checkpoint_names_and_shapes():
    """The rewrite must not move a single parameter, or every trained sub-pixel head is stranded.

    `self.expand` stays an `nn.ConvTranspose3d` precisely so this holds; only the forward changed.
    """
    head = SubPixelHead(in_dim=8, patch_size=(4, 4, 4), out_channels=3, hidden=6, readout=2)
    state = head.state_dict()
    assert state["expand.weight"].shape == (6, 2, 4, 4, 4)  # (hidden, readout, *patch)
    assert state["expand.bias"].shape == (2,)


@pytest.mark.unit
def test_an_overlapping_expansion_is_refused_rather_than_silently_wrong():
    """The matmul is only the transposed convolution while the blocks are disjoint.

    Guards the one change the class docstring proposes -- an overlapping `kernel_size =
    2 * patch_size` to widen support across seams. Under it the matmul would drop the overlap and
    compute a different function, and the equivalence test would miss it, since that test builds
    the head through the constructor being patched.
    """
    import torch.nn as nn

    from layers.common import dense_heads

    class _Overlapping(nn.ConvTranspose3d):
        """The docstring's proposal: kernel twice the patch, stride still the patch."""

        def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple[int, ...], stride: Any):
            widened = (2 * kernel_size[0], 2 * kernel_size[1], 2 * kernel_size[2])
            super().__init__(in_ch, out_ch, kernel_size=widened, stride=stride)

    dense_heads.CONV_TRANSPOSE[3] = _Overlapping
    try:
        with pytest.raises(ValueError, match="disjoint"):
            SubPixelHead(in_dim=4, patch_size=(4, 4, 4), out_channels=2, hidden=4, readout=2)
    finally:
        dense_heads.CONV_TRANSPOSE[3] = nn.ConvTranspose3d
