"""The coordinate frame prompts and image patches share, and the rotary schedule over it.

The single property this file exists to protect: **a point prompt and the patch token containing
it must be positioned identically.** That is what lets a promptable decoder express "the click is
here" as a zero displacement rather than as a correspondence it has to learn between two unrelated
encodings. Nothing downstream can detect a violation -- a decoder positioned half a patch off
still trains, still converges, and is simply worse.
"""

from __future__ import annotations

import math

import pytest
import torch

from layers.common.rope import (
    AxialRotaryEmbedding,
    patch_grid_coords,
    voxel_coords,
)


@pytest.mark.unit
def test_a_prompt_at_a_patch_centre_gets_that_patch_token_s_coordinate():
    grid, patch, extent = (4, 5, 6), (16, 16, 16), (64, 80, 96)
    tokens = patch_grid_coords(grid, patch, extent).reshape(*grid, 3)

    for index in ((0, 0, 0), (1, 0, 3), (3, 4, 5)):
        pairs = zip(index, patch, strict=True)
        centre = torch.tensor([[float(i * p + p // 2 - 0.5) for i, p in pairs]])
        torch.testing.assert_close(voxel_coords(centre, extent)[0], tokens[index])


@pytest.mark.unit
def test_coordinates_span_the_crop_without_leaving_the_unit_frame():
    extent = (64, 64, 64)
    corners = torch.tensor([[0.0, 0.0, 0.0], [63.0, 63.0, 63.0]])
    coords = voxel_coords(corners, extent)
    # Half a voxel inside each face, which is the convention `patch_grid_coords` and DINOv3's own
    # rotary both use; a naive `2 * i / E - 1` would put voxel 0 exactly on -1 and voxel E-1 short
    # of +1, which is asymmetric.
    assert coords.min() == pytest.approx(-1 + 1 / 64)
    assert coords.max() == pytest.approx(1 - 1 / 64)


@pytest.mark.unit
def test_normalize_modes_differ_only_for_an_anisotropic_crop():
    voxels = torch.tensor([[10.0, 10.0, 10.0]])
    cubic = voxel_coords(voxels, (64, 64, 64))
    for mode in ("separate", "max", "min"):
        torch.testing.assert_close(voxel_coords(voxels, (64, 64, 64), mode), cubic)

    # A flat crop: "separate" stretches the short axis to fill [-1, 1], the others do not.
    flat = (64, 64, 16)
    assert voxel_coords(voxels, flat, "separate")[0, 2] > voxel_coords(voxels, flat, "max")[0, 2]
    assert voxel_coords(voxels, flat, "min")[0, 0] > voxel_coords(voxels, flat, "separate")[0, 0]


@pytest.mark.unit
def test_patch_centres_use_the_crop_extent_not_the_covered_extent():
    # A crop of 70 at patch 16 gives a grid of 4, covering 64 voxels and leaving a 6-voxel rim the
    # encoder never sees. Normalising by 64 would push the last token past where a prompt in the
    # rim would land; normalising by the true extent keeps one frame for both.
    covered = patch_grid_coords((4,), (16,), (64,))
    actual = patch_grid_coords((4,), (16,), (70,))
    assert actual.max() < covered.max()
    # Patch j spans continuous [j*p, (j+1)*p], so its centre is at continuous (j+0.5)*p, which
    # is voxel index (j+0.5)*p - 0.5 in the frame `voxel_coords` takes.
    centres = torch.tensor([[7.5], [23.5], [39.5], [55.5]])
    torch.testing.assert_close(actual, voxel_coords(centres, (70,)))


@pytest.mark.unit
def test_period_initialisation_spans_the_requested_wavelengths():
    rope = AxialRotaryEmbedding(
        head_dim=24, spatial_rank=3, base=None, min_period=0.02, max_period=4.0
    )
    for inv_freq in rope.inv_freqs:
        periods = 2 * math.pi / inv_freq.detach()
        assert periods[0] == pytest.approx(0.02, rel=1e-5)
        assert periods[-1] == pytest.approx(4.0, rel=1e-5)
        assert torch.all(periods.diff() > 0), "the schedule must be monotone in wavelength"


@pytest.mark.unit
def test_base_and_periods_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        AxialRotaryEmbedding(head_dim=24, spatial_rank=3, min_period=0.02, max_period=4.0)
    with pytest.raises(ValueError, match="exactly one"):
        AxialRotaryEmbedding(head_dim=24, spatial_rank=3, base=None)
    with pytest.raises(ValueError, match="min_period < max_period"):
        AxialRotaryEmbedding(head_dim=24, spatial_rank=3, base=None, min_period=4.0, max_period=1.0)


@pytest.mark.unit
def test_the_period_schedule_stays_relative_over_normalised_coordinates():
    # The defining property, checked at the scale the decoder actually works at: coordinates in
    # [-1, 1] rather than the integer indices `test_rope.py` uses. A schedule whose angles were all
    # tiny here would still pass a relativity test -- it would just be uninformative -- so the
    # spread of the logits is checked too.
    rope = AxialRotaryEmbedding(
        head_dim=24, spatial_rank=3, base=None, min_period=0.02, max_period=4.0
    )
    query, key = torch.randn(1, 6, 2, 24), torch.randn(1, 6, 2, 24)
    coords = torch.rand(1, 6, 3) * 2 - 1
    shifted = coords + torch.tensor([0.11, -0.07, 0.03])

    def logits(at: torch.Tensor) -> torch.Tensor:
        tables = rope(at)
        return torch.einsum("bnhd,bmhd->bhnm", tables(query), tables(key))

    # 1e-4 rather than tighter: at `min_period=0.02` a coordinate of 1 is an angle of 314
    # radians, where fp32 keeps about five decimal digits, so the two sides round differently
    # in the last place. That costs ~1e-7 of a coordinate unit -- a ten-thousandth of a voxel
    # at a 256-crop -- against logits whose spread is checked below.
    torch.testing.assert_close(logits(coords), logits(shifted), atol=1e-4, rtol=1e-4)
    assert logits(coords).std() > 0.1, "a schedule this coarse would carry no usable position"
