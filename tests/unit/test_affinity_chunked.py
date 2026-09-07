"""Decoding the volume in slabs must compute exactly what decoding it whole computes.

`decode_chunks` exists because everything downstream of the patch grid is proportional to *voxels*
-- the crop cubed -- and on a 7B encoder at a 512-cube that stage was 72% of the step's time and
most of its memory (`experiments/b300_capability_run`). Splitting it is only worth anything if it
is not also a change of objective, and the ways it could silently become one are all quiet:

  - a slab decoded without halo has the wrong values within `refine_depth` voxels of each seam,
  - a slab scored without reaching `long_range` past its end gets the wrong affinity target there,
  - and sums combined as means rather than as ratios of sums weight uneven slabs wrongly.

None of the three changes a shape, and all three move the loss by a few percent -- which is
indistinguishable from a learning-rate change on a curve. So the check is equality against the
undivided path, not finiteness.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from algorithms.affinity_seg import AffinitySegmentation
from models.vit import ViT3D

CROP = 48
PATCH = 8


def _algorithm(seed: int = 0, **overrides: Any) -> AffinitySegmentation:
    torch.manual_seed(seed)
    encoder = ViT3D(
        img_size=(CROP, CROP, CROP), patch_size=(PATCH, PATCH, PATCH), in_channels=1,
        embed_dim=32, depth=1, num_heads=4,
    )
    kwargs: dict[str, Any] = dict(
        input_axes="lcxyz", decoder="subpixel", long_range=4,
        decoder_hidden_dim=8, decoder_readout_dim=4, decoder_refine_depth=2,
        # Off, so the head starts as something other than a constant predictor -- a zero-init
        # output would make every arm agree for the wrong reason.
        decoder_zero_init_output=False,
        split_disconnected=False,
    )
    kwargs.update(overrides)
    return AffinitySegmentation(encoder, **kwargs)


def _batch(batch_size: int = 2, ids: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "img": torch.rand(batch_size, 1, 1, CROP, CROP, CROP),
        "label": torch.randint(0, ids, (batch_size, 1, CROP, CROP, CROP)),
    }


@pytest.mark.unit
@pytest.mark.parametrize("chunks", [2, 3, 4])
def test_chunked_metrics_match_the_undivided_decode(chunks):
    batch = _batch()
    whole = _algorithm().training_step(batch)
    split = _algorithm(decode_chunks=chunks).training_step(batch)

    assert set(whole) == set(split)
    for name, expected in whole.items():
        assert float(split[name]) == pytest.approx(float(expected), rel=1e-5, abs=1e-6), (
            f"{name} differs at decode_chunks={chunks}"
        )


def _gradient_disagreement(dtype: torch.dtype) -> float:
    """Largest relative gradient difference between the chunked and undivided decode."""
    torch.set_default_dtype(dtype)
    try:
        batch = {
            name: value.to(dtype) if value.is_floating_point() else value
            for name, value in _batch().items()
        }
        whole, split = _algorithm().to(dtype), _algorithm(decode_chunks=3).to(dtype)
        whole.training_step(batch)["loss"].backward()
        split.training_step(batch)["loss"].backward()

        reference = dict(whole.named_parameters())
        worst = 0.0
        for name, parameter in split.named_parameters():
            assert parameter.grad is not None, f"{name} received no gradient when chunked"
            expected = reference[name].grad
            difference = float((parameter.grad - expected).abs().max())
            worst = max(worst, difference / max(float(expected.abs().max()), 1e-30))
        return worst
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.unit
def test_chunked_gradients_match_the_undivided_decode():
    """Equal losses are not enough: a wrong halo can cancel in a mean and not in a gradient.

    Checked in **float64**, and that is the point of the test rather than an affectation. In
    float32 the two disagree by ~6e-4 relative, which is neither obviously noise nor obviously a
    bug -- the loss is a ratio of sums over ~10^6 values, and summing three partial sums does not
    associate the same way as summing the lot, so a naive accumulation drifts by roughly
    sqrt(N) * eps. Widening the tolerance until float32 passes would have accepted a genuine halo
    error of the same size. At double precision the difference falls to ~3e-15, i.e. to nothing,
    which says the decomposition is exact and the float32 gap is arithmetic rather than algebra.
    """
    assert _gradient_disagreement(torch.float64) < 1e-12

    # And the float32 gap is bounded, so a regression that is *not* rounding still fails here.
    assert _gradient_disagreement(torch.float32) < 5e-3


@pytest.mark.unit
def test_uneven_chunks_are_allowed():
    """The grid need not divide by the chunk count; remainders go to the earliest slabs."""
    grid = CROP // PATCH
    algorithm = _algorithm(decode_chunks=4)
    spans = algorithm._chunk_spans(grid)
    assert [stop - start for start, stop in spans] == [2, 2, 1, 1]
    assert spans[0][0] == 0 and spans[-1][1] == grid

    # 5 chunks over a 6-wide grid, and more chunks than the grid has planes, both stay covering.
    five = _algorithm(decode_chunks=5)._chunk_spans(grid)
    assert sum(stop - start for start, stop in five) == grid
    assert len(_algorithm(decode_chunks=99)._chunk_spans(grid)) == grid


@pytest.mark.unit
def test_chunking_refuses_the_interpolating_head():
    # Its `F.interpolate` derives the scale from the sizes it is handed, so a slab plus halo
    # samples at a different rate and the seams are wrong with no shape to catch it.
    with pytest.raises(ValueError, match="subpixel"):
        _algorithm(decoder="interpolate", decode_chunks=2)


# ------------------------------------------------------------------ label dtype


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_signed_integer_labels_are_not_widened(dtype):
    """The label volume is the largest tensor in a large-crop step; widening it doubles it.

    `.long()` on a narrower label cannot be a view, so it allocates a second copy while the
    caller's batch still holds the first -- 45.8 GiB of a ~265 GiB step at a 1600-cube. Nothing
    downstream reads the width, so a signed integer passes through untouched.
    """
    algorithm = _algorithm()
    batch = _batch()
    labels = algorithm._prepare_labels(batch["label"].to(dtype))
    assert labels.dtype == dtype


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_other_label_dtypes_are_still_widened(dtype):
    """A float cannot be trusted to have kept its ids distinct, and an unsigned type cannot hold
    `ignore_index`; for both, the conversion carries information and is kept."""
    algorithm = _algorithm()
    labels = algorithm._prepare_labels(_batch()["label"].to(dtype))
    assert labels.dtype == torch.int64


@pytest.mark.unit
def test_label_dtype_does_not_change_what_is_computed():
    """The point of the change: narrower labels are cheaper and mean exactly the same thing."""
    batch = _batch()
    wide = _algorithm().training_step({**batch, "label": batch["label"].to(torch.int64)})
    narrow = _algorithm().training_step({**batch, "label": batch["label"].to(torch.int32)})
    for name, expected in wide.items():
        assert float(narrow[name]) == pytest.approx(float(expected), rel=1e-6), name
