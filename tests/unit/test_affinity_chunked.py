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
from models.convnet3d import ConvNet3D
from models.dinov3_vit3d import DinoVisionTransformer3D
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


def _unet_algorithm(seed: int = 0, **overrides: Any) -> AffinitySegmentation:
    """A hierarchical encoder with the U-Net head: skips at strides 4 and 8, the image at 1."""
    torch.manual_seed(seed)
    encoder = ConvNet3D(img_size=CROP, widths=(8, 12), depths=(1, 1))
    kwargs: dict[str, Any] = dict(
        input_axes="lcxyz", decoder="unet", long_range=4, decoder_widths=(3, 4, 6),
        decoder_zero_init_output=False, split_disconnected=False,
    )
    kwargs.update(overrides)
    return AffinitySegmentation(encoder, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("chunks", [2, 3, 6])
def test_unet_chunked_metrics_match_the_undivided_decode(chunks):
    batch = _batch()
    whole = _unet_algorithm().training_step(batch)
    split = _unet_algorithm(decode_chunks=chunks).training_step(batch)
    assert set(whole) == set(split)
    for name, expected in whole.items():
        assert float(split[name]) == pytest.approx(float(expected), rel=1e-5, abs=1e-6), (
            f"{name} differs at decode_chunks={chunks}"
        )


@pytest.mark.unit
def test_unet_chunked_gradients_match_the_undivided_decode():
    """In float64, for the reason `test_chunked_gradients_match_the_undivided_decode` gives."""
    torch.set_default_dtype(torch.float64)
    try:
        batch = {
            name: value.double() if value.is_floating_point() else value
            for name, value in _batch().items()
        }
        whole, split = _unet_algorithm(), _unet_algorithm(decode_chunks=3)
        whole.training_step(batch)["loss"].backward()
        split.training_step(batch)["loss"].backward()
        reference = dict(whole.named_parameters())
        for name, parameter in split.named_parameters():
            assert parameter.grad is not None, f"{name} received no gradient when chunked"
            expected = reference[name].grad
            scale = max(float(expected.abs().max()), 1e-30)
            assert float((parameter.grad - expected).abs().max()) / scale < 1e-12, name
    finally:
        torch.set_default_dtype(torch.float32)


def _unetr_algorithm(seed: int = 0, **overrides: Any) -> AffinitySegmentation:
    """A DINOv3 ViT with the UNETR head: skips from three blocks, the image at full resolution."""
    torch.manual_seed(seed)
    encoder = DinoVisionTransformer3D(
        img_size=CROP, patch_size=PATCH, in_chans=1, embed_dim=32, depth=3, num_heads=4,
        n_storage_tokens=2, pos_embed_rope_dtype="fp32",
    )
    kwargs: dict[str, Any] = dict(
        input_axes="lcxyz", decoder="unetr", long_range=4, decoder_widths=(3, 4, 6),
        decoder_zero_init_output=False, split_disconnected=False,
    )
    kwargs.update(overrides)
    return AffinitySegmentation(encoder, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("chunks", [2, 3, 6])
def test_unetr_chunked_metrics_match_the_undivided_decode(chunks):
    batch = _batch()
    whole = _unetr_algorithm().training_step(batch)
    split = _unetr_algorithm(decode_chunks=chunks).training_step(batch)
    assert set(whole) == set(split)
    for name, expected in whole.items():
        assert float(split[name]) == pytest.approx(float(expected), rel=1e-5, abs=1e-6), (
            f"{name} differs at decode_chunks={chunks}"
        )


@pytest.mark.unit
def test_unetr_chunked_gradients_match_the_undivided_decode():
    """In float64, for the reason `test_chunked_gradients_match_the_undivided_decode` gives."""
    torch.set_default_dtype(torch.float64)
    try:
        batch = {
            name: value.double() if value.is_floating_point() else value
            for name, value in _batch().items()
        }
        whole, split = _unetr_algorithm(), _unetr_algorithm(decode_chunks=3)
        whole.training_step(batch)["loss"].backward()
        split.training_step(batch)["loss"].backward()
        reference = dict(whole.named_parameters())
        for name, parameter in split.named_parameters():
            assert parameter.grad is not None, f"{name} received no gradient when chunked"
            expected = reference[name].grad
            scale = max(float(expected.abs().max()), 1e-30)
            assert float((parameter.grad - expected).abs().max()) / scale < 1e-12, name
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.unit
@pytest.mark.parametrize(
    "build", [_algorithm, _unetr_algorithm, _unet_algorithm], ids=["subpixel", "unetr", "unet"]
)
def test_chunked_logits_match_the_undivided_decode(build):
    """Prediction calls `logits()`, which decodes in the training step's slabs when it chunks."""
    torch.manual_seed(2)
    volumes = torch.rand(2, 1, CROP, CROP, CROP)
    whole, split = build().eval(), build(decode_chunks=4).eval()
    with torch.no_grad():
        expected = whole.logits(volumes)
        assert expected.shape == (2, len(whole.offsets), CROP, CROP, CROP)
        torch.testing.assert_close(split.logits(volumes), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_unet_needs_an_encoder_with_a_pyramid():
    with pytest.raises(ValueError, match="pyramid_features"):
        _algorithm(decoder="unet")


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


@pytest.mark.unit
def test_eroded_targets_match_the_undivided_decode():
    """Erosion runs once on the whole crop: eroded per slab, every seam would stay uneroded."""
    torch.manual_seed(2)
    blocks = torch.randint(1, 6, (2, 1, CROP // 8, CROP // 8, CROP // 8))
    label = blocks.repeat_interleave(8, 2).repeat_interleave(8, 3).repeat_interleave(8, 4)
    batch = {**_batch(), "label": label}
    whole = _algorithm(label_erosion=1).training_step(batch)
    split = _algorithm(label_erosion=1, decode_chunks=3).training_step(batch)
    for name, expected in whole.items():
        assert float(split[name]) == pytest.approx(float(expected), rel=1e-5, abs=1e-6), name


@pytest.mark.unit
def test_refinement_runs_on_the_slab_plus_its_reach():
    """The halo is cropped to what the refinement convolutions reach *before* they run.

    The equivalence tests above cannot see this: cropping only after the convolutions is just as
    exact. But it hands them whole halo patches -- three times the input at one plane per slab --
    and that input is what has to fit under cuDNN's 2^31 elements, so record what they are given.
    """
    grid = CROP // PATCH
    algorithm = _algorithm(decode_chunks=grid)  # one patch plane per slab
    reach = algorithm._refine_reach()
    depths: list[int] = []
    algorithm.decoder_out.refine.register_forward_pre_hook(
        lambda _module, args: depths.append(args[0].shape[2])
    )
    algorithm.training_step(_batch())
    # The end slabs border a volume face, where an undivided decode has no context either.
    assert depths == [PATCH + reach] + [PATCH + 2 * reach] * (grid - 2) + [PATCH + reach]
