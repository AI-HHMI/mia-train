"""Local shape descriptors as the affinity algorithm's auxiliary target -- the paper's MTLSD.

What has to hold: with `lsd_sigma` unset the algorithm is the one every existing checkpoint was
trained with; with it set, the head grows by the descriptor channels *after* the affinity ones,
prediction still sees only the affinity block, the two losses compose as declared, and decoding
the volume in slabs computes exactly what decoding it whole does -- now with a target whose window
is symmetric, so the slab's labels have to reach backwards as well as forwards.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from algorithms.affinity.lsd import lsd_channels
from algorithms.affinity_seg import AffinitySegmentation
from layers.common.dense_heads import SubPixelHead, VoxelHead
from models.vit import ViT3D
from prediction.dense import DensePredictor

CROP = 16
PATCH = 8
PIXEL_SIZE = (9.0, 9.0, 20.0)  # NISB's voxel, in nanometres
SIGMA = 20.0                   # nm: (2.2, 2.2, 1.0) voxels here, a window of radius (7, 7, 3)
AFFINITY_METRICS = {
    "loss", "affinity_accuracy", "boundary_accuracy", "target_positive_rate", "masked_fraction",
}
LSD_METRICS = {
    "loss_affinity", "loss_lsd", "lsd_mse_offset", "lsd_mse_variance", "lsd_mse_pearson",
    "lsd_mse_size", "lsd_masked_fraction",
}


def _encoder(crop: int = CROP) -> ViT3D:
    return ViT3D(
        img_size=(crop, crop, crop), patch_size=(PATCH, PATCH, PATCH), in_channels=1,
        embed_dim=32, depth=1, num_heads=4,
    )


def _algorithm(crop: int = CROP, seed: int = 0, **overrides: Any) -> AffinitySegmentation:
    torch.manual_seed(seed)
    kwargs: dict[str, Any] = dict(
        input_axes="lcxyz", decoder_hidden_dim=8, long_range=4, split_disconnected=False,
        lsd_sigma=SIGMA,
    )
    kwargs.update(overrides)
    return AffinitySegmentation(_encoder(crop), **kwargs)


def _batch(
    batch_size: int = 2, ids: int = 4, crop: int = CROP, pixel_size: bool = True
) -> dict[str, torch.Tensor]:
    """Image (B, L, C, X, Y, Z), blocky labels (B, L, X, Y, Z), and miao's (B, L, 3) voxel size."""
    torch.manual_seed(1)
    coarse = torch.randint(0, ids, (batch_size, crop // 4, crop // 4, crop // 4))
    labels = coarse
    for axis in (1, 2, 3):
        labels = labels.repeat_interleave(4, dim=axis)
    batch = {
        "img": torch.rand(batch_size, 1, 1, crop, crop, crop),
        "label": labels[:, None],
    }
    if pixel_size:
        batch["pixel_size"] = torch.tensor(PIXEL_SIZE).expand(batch_size, 1, 3).clone()
    return batch


def _output_conv(algorithm: AffinitySegmentation) -> torch.nn.Conv3d:
    head = algorithm.decoder_out
    return head.out if isinstance(head, SubPixelHead) else head[-1]


# --------------------------------------------------------------------------------- off


@pytest.mark.unit
@pytest.mark.parametrize("decoder", ["interpolate", "subpixel"])
def test_off_by_default_the_algorithm_is_unchanged(decoder):
    algorithm = _algorithm(lsd_sigma=None, decoder=decoder)
    assert algorithm.lsd_channels == 0
    assert _output_conv(algorithm).out_channels == 6
    # A batch that happens to carry a voxel size changes nothing either.
    assert set(_algorithm(lsd_sigma=None).training_step(_batch())) == AFFINITY_METRICS


# --------------------------------------------------------------------------------- head


@pytest.mark.unit
@pytest.mark.parametrize("decoder", ["interpolate", "subpixel"])
def test_head_emits_the_affinities_then_the_descriptors(decoder):
    algorithm = _algorithm(decoder=decoder)
    assert algorithm.lsd_channels == lsd_channels(3) == 10
    assert _output_conv(algorithm).out_channels == 16
    assert isinstance(algorithm.decoder_out, SubPixelHead if decoder == "subpixel" else VoxelHead)

    volumes = _batch(batch_size=1)["img"][:, 0]
    tokens, grid = algorithm.encoder.patch_features(volumes)
    assert algorithm._decode(tokens, grid, volumes.shape[2:]).shape[1] == 16
    # Prediction sees the affinity block alone: six channels, the artifact's contract.
    assert algorithm.logits(volumes).shape == (1, 6, CROP, CROP, CROP)
    assert algorithm.prediction_channels == 6
    DensePredictor(algorithm)  # the dense protocol is still satisfied


# --------------------------------------------------------------------------------- loss


@pytest.mark.unit
def test_metrics_compose_the_two_losses_as_declared():
    algorithm = _algorithm(lsd_weight=0.5)
    batch = _batch()
    metrics = algorithm.training_step(batch)
    assert set(metrics) == AFFINITY_METRICS | LSD_METRICS
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics["loss"].item() == pytest.approx(
        metrics["loss_affinity"].item() + 0.5 * metrics["loss_lsd"].item(), rel=1e-6
    )
    # The default masks background out of the descriptor loss, as the reference does.
    foreground = (batch["label"] > 0).float().mean()
    assert float(metrics["lsd_masked_fraction"]) == pytest.approx(float(foreground), abs=1e-6)
    # Squared errors on sigmoid outputs against targets in [0, 1] cannot exceed one.
    for name in LSD_METRICS - {"loss_affinity", "lsd_masked_fraction"}:
        assert 0.0 <= float(metrics[name]) <= 1.0


@pytest.mark.unit
def test_validation_step_reports_the_same_metrics():
    algorithm = _algorithm().eval()
    batch = _batch()
    with torch.no_grad():
        train, val = algorithm.training_step(batch), algorithm.validation_step(batch)
    assert set(train) == set(val)
    for name in train:
        assert torch.equal(train[name], val[name])


@pytest.mark.unit
def test_background_zero_supervises_every_labelled_voxel():
    metrics = _algorithm(lsd_background="zero").training_step(_batch())
    assert float(metrics["lsd_masked_fraction"]) == pytest.approx(1.0)


@pytest.mark.unit
def test_sigma_is_converted_per_sample_from_the_batch_voxel_size():
    algorithm = _algorithm()
    batch = _batch(batch_size=2)
    batch["pixel_size"][1] = torch.tensor([8.0, 8.0, 8.0])  # a second, isotropic volume
    sigma = algorithm._sigma_voxels(batch)
    assert sigma.shape == (2, 3)
    expected = torch.tensor([SIGMA / 9, SIGMA / 9, SIGMA / 20], dtype=torch.float64)
    assert torch.allclose(sigma[0], expected)
    assert torch.allclose(sigma[1], torch.full((3,), SIGMA / 8, dtype=torch.float64))
    per_axis = _algorithm(lsd_sigma=[10.0, 20.0, 40.0])._sigma_voxels(_batch(batch_size=1))
    assert torch.allclose(per_axis[0], torch.tensor([10 / 9, 20 / 9, 2.0], dtype=torch.float64))


@pytest.mark.unit
def test_gradient_reaches_the_descriptor_rows_only_through_their_loss():
    """The last convolution's rows 6:16 are the descriptors'. They train through `loss_lsd` and
    through nothing else, so at `lsd_weight = 0` they receive no gradient while the affinity rows
    still do -- which is also what makes `lsd_weight` an honest ablation knob."""
    for weight, expect_gradient in ((1.0, True), (0.0, False)):
        algorithm = _algorithm(lsd_weight=weight)
        algorithm.training_step(_batch())["loss"].backward()
        grad = _output_conv(algorithm).weight.grad
        assert grad is not None
        assert bool(grad[:6].abs().sum() > 0)
        assert bool(grad[6:].abs().sum() > 0) is expect_gradient


@pytest.mark.unit
def test_descriptor_loss_gradient_is_the_masked_mse_on_a_sigmoid():
    """`d loss_lsd / d logit = 2 (sigmoid(z) - t) sigmoid'(z) mask / (masked voxels * channels)`:
    the reference's loss, checked against autograd so the plumbing between the head's channel
    block, the target and the mask is pinned by the number rather than by a convergence curve.
    (The curve is `tests/sanity/test_lsd_overfit.py`.)"""
    algorithm = _algorithm()
    batch = _batch(batch_size=1)
    labels = algorithm._prepare_labels(batch["label"])
    target, mask = algorithm._lsd_targets(labels, algorithm._sigma_voxels(batch))
    logits = torch.randn(1, 10, CROP, CROP, CROP, requires_grad=True)

    loss = algorithm._lsd_metrics(algorithm._lsd_sums(logits, target, mask))["loss_lsd"]
    loss.backward()

    probability = torch.sigmoid(logits.detach())
    expected = (
        2 * (probability - target) * probability * (1 - probability) * mask
        / (mask.sum() * 10)
    )
    assert logits.grad is not None
    assert torch.allclose(logits.grad, expected, atol=1e-8)
    # The mean squared error itself, over masked voxels and all channels.
    manual = (((probability - target) ** 2) * mask).sum() / (mask.sum() * 10)
    assert loss.item() == pytest.approx(manual.item(), rel=1e-6)


# --------------------------------------------------------------------------------- contracts


@pytest.mark.unit
def test_a_batch_without_a_voxel_size_is_a_clear_error():
    with pytest.raises(KeyError, match="pixel_size"):
        _algorithm().training_step(_batch(pixel_size=False))
    batch = _batch()
    batch["pixel_size"] = batch["pixel_size"][:, 0]  # (B, 3): the level axis dropped
    with pytest.raises(ValueError, match="levels"):
        _algorithm().training_step(batch)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(lsd_sigma=-1.0), "positive"),
        (dict(lsd_sigma=[10.0, 10.0]), "positive"),
        (dict(lsd_weight=-0.1), "non-negative"),
        (dict(lsd_downsample=0), "at least 1"),
        (dict(lsd_background="mask"), "lsd_background"),
    ],
)
def test_rejects_bad_descriptor_settings(overrides, match):
    with pytest.raises(ValueError, match=match):
        _algorithm(**overrides)


# --------------------------------------------------------------------------------- slabs

SLAB_CROP = 48


def _slab_algorithm(seed: int = 0, **overrides: Any) -> AffinitySegmentation:
    kwargs: dict[str, Any] = dict(
        decoder="subpixel", decoder_readout_dim=4, decoder_refine_depth=2,
        decoder_zero_init_output=False,
    )
    kwargs.update(overrides)
    return _algorithm(crop=SLAB_CROP, seed=seed, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("downsample", [1, 2])
@pytest.mark.parametrize("chunks", [2, 3, 4])
def test_chunked_metrics_match_the_undivided_decode(chunks, downsample):
    """The descriptor window is symmetric, so a slab's labels must reach *backwards* too; a halo
    that only reaches forward, as the affinities alone needed, gets every seam wrong quietly."""
    batch = _batch(crop=SLAB_CROP)
    whole = _slab_algorithm(lsd_downsample=downsample).training_step(batch)
    split = _slab_algorithm(lsd_downsample=downsample, decode_chunks=chunks).training_step(batch)
    assert set(whole) == set(split) == AFFINITY_METRICS | LSD_METRICS
    for name, expected in whole.items():
        assert float(split[name]) == pytest.approx(float(expected), rel=1e-5, abs=1e-6), name


@pytest.mark.unit
def test_chunked_gradients_match_the_undivided_decode():
    """In float64, as `test_affinity_chunked` does, so rounding cannot hide a halo error. The
    descriptor target itself is float32 whatever the default dtype -- it is built from a one-hot,
    where float32 is exact -- so the bound is its rounding, not float64's."""
    torch.set_default_dtype(torch.float64)
    try:
        batch = {
            name: value.to(torch.float64) if value.is_floating_point() else value
            for name, value in _batch(crop=SLAB_CROP).items()
        }
        whole, split = _slab_algorithm().double(), _slab_algorithm(decode_chunks=3).double()
        whole.training_step(batch)["loss"].backward()
        split.training_step(batch)["loss"].backward()
        reference = dict(whole.named_parameters())
        for name, parameter in split.named_parameters():
            assert parameter.grad is not None, name
            expected = reference[name].grad
            scale = max(float(expected.abs().max()), 1e-30)
            assert float((parameter.grad - expected).abs().max()) / scale < 1e-6, name
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.unit
def test_chunking_refuses_a_downsample_that_does_not_divide_the_patch():
    with pytest.raises(ValueError, match="must divide the patch size"):
        _slab_algorithm(lsd_downsample=3, decode_chunks=2).training_step(_batch(crop=SLAB_CROP))
