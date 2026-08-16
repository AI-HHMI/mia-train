"""Unit tests for the supervised affinity instance-segmentation algorithm."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from algorithms.affinity.targets import cc3d_available
from algorithms.affinity_seg import AffinitySegmentation
from algorithms.registry import AlgorithmRegistry
from layers.common.dense_heads import VoxelHead
from models.dinov3_vit3d import DinoVisionTransformer3D
from models.muvit import MuViT3D
from models.vit import ViT3D

CROP = 16
PATCH = 8


def _encoder() -> ViT3D:
    return ViT3D(
        img_size=(CROP, CROP, CROP), patch_size=(PATCH, PATCH, PATCH), in_channels=1,
        embed_dim=32, depth=1, num_heads=4,
    )


def _algorithm(**overrides: Any) -> AffinitySegmentation:
    kwargs: dict[str, Any] = dict(input_axes="lcxyz", decoder_hidden_dim=8, long_range=4)
    kwargs.update(overrides)
    return AffinitySegmentation(_encoder(), **kwargs)


def _batch(batch_size: int = 2, ids: int = 3) -> dict[str, torch.Tensor]:
    """A dataset-shaped batch: image (B, L, C, X, Y, Z), label (B, L, X, Y, Z)."""
    torch.manual_seed(0)
    image = torch.rand(batch_size, 1, 1, CROP, CROP, CROP)
    label = torch.randint(0, ids, (batch_size, 1, CROP, CROP, CROP))
    return {"img": image, "label": label}


# ---------------------------------------------------------------- registration


@pytest.mark.unit
def test_registered_under_its_config_name():
    assert AlgorithmRegistry.get("affinity_seg") is AffinitySegmentation


@pytest.mark.unit
def test_offsets_follow_the_configured_long_range():
    assert _algorithm(long_range=7).offsets == (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (7, 0, 0), (0, 7, 0), (0, 0, 7),
    )


# ---------------------------------------------------------------- axis order


@pytest.mark.unit
@pytest.mark.parametrize("axes", ["lczyx", "lzyx", "lcyxz"])
def test_rejects_a_spatial_order_other_than_xyz(axes):
    """The affinity channels are defined against the benchmark's x,y,z indexing.

    A z,y,x dataset would build transposed targets, train perfectly happily, and score as
    nonsense -- there is no downstream symptom, so it has to be caught here.
    """
    with pytest.raises(ValueError, match="spatial order"):
        _algorithm(input_axes=axes)


@pytest.mark.unit
@pytest.mark.parametrize("axes", ["lcxyz", "lxyz"])
def test_accepts_xyz_with_or_without_a_channel_axis(axes):
    assert _algorithm(input_axes=axes).input_axes == axes


@pytest.mark.unit
def test_rejects_input_axes_contradicting_the_dataset():
    class _Dataset:
        sample_axes = "lcxyz"

    with pytest.raises(ValueError, match="contradicts"):
        AffinitySegmentation(_encoder(), _Dataset(), input_axes="lczyx")


@pytest.mark.unit
def test_requires_an_axis_order_from_somewhere():
    with pytest.raises(ValueError, match="no axis order"):
        AffinitySegmentation(_encoder())


# ---------------------------------------------------------------- the step


@pytest.mark.unit
def test_training_step_reports_loss_and_diagnostics():
    out = _algorithm().training_step(_batch())
    assert set(out) == {
        "loss", "affinity_accuracy", "boundary_accuracy", "target_positive_rate",
        "masked_fraction",
    }
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert 0.0 <= out["affinity_accuracy"] <= 1.0
    assert 0.0 <= out["boundary_accuracy"] <= 1.0


@pytest.mark.unit
def test_validation_step_matches_training_step_in_eval():
    algorithm = _algorithm().eval()
    batch = _batch()
    with torch.no_grad():
        assert torch.allclose(
            algorithm.training_step(batch)["loss"], algorithm.validation_step(batch)["loss"]
        )


@pytest.mark.unit
def test_gradient_reaches_both_the_encoder_and_the_decoder():
    algorithm = _algorithm()
    algorithm.training_step(_batch())["loss"].backward()

    named = dict(algorithm.named_parameters())
    # The encoder is stored by the base class as `model`; `self.encoder` is the same module, so
    # torch reports it once under the first name it was registered with.
    assert any(n.startswith("model.") and p.grad is not None for n, p in named.items())
    assert any(n.startswith("decoder") and p.grad is not None for n, p in named.items())
    assert all(p.grad is not None for p in algorithm.parameters()), "every parameter trains"


@pytest.mark.unit
def test_masked_fraction_matches_the_analytic_border_fraction():
    """Every offset loses the slab it shifts in from, and nothing else."""
    algorithm = _algorithm(long_range=4, split_disconnected=False)
    out = algorithm.training_step(_batch())

    expected = (3 * (CROP - 1) + 3 * (CROP - 4)) / (6 * CROP)
    assert out["masked_fraction"].item() == pytest.approx(expected, abs=1e-6)


@pytest.mark.unit
def test_a_perfect_prediction_scores_near_zero_loss():
    """Sanity that the loss is oriented correctly: matching the target must be rewarded."""
    algorithm = _algorithm(split_disconnected=False).eval()
    batch = _batch()
    labels = algorithm._prepare_labels(batch["label"])
    target, mask = algorithm._targets(labels)

    confident = (target * 2 - 1) * 20.0  # +-20 logits, i.e. the right answer, loudly
    per_voxel = torch.nn.functional.binary_cross_entropy_with_logits(
        confident, target, reduction="none"
    )
    assert (per_voxel * mask).sum() / mask.sum() < 1e-6


@pytest.mark.unit
def test_missing_label_key_says_what_to_configure():
    batch = _batch()
    del batch["label"]
    with pytest.raises(KeyError, match="label_key"):
        _algorithm().training_step(batch)


@pytest.mark.unit
def test_rejects_a_label_crop_that_does_not_match_the_image():
    batch = _batch()
    batch["label"] = batch["label"][:, :, : CROP // 2]
    with pytest.raises(ValueError, match="co-registered"):
        _algorithm().training_step(batch)


@pytest.mark.unit
def test_rejects_a_multi_level_label_batch():
    batch = _batch()
    batch["label"] = batch["label"].repeat(1, 2, 1, 1, 1)
    with pytest.raises(ValueError, match="single-scale"):
        _algorithm().training_step(batch)


@pytest.mark.unit
def test_rejects_a_label_of_the_wrong_rank():
    batch = _batch()
    batch["label"] = batch["label"].squeeze(1)  # dropped the level axis
    with pytest.raises(ValueError, match="D batch"):
        _algorithm().training_step(batch)


# ---------------------------------------------------------------- patch_features contract


@pytest.mark.unit
def test_vit3d_patch_features_covers_the_whole_grid():
    encoder = _encoder()
    tokens, grid = encoder.patch_features(torch.randn(2, 1, CROP, CROP, CROP))
    assert grid == (2, 2, 2)
    assert tokens.shape == (2, 8, encoder.embed_dim), "one token per patch, none dropped"


@pytest.mark.unit
def test_dinov3_patch_features_drops_cls_and_storage_tokens():
    encoder = DinoVisionTransformer3D(
        img_size=CROP, patch_size=PATCH, in_chans=1, embed_dim=32, depth=1, num_heads=4,
        n_storage_tokens=3, pos_embed_rope_dtype="fp32",
    )
    tokens, grid = encoder.patch_features(torch.randn(2, 1, CROP, CROP, CROP))
    assert grid == (2, 2, 2)
    assert tokens.shape == (2, 8, 32), "patch tokens only -- CLS and registers have no grid slot"


@pytest.mark.unit
def test_multiscale_encoder_declines_to_produce_one_patch_grid():
    encoder = MuViT3D(
        levels=(1, 4), img_size=(CROP, CROP, CROP), patch_size=(PATCH, PATCH, PATCH),
        in_channels=1, embed_dim=32, depth=1, num_heads=4,
    )
    with pytest.raises(NotImplementedError, match="patch_features"):
        encoder.patch_features(torch.randn(1, 1, CROP, CROP, CROP))


@pytest.mark.unit
def test_decoder_rejects_tokens_that_do_not_fill_the_grid():
    algorithm = _algorithm()
    tokens = torch.randn(2, 7, 32)  # a 2x2x2 grid needs 8
    with pytest.raises(ValueError, match="does not fill|holds"):
        algorithm._decode(tokens, (2, 2, 2), torch.Size((CROP, CROP, CROP)))


@pytest.mark.unit
def test_logits_come_back_at_voxel_resolution_with_one_channel_per_offset():
    algorithm = _algorithm()
    tokens, grid = algorithm.encoder.patch_features(torch.randn(2, 1, CROP, CROP, CROP))
    logits = algorithm._decode(tokens, grid, torch.Size((CROP, CROP, CROP)))
    assert logits.shape == (2, 6, CROP, CROP, CROP)


# ---------------------------------------------------------------- decoder choice


@pytest.mark.unit
def test_subpixel_decoder_produces_the_same_shaped_logits():
    """Swapping the decoder must be invisible to everything downstream of the head."""
    interpolate = _algorithm().training_step(_batch())
    subpixel = _algorithm(decoder="subpixel", decoder_readout_dim=4).training_step(_batch())
    assert set(interpolate) == set(subpixel)
    assert torch.isfinite(subpixel["loss"])


@pytest.mark.unit
def test_subpixel_decoder_starts_from_the_trivial_constant():
    """Zero-initialised output: every logit identical, so a warm encoder sees no random gradient."""
    algorithm = _algorithm(decoder="subpixel", decoder_readout_dim=4)
    volumes = algorithm.encoder.prepare_input(_batch()["img"], "lcxyz")
    with torch.no_grad():
        tokens, grid = algorithm.encoder.patch_features(volumes)
        logits = algorithm._decode(tokens, grid, volumes.shape[2:])
    assert torch.equal(logits, torch.zeros_like(logits))


@pytest.mark.unit
def test_subpixel_decoder_is_the_checkpointable_region():
    algorithm = _algorithm(decoder="subpixel", decoder_readout_dim=4)
    assert algorithm.checkpointable_modules() == (algorithm.decoder_out,)


@pytest.mark.unit
def test_interpolate_stays_the_default():
    """Every checkpoint trained on this task so far carries the interpolating head."""
    assert isinstance(_algorithm().decoder_out, VoxelHead)


@pytest.mark.unit
def test_unknown_decoder_is_rejected():
    with pytest.raises(ValueError, match="decoder must be one of"):
        _algorithm(decoder="transposed")


# ---------------------------------------------------- delegating the split to the dataloader


@pytest.mark.unit
def test_no_sample_transform_when_splitting_is_off():
    assert _algorithm(split_disconnected=False).sample_transform() is None


def _reentering_labels() -> torch.Tensor:
    """(B, L, X, Y, Z) holding one id in two runs 4 voxels apart -- exactly `long_range` here.

    The gap is the point: `_algorithm` uses `long_range=4`, so the (4, 0, 0) offset relates x=0
    to x=4, which carry the same id on disk but belong to different components inside the crop.
    That one affinity channel is where splitting is visible and everything else is unchanged.
    """
    labels = torch.zeros(1, 1, CROP, 1, 1, dtype=torch.long)
    labels[0, 0, 0:2] = 5
    labels[0, 0, 4:6] = 5
    return labels


@pytest.mark.unit
def test_the_device_path_still_splits_when_nothing_was_delegated():
    """An algorithm driven without the engine must not silently produce unsplit targets.

    `sample_transform` is what hands the work to the workers, and an algorithm nobody asked for
    one is still responsible for doing it. A test or a notebook that builds this class directly
    would otherwise train against targets joining pieces the crop shows as separate.
    """
    algorithm = _algorithm(split_disconnected=True)
    assert algorithm._split_delegated is False

    target, _ = algorithm._targets(algorithm._prepare_labels(_reentering_labels()))
    long_range_x = algorithm.offsets.index((4, 0, 0))
    assert target[0, long_range_x, 0, 0, 0] == 0.0, "the two runs must not be one object"


@pytest.mark.unit
def test_an_unsplit_run_would_call_them_one_object():
    """The control for the test above: without splitting, that same affinity is positive.

    Without it, the assertion above would pass just as well against a target that is zero for an
    unrelated reason.
    """
    algorithm = _algorithm(split_disconnected=False)
    target, _ = algorithm._targets(algorithm._prepare_labels(_reentering_labels()))
    long_range_x = algorithm.offsets.index((4, 0, 0))
    assert target[0, long_range_x, 0, 0, 0] == 1.0


@pytest.mark.unit
@pytest.mark.skipif(not cc3d_available(), reason="needs the 'affinity' extra (cc3d)")
def test_delegating_stops_the_algorithm_doing_it_twice(monkeypatch: pytest.MonkeyPatch):
    """Once delegated, `_targets` must not repeat the pass -- that was the entire point.

    Patched on `algorithms.affinity_seg`, not on the module that defines it: the algorithm does
    `from .affinity.targets import relabel_connected`, which binds the name in its own namespace
    at import, so patching the definition site would leave the call site untouched and the spy
    would never fire.
    """
    algorithm = _algorithm(split_disconnected=True)
    assert algorithm.sample_transform() is not None
    assert algorithm._split_delegated is True

    def fail(*args: Any, **kwargs: Any):
        raise AssertionError("the device-side split ran even though the workers had it")

    monkeypatch.setattr("algorithms.affinity_seg.relabel_connected", fail)
    algorithm._targets(algorithm._prepare_labels(_reentering_labels()))
