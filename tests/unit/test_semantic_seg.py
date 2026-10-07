"""`semantic_seg`: its heads, its ignored voxels, and slabs computing exactly the undivided decode.

The slab half mirrors `test_affinity_chunked.py`, for the reason given there: decoding the volume in
slabs is only worth having if it is not also a change of objective, and the ways it could become one
-- a halo shorter than a head's reach, a slab upsampled at another rate than the volume, means taken
of means -- change no shape and move the loss by a little. So every check is equality against the
undivided decode, not finiteness.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from algorithms.semantic_seg import SemanticSegmentation
from models.dinov3_vit3d import DinoVisionTransformer3D
from models.vit import ViT3D

CROP = 48
PATCH = 8
CLASSES = 4
IGNORE = 255
EMBED = 32
DECODERS = ("interpolate", "linear", "subpixel", "unetr")


def _encoder(kind: str, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    if kind == "dinov3":
        return DinoVisionTransformer3D(
            img_size=CROP, patch_size=PATCH, in_chans=1, embed_dim=EMBED, depth=3, num_heads=4,
            n_storage_tokens=2, pos_embed_rope_dtype="fp32",
        )
    return ViT3D(
        img_size=(CROP, CROP, CROP), patch_size=(PATCH, PATCH, PATCH), in_channels=1,
        embed_dim=EMBED, depth=1, num_heads=4,
    )


def _algorithm(
    decoder: str = "interpolate", seed: int = 0, **overrides: Any
) -> SemanticSegmentation:
    """The UNETR head reads intermediate layers, which the DINOv3 model provides and `ViT3D` not."""
    encoder = overrides.pop("encoder", "dinov3" if decoder == "unetr" else "vit")
    kwargs: dict[str, Any] = dict(
        input_axes="lcxyz", num_classes=CLASSES, decoder=decoder, ignore_index=IGNORE,
        decoder_hidden_dim=8, decoder_readout_dim=4, decoder_widths=(3, 4, 6),
        # Off, so every head starts as something other than a constant predictor -- a zero-init
        # output would make the chunked and undivided arms agree for the wrong reason.
        decoder_zero_init_output=False,
    )
    kwargs.update(overrides)
    return SemanticSegmentation(_encoder(encoder, seed), **kwargs)


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Images, and labels ignored outside a box -- as CMCT's are outside their annotated region.

    The box is aligned to neither the patches nor the slabs, so slab faces cut through supervised
    and ignored voxels alike.
    """
    torch.manual_seed(1)
    labels = torch.randint(0, CLASSES, (batch_size, 1, CROP, CROP, CROP))
    outside = torch.ones_like(labels, dtype=torch.bool)
    outside[..., 10:37, 6:30, 12:44] = False
    labels[outside] = IGNORE
    return {"img": torch.rand(batch_size, 1, 1, CROP, CROP, CROP), "label": labels}


def _value(tensor: torch.Tensor) -> float:
    return float(tensor.detach())


# ---------------------------------------------------------------- the heads


@pytest.mark.unit
@pytest.mark.parametrize("decoder", DECODERS)
def test_scores_come_back_at_voxel_resolution_with_one_channel_per_class(decoder):
    algorithm = _algorithm(decoder).eval()
    with torch.no_grad():
        scores = algorithm.logits(torch.rand(2, 1, CROP, CROP, CROP))
    assert scores.shape == (2, CLASSES, CROP, CROP, CROP)


@pytest.mark.unit
def test_the_default_head_keeps_its_parameter_names():
    """Checkpoints written before the head was shared with `affinity_seg` still load into it."""
    # The encoder is registered twice, as the algorithm's `model` and as its `encoder`.
    names = {
        name for name in _algorithm().state_dict() if not name.startswith(("encoder.", "model."))
    }
    assert names == {
        "decoder.0.weight", "decoder.0.bias",
        "decoder_out.0.weight", "decoder_out.0.bias",
        "decoder_out.2.weight", "decoder_out.2.bias",
    }


@pytest.mark.unit
def test_the_linear_head_learns_nothing_at_voxel_resolution():
    algorithm = _algorithm("linear")
    assert not list(algorithm.decoder_out.parameters())
    assert tuple(algorithm.decoder.weight.shape) == (CLASSES, EMBED, 1, 1, 1)


# ---------------------------------------------------------------- what is supervised


@pytest.mark.unit
def test_ignored_voxels_leave_the_loss_and_the_metrics():
    algorithm = _algorithm("linear")
    batch = _batch()
    reported = algorithm.training_step(batch)

    labels = algorithm._prepare_labels(batch["label"])
    scores = algorithm.logits(algorithm.encoder.prepare_input(batch["img"], "lcxyz"))
    expected = F.cross_entropy(scores, labels, ignore_index=IGNORE)
    assert _value(reported["loss"]) == pytest.approx(_value(expected), rel=1e-5)

    valid = labels != IGNORE
    accuracy = (scores.argmax(dim=1)[valid] == labels[valid]).double().mean()
    assert _value(reported["pixel_accuracy"]) == pytest.approx(float(accuracy), rel=1e-6)


@pytest.mark.unit
def test_a_batch_with_nothing_supervised_has_a_zero_loss_and_finite_gradients():
    """0/0 would be NaN, and a NaN loss would turn every parameter it reaches into NaN."""
    algorithm = _algorithm("subpixel")
    batch = _batch()
    batch["label"][:] = IGNORE
    reported = algorithm.training_step(batch)

    assert _value(reported["loss"]) == 0.0
    reported["loss"].backward()
    for name, parameter in algorithm.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert torch.isnan(reported["pixel_accuracy"]) and torch.isnan(reported["mean_iou"])


@pytest.mark.unit
def test_class_weights_normalise_as_cross_entropy_does_and_survive_chunking():
    weights = (0.5, 2.0, 1.0, 3.0)
    batch = _batch()
    whole = _algorithm("subpixel", class_weights=weights)
    split = _algorithm("subpixel", class_weights=weights, decode_chunks=3)

    labels = whole._prepare_labels(batch["label"])
    scores = whole.logits(whole.encoder.prepare_input(batch["img"], "lcxyz"))
    expected = F.cross_entropy(scores, labels, weight=torch.tensor(weights), ignore_index=IGNORE)
    whole_loss = _value(whole.training_step(batch)["loss"])
    assert whole_loss == pytest.approx(_value(expected), rel=1e-5)
    assert _value(split.training_step(batch)["loss"]) == pytest.approx(whole_loss, rel=1e-5)


def _reference_dice_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """muvit2-experiments' batch-pooled soft Dice (`MaskedCCEDiceLoss`, `dice_batch`), spelt out."""
    probs = scores.double().softmax(dim=1)
    valid = labels != IGNORE
    per_class = []
    for c in range(1, CLASSES):
        member = labels == c
        overlap = (probs[:, c] * member).sum()
        mass = (probs[:, c] * valid).sum()
        per_class.append(1 - (2 * overlap + 1e-6) / (mass + member.sum() + 1e-6))
    return torch.stack(per_class).mean()


@pytest.mark.unit
def test_without_a_dice_weight_the_loss_is_cross_entropy_alone():
    reported = _algorithm("linear").training_step(_batch())
    assert "dice_loss" not in reported and "cross_entropy" not in reported


@pytest.mark.unit
@pytest.mark.parametrize("weight", [1.0, 0.5])
def test_the_dice_term_is_soft_dice_over_the_foreground_pooled_over_the_batch(weight):
    """Two crops in one batch, so the pooling across samples is part of what is checked."""
    algorithm = _algorithm("subpixel", dice_weight=weight)
    batch = _batch()
    reported = algorithm.training_step(batch)

    labels = algorithm._prepare_labels(batch["label"])
    scores = algorithm.logits(algorithm.encoder.prepare_input(batch["img"], "lcxyz"))
    cross_entropy = _value(F.cross_entropy(scores, labels, ignore_index=IGNORE))
    dice = float(_reference_dice_loss(scores, labels))
    assert _value(reported["cross_entropy"]) == pytest.approx(cross_entropy, rel=1e-5)
    assert _value(reported["dice_loss"]) == pytest.approx(dice, rel=1e-5)
    assert _value(reported["loss"]) == pytest.approx(cross_entropy + weight * dice, rel=1e-5)


@pytest.mark.unit
def test_a_batch_with_nothing_supervised_has_a_zero_dice_term():
    """No class labelled and none predicted on a supervised voxel: every ratio is a perfect 1."""
    algorithm = _algorithm("subpixel", dice_weight=1.0)
    batch = _batch()
    batch["label"][:] = IGNORE
    reported = algorithm.training_step(batch)

    assert _value(reported["dice_loss"]) == 0.0 and _value(reported["loss"]) == 0.0
    reported["loss"].backward()
    for name, parameter in algorithm.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.unit
def test_dice_weight_must_not_be_negative():
    with pytest.raises(ValueError, match="dice_weight"):
        _algorithm("linear", dice_weight=-1.0)


@pytest.mark.unit
def test_checkpointing_the_head_changes_nothing_but_memory():
    batch = _batch()
    plain = _algorithm().training_step(batch)
    recomputed = _algorithm(checkpoint_decoder=True).training_step(batch)
    assert _value(recomputed["loss"]) == pytest.approx(_value(plain["loss"]), rel=1e-6)
    recomputed["loss"].backward()


# ---------------------------------------------------------------- slabs


@pytest.mark.unit
@pytest.mark.parametrize("chunks", [2, 4])
@pytest.mark.parametrize("decoder", DECODERS)
def test_chunked_metrics_match_the_undivided_decode(decoder, chunks):
    batch = _batch()
    whole = _algorithm(decoder).training_step(batch)
    split = _algorithm(decoder, decode_chunks=chunks).training_step(batch)

    assert set(whole) == set(split)
    for name, expected in whole.items():
        assert _value(split[name]) == pytest.approx(_value(expected), rel=1e-5, abs=1e-6), (
            f"{name} differs at decode_chunks={chunks}"
        )


@pytest.mark.unit
@pytest.mark.parametrize("decoder", DECODERS)
def test_chunked_logits_match_the_undivided_decode(decoder):
    """Prediction calls `logits()`, which decodes in the training step's slabs when it chunks."""
    torch.manual_seed(2)
    volumes = torch.rand(2, 1, CROP, CROP, CROP)
    whole, split = _algorithm(decoder).eval(), _algorithm(decoder, decode_chunks=4).eval()
    with torch.no_grad():
        expected = whole.logits(volumes)
        torch.testing.assert_close(split.logits(volumes), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("decoder", DECODERS)
def test_chunked_dice_matches_the_undivided_decode(decoder):
    """The Dice sums are accumulated over slabs before the ratio, like every other term."""
    batch = _batch()
    whole = _algorithm(decoder, dice_weight=1.0).training_step(batch)
    split = _algorithm(decoder, dice_weight=1.0, decode_chunks=3).training_step(batch)
    for name in ("loss", "cross_entropy", "dice_loss"):
        assert _value(split[name]) == pytest.approx(_value(whole[name]), rel=1e-5, abs=1e-6), name


def _gradient_disagreement(decoder: str, dtype: torch.dtype, **overrides: Any) -> float:
    """Largest gradient difference, relative to the undivided decode's, over every parameter."""
    torch.set_default_dtype(dtype)
    try:
        batch = {
            name: value.to(dtype) if value.is_floating_point() else value
            for name, value in _batch().items()
        }
        whole = _algorithm(decoder, **overrides)
        split = _algorithm(decoder, decode_chunks=3, **overrides)
        whole.training_step(batch)["loss"].backward()
        split.training_step(batch)["loss"].backward()
        reference = dict(whole.named_parameters())
        worst = 0.0
        for name, parameter in split.named_parameters():
            assert parameter.grad is not None, f"{name} received no gradient when chunked"
            expected = reference[name].grad
            assert expected is not None
            scale = max(float(expected.abs().max()), 1e-30)
            worst = max(worst, float((parameter.grad - expected).abs().max()) / scale)
        return worst
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.unit
@pytest.mark.parametrize("decoder", ["subpixel", "unetr"])
def test_chunked_gradients_match_the_undivided_decode(decoder):
    """In float64, where summing in slabs and summing at once can only differ by algebra.

    `test_affinity_chunked.test_chunked_gradients_match_the_undivided_decode` explains why float32
    cannot tell a rounding difference from a small halo error.
    """
    assert _gradient_disagreement(decoder, torch.float64) < 1e-12


@pytest.mark.unit
@pytest.mark.parametrize("decoder", ["subpixel", "unetr"])
def test_chunked_dice_gradients_match_the_undivided_decode(decoder):
    """The ratio of slab-summed terms has to backpropagate into every slab, not only the last."""
    assert _gradient_disagreement(decoder, torch.float64, dice_weight=1.0) < 1e-12


@pytest.mark.unit
@pytest.mark.parametrize("decoder", ["interpolate", "linear"])
def test_interpolating_chunked_gradients_match_the_undivided_decode(decoder):
    """Bounded in float32 only: `VoxelHead` upsamples in float32 whatever the default dtype.

    The forward equality above is the exact check for these heads -- a halo too short changes the
    scores at every slab face by far more than this.
    """
    assert _gradient_disagreement(decoder, torch.float32) < 5e-3

