"""Unit tests for multi-resolution masked autoencoding."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from algorithms.muvit_mae import MuViTDecoderLayer, MuViTMAE
from algorithms.registry import AlgorithmRegistry
from models.muvit import MuViT3D

LEVELS = (1, 4)
PATCHES_PER_LEVEL = 8  # 16^3 volume, 8^3 patches -> 2^3 grid
TOTAL_PATCHES = PATCHES_PER_LEVEL * len(LEVELS)


def _encoder(**overrides: Any) -> MuViT3D:
    kwargs: dict[str, Any] = dict(
        levels=LEVELS, img_size=(16, 16, 16), patch_size=(8, 8, 8), in_channels=1,
        embed_dim=32, depth=2, num_heads=2, attention_backend="sdpa",
    )
    kwargs.update(overrides)
    return MuViT3D(**kwargs)


def _algorithm(encoder: MuViT3D | None = None, **overrides: Any) -> MuViTMAE:
    encoder = encoder or _encoder()
    kwargs: dict[str, Any] = dict(
        input_axes="lzyx", mask_ratio=0.75, decoder_embed_dim=16, decoder_depth=2,
        decoder_num_heads=2,
    )
    kwargs.update(overrides)
    return MuViTMAE(encoder, **kwargs)


def _batch(batch_size: int = 2, encoder: MuViT3D | None = None) -> dict[str, torch.Tensor]:
    encoder = encoder or _encoder()
    return {
        "img": torch.randn(batch_size, len(LEVELS), 16, 16, 16),
        "bbox": encoder.default_bbox(batch_size, torch.device("cpu")),
    }


@pytest.mark.unit
def test_registered_under_its_name():
    algorithm = AlgorithmRegistry.build("muvit_mae", _encoder(), input_axes="lzyx")
    assert isinstance(algorithm, MuViTMAE)


@pytest.mark.unit
def test_produces_a_finite_loss():
    algorithm = _algorithm()
    metrics = algorithm(_batch())
    assert torch.isfinite(metrics["loss"])
    assert metrics["loss"].item() > 0


@pytest.mark.unit
def test_gradients_reach_the_encoder_and_every_decoder():
    encoder = _encoder()
    algorithm = _algorithm(encoder)
    algorithm(_batch(encoder=encoder))["loss"].backward()

    assert encoder.level_embed.grad is not None
    assert all(proj[0].weight.grad is not None for proj in encoder.patch_proj)
    assert all(block.rotary.inv_freqs[0].grad is not None for block in encoder.blocks)
    assert algorithm.mask_tokens.grad is not None
    # Every level's decoder must be exercised, not just the first.
    for decoder in algorithm.decoders:
        assert decoder.head.weight.grad is not None
        assert decoder.layers[0].attn.qkv.weight.grad is not None
        assert decoder.layers[0].cross_attn.to_q.weight.grad is not None


@pytest.mark.unit
def test_one_decoder_and_one_head_per_level():
    algorithm = _algorithm(_encoder(levels=(1, 4, 16)))
    assert len(algorithm.decoders) == 3
    assert all(decoder.head.out_features == 512 for decoder in algorithm.decoders)
    assert algorithm.mask_tokens.shape == (3, 1, 16)


@pytest.mark.unit
def test_only_the_first_decoder_layer_cross_attends():
    # The paper places cross-attention in the first layer only; it is where scales exchange
    # information, and putting it everywhere would be a different (larger) architecture.
    algorithm = _algorithm(decoder_depth=3)
    for decoder in algorithm.decoders:
        assert decoder.layers[0].cross_attn is not None
        assert all(layer.cross_attn is None for layer in list(decoder.layers)[1:])


# --------------------------------------------------------------------------------------------
# Masking: the part that makes this multi-resolution rather than plain MAE.
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_masking_keeps_the_requested_global_fraction():
    algorithm = _algorithm(mask_ratio=0.75)
    keep, mask = algorithm._dirichlet_masking(4, torch.device("cpu"))
    expected_keep = round(TOTAL_PATCHES * 0.25)
    assert keep.shape == (4, expected_keep)
    assert torch.allclose(mask.sum(dim=1), torch.full((4,), float(TOTAL_PATCHES - expected_keep)))


@pytest.mark.unit
def test_masking_marks_exactly_the_tokens_it_did_not_keep():
    algorithm = _algorithm()
    keep, mask = algorithm._dirichlet_masking(3, torch.device("cpu"))
    for row in range(3):
        assert (mask[row, keep[row]] == 0).all(), "kept tokens must not be marked hidden"
        assert mask[row].sum() == TOTAL_PATCHES - keep.shape[1]


@pytest.mark.unit
def test_kept_indices_are_distinct():
    # Sampling without replacement. With replacement, a token could be encoded twice while another
    # was silently never seen, and the loss would still look reasonable.
    algorithm = _algorithm()
    keep, _ = algorithm._dirichlet_masking(8, torch.device("cpu"))
    for row in range(8):
        assert len(set(keep[row].tolist())) == keep.shape[1]


@pytest.mark.unit
def test_masking_splits_unevenly_across_levels():
    # The whole purpose of the Dirichlet draw. Uniform masking would put close to the same number of
    # visible tokens in each level every time; this must vary.
    torch.manual_seed(0)
    algorithm = _algorithm(dirichlet_alpha=0.5)
    _, mask = algorithm._dirichlet_masking(64, torch.device("cpu"))
    hidden = mask.reshape(64, len(LEVELS), PATCHES_PER_LEVEL).sum(dim=-1)
    spread = (hidden[:, 0] - hidden[:, 1]).abs()
    assert spread.max() > 0, "levels were masked identically in every sample"
    mean_spread = spread.float().mean()
    assert mean_spread > 0.5, f"level split barely varies: mean spread {mean_spread}"


@pytest.mark.unit
def test_a_larger_alpha_evens_the_split_out():
    # Pins the direction of the knob: alpha -> infinity approaches uniform masking, small alpha
    # concentrates on one level. Getting this backwards would still train.
    torch.manual_seed(0)
    lopsided = _algorithm(dirichlet_alpha=0.05)
    balanced = _algorithm(dirichlet_alpha=50.0)

    def mean_spread(algorithm):
        _, mask = algorithm._dirichlet_masking(128, torch.device("cpu"))
        hidden = mask.reshape(128, len(LEVELS), PATCHES_PER_LEVEL).sum(dim=-1)
        return (hidden[:, 0] - hidden[:, 1]).abs().float().mean()

    assert mean_spread(lopsided) > mean_spread(balanced)


@pytest.mark.unit
def test_masking_varies_between_samples_in_a_batch():
    torch.manual_seed(0)
    algorithm = _algorithm()
    _, mask = algorithm._dirichlet_masking(16, torch.device("cpu"))
    assert not torch.allclose(mask[0], mask[1:]), "every sample got the same mask"


def _visible_per_level(algorithm: MuViTMAE, batch_size: int) -> torch.Tensor:
    _, mask = algorithm._dirichlet_masking(batch_size, torch.device("cpu"))
    return (1.0 - mask).reshape(batch_size, len(LEVELS), PATCHES_PER_LEVEL).sum(dim=-1)


@pytest.mark.unit
def test_a_small_alpha_concentrates_visibility_on_one_level():
    # Distinguishes Dirichlet weighting from plain uniform masking, which the weaker
    # "levels differ at all" check cannot: uniform sampling of 4 visible tokens from two levels of
    # 8 lands them all in one level 2*C(8,4)/C(16,4) = 7.7% of the time, purely by chance. A small
    # alpha makes the weights near one-hot, so it should happen almost always. Mutation testing is
    # how this surfaced -- ignoring the weights entirely left the earlier test passing.
    torch.manual_seed(0)
    visible = _visible_per_level(_algorithm(dirichlet_alpha=0.05), 256)
    concentrated = (visible.min(dim=1).values == 0).float().mean().item()
    assert concentrated > 0.5, f"only {concentrated:.0%} of samples concentrated on one level"


@pytest.mark.unit
def test_each_sample_draws_its_own_level_weights():
    # One draw shared across the batch would make every sample in a step favour the same level.
    # With a near-one-hot alpha, per-sample draws must disagree about which level that is.
    torch.manual_seed(0)
    visible = _visible_per_level(_algorithm(dirichlet_alpha=0.05), 32)
    favoured = visible.argmax(dim=1)
    assert favoured.unique().numel() > 1, (
        "every sample favoured the same level, which is what one shared draw per step looks like"
    )


@pytest.mark.unit
def test_visible_coordinates_stay_in_step_with_visible_tokens():
    # The silent failure this guards: masking the tokens but gathering the coordinates with
    # different indices attaches every visible token to another patch's position. Shapes still line
    # up, the loss still falls, and the world-coordinate geometry is quietly nonsense.
    encoder = _encoder()
    algorithm = _algorithm(encoder)
    batch = _batch(1, encoder=encoder)

    keep = torch.tensor([[0, 5, 9]])
    mask = torch.ones(1, TOTAL_PATCHES)
    mask[0, keep[0]] = 0.0
    algorithm._dirichlet_masking = lambda *args: (keep, mask)

    seen: dict[str, torch.Tensor] = {}
    real_encode = encoder.encode

    def spy(tokens, coords):
        seen["tokens"], seen["coords"] = tokens, coords
        return real_encode(tokens, coords)

    encoder.encode = spy
    algorithm(batch)

    # `embed` is deterministic, so recomputing it gives the same tokens to compare against.
    volumes = encoder.prepare_input(batch["img"], "lzyx")
    all_tokens, all_coords = encoder.embed(volumes, batch["bbox"])
    assert torch.allclose(seen["tokens"], all_tokens[:, keep[0]], atol=1e-5)
    assert torch.allclose(seen["coords"], all_coords[:, keep[0]], atol=1e-5)


# --------------------------------------------------------------------------------------------
# Loss shape: masked-only, averaged per level.
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_loss_ignores_visible_patches():
    # Scoring visible patches would reward copying the input. Perturbing the prediction at visible
    # positions must not move the loss; at hidden positions it must.
    torch.manual_seed(0)
    encoder = _encoder()
    algorithm = _algorithm(encoder, norm_pix_loss=False)
    batch = _batch(encoder=encoder)

    captured: dict[str, Any] = {}
    real_decode = algorithm._decode

    def spy(latent, coords, keep_indices, visible_coords):
        prediction = real_decode(latent, coords, keep_indices, visible_coords)
        captured["keep"] = keep_indices
        captured["prediction"] = prediction
        return prediction

    algorithm._decode = spy
    torch.manual_seed(1)
    baseline = algorithm(batch)["loss"].item()

    # Re-run with the prediction altered only where tokens were visible.
    def visible_only(latent, coords, keep_indices, visible_coords):
        prediction = real_decode(latent, coords, keep_indices, visible_coords)
        altered = prediction.clone()
        rows = torch.arange(prediction.shape[0]).unsqueeze(1)
        altered[rows, captured["keep"]] += 1000.0
        return altered

    algorithm._decode = visible_only
    torch.manual_seed(1)
    perturbed = algorithm(batch)["loss"].item()
    assert abs(perturbed - baseline) < 1e-4, "the loss saw visible patches"


@pytest.mark.unit
def test_loss_weights_levels_equally_despite_uneven_masking():
    # Averaging per level and then across levels, as the paper specifies, rather than one mean over
    # all masked patches. The two differ exactly when levels have different numbers of hidden
    # patches -- which the Dirichlet draw guarantees -- so this pins the choice.
    #
    # Both the mask and the prediction error are fixed, so the expected loss is known in closed
    # form and the assertion is against the implementation rather than against arithmetic repeated
    # in the test.
    encoder = _encoder()
    algorithm = _algorithm(encoder, norm_pix_loss=False)
    batch = _batch(1, encoder=encoder)

    # Level 0: exactly one hidden patch. Level 1: every patch hidden.
    mask = torch.zeros(1, TOTAL_PATCHES)
    mask[0, 0] = 1.0
    mask[0, PATCHES_PER_LEVEL:] = 1.0
    keep = torch.arange(1, PATCHES_PER_LEVEL).unsqueeze(0)
    algorithm._dirichlet_masking = lambda *args: (keep, mask)

    # Put a squared error of 12 on the one hidden patch of level 0, and none anywhere else. Adding
    # a constant c across a patch makes that patch's mean squared error exactly c**2.
    volumes = encoder.prepare_input(batch["img"], "lzyx")
    target = encoder.patchify(volumes)
    offset = torch.zeros_like(target)
    offset[0, 0, :] = 12.0**0.5
    algorithm._decode = lambda *args: target + offset

    loss = algorithm(batch)["loss"].item()

    # Per level: level 0 contributes 12/1, level 1 contributes 0/8, averaged over the two -> 6.
    # A single mean over all 9 hidden patches would instead give 12/9 = 1.33.
    assert abs(loss - 6.0) < 1e-4, f"expected per-level averaging (6.0), got {loss}"
    assert abs(loss - 12.0 / 9.0) > 1.0, "loss looks like a global mean over masked patches"


@pytest.mark.unit
def test_reported_metrics_describe_the_draw():
    algorithm = _algorithm(mask_ratio=0.75)
    metrics = algorithm(_batch())
    assert 0.0 <= metrics["finest_visible_share"].item() <= 1.0
    assert abs(metrics["masked_fraction"].item() - 0.75) < 0.2


# --------------------------------------------------------------------------------------------
# Contracts.
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_requires_world_coordinates_in_the_batch():
    # Falling back to a guessed geometry is the one failure this must not have: the paper measures a
    # substantial cost for wrong coordinates, and nothing in the loss would reveal it.
    algorithm = _algorithm()
    with pytest.raises(KeyError, match="needs world coordinates"):
        algorithm({"img": torch.randn(2, 2, 16, 16, 16)})


@pytest.mark.unit
def test_rejects_a_mask_ratio_outside_the_open_unit_interval():
    for ratio in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="mask_ratio"):
            _algorithm(mask_ratio=ratio)


@pytest.mark.unit
def test_rejects_a_nonpositive_dirichlet_alpha():
    with pytest.raises(ValueError, match="dirichlet_alpha"):
        _algorithm(dirichlet_alpha=0.0)


@pytest.mark.unit
def test_rejects_a_decoder_with_no_layers():
    # Depth zero would silently drop the cross-attention that carries information across scales.
    with pytest.raises(ValueError, match="decoder_depth"):
        _algorithm(decoder_depth=0)


@pytest.mark.unit
def test_input_axes_contradicting_the_dataset_is_rejected():
    class _Dataset:
        sample_axes = "lcxyz"

        def __getattr__(self, name):
            raise AttributeError(name)

    with pytest.raises(ValueError, match="contradicts the dataset"):
        MuViTMAE(_encoder(), _Dataset(), input_axes="lzyx")


@pytest.mark.unit
def test_axes_are_taken_from_the_dataset_when_not_given():
    class _Dataset:
        sample_axes = "lzyx"

    assert MuViTMAE(_encoder(), _Dataset()).input_axes == "lzyx"


@pytest.mark.unit
def test_validation_step_matches_training_step():
    torch.manual_seed(0)
    encoder = _encoder()
    algorithm = _algorithm(encoder).eval()
    batch = _batch(encoder=encoder)

    torch.manual_seed(1)
    train = algorithm.training_step(batch)["loss"]
    torch.manual_seed(1)
    validate = algorithm.validation_step(batch)["loss"]
    assert torch.allclose(train, validate)


@pytest.mark.unit
def test_forward_is_the_training_step():
    # What makes plain DDP correct: the gradient all-reduce hangs off forward hooks.
    torch.manual_seed(0)
    encoder = _encoder()
    algorithm = _algorithm(encoder)
    batch = _batch(encoder=encoder)

    torch.manual_seed(1)
    through_forward = algorithm(batch)["loss"]
    torch.manual_seed(1)
    direct = algorithm.training_step(batch)["loss"]
    assert torch.allclose(through_forward, direct)


@pytest.mark.unit
def test_decoder_layer_without_cross_attention_needs_no_context():
    layer = MuViTDecoderLayer(16, 2, attention_backend="sdpa", with_cross_attention=False)
    x = torch.randn(2, 5, 16)
    assert layer(x, torch.randn(2, 5, 3)).shape == x.shape


@pytest.mark.unit
def test_cross_attending_layer_demands_context_coordinates():
    layer = MuViTDecoderLayer(16, 2, attention_backend="sdpa", with_cross_attention=True)
    with pytest.raises(ValueError, match="context_coords"):
        layer(torch.randn(2, 5, 16), torch.randn(2, 5, 3), context=torch.randn(2, 3, 16))


@pytest.mark.unit
def test_cross_attention_reads_the_context():
    # If the context were ignored the decoder could not use information from other scales at all,
    # which is the mechanism the paper credits for cross-scale learning.
    torch.manual_seed(0)
    layer = MuViTDecoderLayer(16, 2, attention_backend="sdpa", with_cross_attention=True).eval()
    x, coords = torch.randn(1, 5, 16), torch.randn(1, 5, 3)
    context_coords = torch.randn(1, 4, 3)

    with torch.no_grad():
        first = layer(x, coords, torch.randn(1, 4, 16), context_coords)
        second = layer(x, coords, torch.randn(1, 4, 16), context_coords)
    assert not torch.allclose(first, second)


@pytest.mark.unit
def test_cross_attention_depends_only_on_relative_position():
    # Cross-attention is where a masked patch reads from visible tokens at other scales, so what it
    # must see is the displacement between them. Sliding the query and context frames together must
    # change nothing. Rotating only the keys -- as the reference implementation does -- passes the
    # "context is read at all" check but fails this, because the logits then track the keys'
    # absolute positions instead.
    torch.manual_seed(0)
    layer = MuViTDecoderLayer(16, 2, attention_backend="sdpa", with_cross_attention=True).eval()
    x, context = torch.randn(1, 5, 16), torch.randn(1, 4, 16)
    coords, context_coords = torch.randn(1, 5, 3), torch.randn(1, 4, 3)
    shift = torch.tensor([13.0, -4.0, 2.5])

    with torch.no_grad():
        here = layer(x, coords, context, context_coords)
        there = layer(x, coords + shift, context, context_coords + shift)
    assert torch.allclose(here, there, atol=1e-4)


@pytest.mark.unit
def test_runs_under_half_precision_autocast():
    # Training runs bf16, and mixed precision breaks things that fp32 tests cannot see: parameters
    # stay fp32 under autocast while the layers around them return bf16, so anything combining the
    # two has to agree on a dtype. This caught a real failure -- scattering bf16 encoder outputs
    # into an fp32 mask-token background -- that every fp32 test passed straight through.
    encoder = _encoder()
    algorithm = _algorithm(encoder)
    batch = _batch(encoder=encoder)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = algorithm(batch)["loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert encoder.level_embed.grad is not None
    assert algorithm.mask_tokens.grad is not None


@pytest.mark.unit
def test_a_single_level_still_works():
    # The degenerate case: one level makes the Dirichlet draw a constant and every cross-attention
    # target a same-level token. It should run rather than divide by zero somewhere.
    encoder = _encoder(levels=(1,))
    algorithm = _algorithm(encoder)
    batch = {
        "img": torch.randn(2, 1, 16, 16, 16),
        "bbox": encoder.default_bbox(2, torch.device("cpu")),
    }
    assert torch.isfinite(algorithm(batch)["loss"])
