from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from engine.config import TrainerConfig
from engine.optimizer import build_lr_scheduler, build_optimizer, lr_multiplier


@pytest.mark.unit
def test_build_optimizer_uses_configured_hyperparameters():
    model = nn.Linear(4, 4)
    config = TrainerConfig(max_steps=10, batch_size=1, lr=3e-4, weight_decay=0.05, beta2=0.99)
    optimizer = build_optimizer(model, config)

    # Indexed by what the group *is*, not by position: norms and biases are exempt from decay by
    # default, so a Linear yields two groups and which one lands first is an implementation detail.
    group = next(g for g in optimizer.param_groups if g["wd_multiplier"] == 1.0)
    assert group["lr"] == pytest.approx(3e-4)
    assert group["weight_decay"] == pytest.approx(0.05)
    assert group["betas"] == (0.9, 0.99)


@pytest.mark.unit
def test_warmup_ramps_linearly_to_one():
    config = TrainerConfig(max_steps=100, batch_size=1, warmup_steps=10)
    assert lr_multiplier(0, config) == pytest.approx(0.1)
    assert lr_multiplier(4, config) == pytest.approx(0.5)
    assert lr_multiplier(9, config) == pytest.approx(1.0)


@pytest.mark.unit
def test_cosine_decays_from_one_to_min_lr_ratio():
    config = TrainerConfig(max_steps=100, batch_size=1, warmup_steps=10, min_lr_ratio=0.1)
    assert lr_multiplier(10, config) == pytest.approx(1.0)
    assert lr_multiplier(100, config) == pytest.approx(0.1)
    midpoint = lr_multiplier(55, config)
    assert 0.1 < midpoint < 1.0


@pytest.mark.unit
def test_schedule_is_monotonically_non_increasing_after_warmup():
    config = TrainerConfig(max_steps=50, batch_size=1, warmup_steps=5)
    values = [lr_multiplier(step, config) for step in range(5, 51)]
    pairs = zip(values, values[1:], strict=False)  # values[1:] is one shorter by construction
    assert all(later <= earlier + 1e-12 for earlier, later in pairs)


@pytest.mark.unit
def test_no_warmup_starts_at_full_lr():
    config = TrainerConfig(max_steps=100, batch_size=1, warmup_steps=0)
    assert lr_multiplier(0, config) == pytest.approx(1.0)


@pytest.mark.unit
def test_scheduler_applies_multiplier_to_optimizer_lr():
    model = nn.Linear(4, 4)
    config = TrainerConfig(max_steps=100, batch_size=1, lr=1e-2, warmup_steps=10)
    optimizer = build_optimizer(model, config)
    scheduler = build_lr_scheduler(optimizer, config)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    for _ in range(9):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)


# ---------------------------------------------------------------- per-parameter lr and wd

import components  # noqa: E402, F401  (populates the model registry for the convention test)
from engine.optimizer import (  # noqa: E402
    _block_count,
    build_param_groups,
    lr_scale,
    parameter_depth,
    wd_scale,
    weight_decay_at,
)
from models.registry import ModelRegistry  # noqa: E402


def _stack(depth: int = 4) -> nn.Module:
    """A model shaped like this repo's: a stem, a numbered block stack, a head."""
    model = nn.Module()
    model.patch_embed = nn.Linear(4, 4)
    model.cls_token = nn.Parameter(torch.zeros(1, 4))
    model.blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(depth))
    model.norm = nn.LayerNorm(4)
    return model


@pytest.mark.unit
def test_learning_rate_shaping_is_off_by_default():
    """Layerwise decay and the patch-embed multiplier are opt-in, so every group shares one lr."""
    config = TrainerConfig(max_steps=10, batch_size=1, lr=3e-4, weight_decay=0.05)
    groups = build_param_groups(_stack(), config)

    assert [group["lr"] for group in groups] == pytest.approx([3e-4] * len(groups))
    assert {group["lr_multiplier"] for group in groups} == {1.0}


@pytest.mark.unit
def test_norms_and_biases_are_exempt_from_decay_by_default():
    """The opposite of the lr knobs: this one is on, because decaying a normalization gain works
    against the normalization itself, and every transformer recipe exempts it."""
    config = TrainerConfig(max_steps=10, batch_size=1, lr=3e-4, weight_decay=0.05)
    groups = build_param_groups(_stack(), config)

    assert sorted(group["weight_decay"] for group in groups) == pytest.approx([0.0, 0.05])


@pytest.mark.unit
def test_decay_exemption_can_be_turned_off():
    config = TrainerConfig(
        max_steps=10, batch_size=1, weight_decay=0.05, zero_weight_decay_on_norm_and_bias=False
    )
    groups = build_param_groups(_stack(), config)

    assert len(groups) == 1, "one group again once nothing is exempt"
    assert groups[0]["weight_decay"] == pytest.approx(0.05)


@pytest.mark.unit
def test_depth_runs_from_the_stem_to_above_the_stack():
    assert parameter_depth("patch_embed.proj.weight", 4) == 0, "the stem"
    assert parameter_depth("cls_token", 4) == 0
    assert parameter_depth("blocks.0.attn.weight", 4) == 1
    assert parameter_depth("blocks.3.attn.weight", 4) == 4
    assert parameter_depth("norm.weight", 4) == 5, "past the last block"


@pytest.mark.unit
def test_layerwise_decay_slows_the_early_layers_most():
    """The point of it: preserve the general features near the input while the top adapts."""
    config = TrainerConfig(max_steps=10, batch_size=1, layerwise_lr_decay=0.9)
    scales = [lr_scale(f"blocks.{i}.attn.weight", 4, config) for i in range(4)]

    assert scales == sorted(scales), "deeper blocks must move at least as fast"
    assert scales[0] == pytest.approx(0.9**4)
    assert scales[-1] == pytest.approx(0.9)
    assert lr_scale("norm.weight", 4, config) == pytest.approx(1.0), "the top is undiscounted"


@pytest.mark.unit
def test_patch_embed_multiplier_compounds_with_the_layerwise_decay():
    config = TrainerConfig(
        max_steps=10, batch_size=1, layerwise_lr_decay=0.9, patch_embed_lr_mult=0.2
    )
    assert lr_scale("patch_embed.proj.weight", 4, config) == pytest.approx(0.9**5 * 0.2)
    assert lr_scale("cls_token", 4, config) == pytest.approx(0.9**5), "only patch_embed is scaled"


@pytest.mark.unit
def test_weight_decay_schedule_runs_between_its_endpoints():
    config = TrainerConfig(
        max_steps=100, batch_size=1, weight_decay=0.04, final_weight_decay=0.4
    )
    assert weight_decay_at(0, config) == pytest.approx(0.04)
    assert weight_decay_at(100, config) == pytest.approx(0.4)
    assert 0.04 < weight_decay_at(50, config) < 0.4


@pytest.mark.unit
def test_weight_decay_is_constant_without_an_endpoint():
    config = TrainerConfig(max_steps=100, batch_size=1, weight_decay=0.04)
    assert weight_decay_at(0, config) == weight_decay_at(100, config) == pytest.approx(0.04)


@pytest.mark.unit
def test_norms_and_biases_can_be_exempted_from_decay():
    config = TrainerConfig(
        max_steps=10, batch_size=1, weight_decay=0.05, zero_weight_decay_on_norm_and_bias=True
    )
    groups = build_param_groups(_stack(), config)
    decays = sorted(group["weight_decay"] for group in groups)

    assert decays == pytest.approx([0.0, 0.05])
    # The exempt group must hold the norm gain, the biases and the learned token.
    exempt = next(g for g in groups if g["weight_decay"] == 0.0)
    assert len(exempt["params"]) > 1


@pytest.mark.unit
def test_scheduler_drives_weight_decay_as_well_as_lr():
    config = TrainerConfig(
        max_steps=100, batch_size=1, lr=1e-2, weight_decay=0.04, final_weight_decay=0.4
    )
    optimizer = build_optimizer(_stack(), config)
    scheduler = build_lr_scheduler(optimizer, config)

    decaying = next(g for g in optimizer.param_groups if g["wd_multiplier"] == 1.0)
    exempt = next(g for g in optimizer.param_groups if g["wd_multiplier"] == 0.0)

    assert decaying["weight_decay"] == pytest.approx(0.04)
    for _ in range(100):
        scheduler.step()
    assert decaying["weight_decay"] == pytest.approx(0.4)
    assert exempt["weight_decay"] == 0.0, "the exemption holds across the whole schedule"


@pytest.mark.unit
def test_schedule_multiplies_each_group_own_base_lr():
    """Layerwise scaling and the warmup/cosine schedule have to compose, not overwrite."""
    config = TrainerConfig(
        max_steps=100, batch_size=1, lr=1e-2, warmup_steps=10, layerwise_lr_decay=0.5
    )
    optimizer = build_optimizer(_stack(depth=2), config)
    scheduler = build_lr_scheduler(optimizer, config)

    ratios_at_warmup_start = [g["lr"] for g in optimizer.param_groups]
    for _ in range(9):
        scheduler.step()
    ratios_at_full_lr = [g["lr"] for g in optimizer.param_groups]

    # Every group scaled by the same schedule factor (0.1 -> 1.0), keeping their relative sizes.
    for start, full in zip(ratios_at_warmup_start, ratios_at_full_lr, strict=True):
        assert full == pytest.approx(start * 10)


# Every registered architecture, built small. `parameter_depth` reads a parameter's place in the
# network out of its name, so these conventions are load-bearing: a model whose stem or stack is
# named differently gets a silently wrong learning-rate profile, not an error.
_TINY_MODELS = {
    "vit3d": dict(img_size=(16, 16, 16), patch_size=(8, 8, 8), in_channels=1, embed_dim=32,
                  depth=2, num_heads=2),
    "muvit3d": dict(levels=(1, 4), img_size=(8, 8, 8), patch_size=(4, 4, 4), in_channels=1,
                    embed_dim=16, depth=2, num_heads=2),
    "dinov3_vit": dict(img_size=16, patch_size=8, in_chans=1, embed_dim=32, depth=2,
                       num_heads=2, pos_embed_rope_dtype="fp32"),
    "dinov3_vit3d": dict(img_size=16, patch_size=8, in_chans=1, embed_dim=32, depth=2,
                         num_heads=2, pos_embed_rope_dtype="fp32"),
}


@pytest.mark.unit
def test_the_tiny_models_cover_the_whole_registry():
    """So the two tests below cannot quietly stop covering a model someone adds."""
    assert set(_TINY_MODELS) == set(ModelRegistry.available())


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_TINY_MODELS))
def test_every_model_has_a_recognisable_block_stack(name):
    model = ModelRegistry.build(name, **_TINY_MODELS[name])
    assert _block_count(model) == 2, (
        f"{name}'s transformer stack is not named `blocks`, so layerwise_lr_decay would appear "
        "to be configured and silently do nothing"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_TINY_MODELS))
def test_every_model_has_a_recognisable_stem(name):
    """The stem must take the *deepest* discount.

    Getting this wrong is worse than not having the feature: an unrecognised stem falls through to
    the above-the-stack bucket and receives the single largest learning rate in the network, which
    is the exact opposite of what layerwise decay is for.
    """
    model = ModelRegistry.build(name, **_TINY_MODELS[name])
    config = TrainerConfig(max_steps=10, batch_size=1, layerwise_lr_decay=0.75)
    n_blocks = _block_count(model)

    stem = [n for n, _ in model.named_parameters() if parameter_depth(n, n_blocks) == 0]
    assert stem, f"{name} has no parameter recognised as its stem"

    slowest_block = min(
        lr_scale(n, n_blocks, config)
        for n, _ in model.named_parameters()
        if "blocks." in n
    )
    for parameter_name in stem:
        assert lr_scale(parameter_name, n_blocks, config) <= slowest_block, parameter_name


@pytest.mark.unit
def test_an_algorithms_own_parameters_are_not_mistaken_for_the_backbone_stem():
    """A strategy's decoder is not the encoder's input layer, however similarly it is named.

    MAE's decoder holds a freshly-initialised `mask_token`; matched by name alone it would take
    the stem's discount and train at a fraction of the rate of the decoder it belongs to.
    """
    from algorithms.registry import AlgorithmRegistry

    encoder = ModelRegistry.build("vit3d", **_TINY_MODELS["vit3d"])
    algorithm = AlgorithmRegistry.build(
        "mae", encoder, None, input_axes="lcxyz", decoder_embed_dim=16, decoder_depth=1,
        decoder_num_heads=2,
    )
    config = TrainerConfig(max_steps=10, batch_size=1, lr=1e-3, layerwise_lr_decay=0.75)
    groups = build_param_groups(algorithm, config)

    by_multiplier = {id(p): g["lr_multiplier"] for g in groups for p in g["params"]}
    named = dict(algorithm.named_parameters())
    assert by_multiplier[id(named["mask_token"])] == 1.0, "the decoder's own token is not the stem"
    assert by_multiplier[id(named["model.patch_embed.weight"])] < 1.0, "the encoder's stem is"


@pytest.mark.unit
def test_rank_one_parameters_are_the_ones_exempted_from_decay():
    """Rank, not name: a LayerNorm inside an nn.Sequential has no "norm" in its parameter name,
    and learned rotary frequencies have no marker at all."""
    config = TrainerConfig(max_steps=10, batch_size=1, weight_decay=0.05,
                           zero_weight_decay_on_norm_and_bias=True)
    for name in sorted(_TINY_MODELS):
        model = ModelRegistry.build(name, **_TINY_MODELS[name])
        for parameter_name, parameter in model.named_parameters():
            expected = 0.0 if parameter.ndim < 2 else 1.0
            assert wd_scale(parameter_name, parameter, config) == expected, (
                f"{name}: {parameter_name} (ndim {parameter.ndim})"
            )


@pytest.mark.unit
def test_frozen_parameters_are_left_out_of_the_optimizer():
    """DINOv3 carries a whole frozen teacher; handing it to AdamW would waste state on it."""
    model = _stack()
    for parameter in model.blocks[0].parameters():
        parameter.requires_grad_(False)

    config = TrainerConfig(max_steps=10, batch_size=1)
    grouped = {id(p) for group in build_param_groups(model, config) for p in group["params"]}
    assert not (grouped & {id(p) for p in model.blocks[0].parameters()})
    assert grouped == {id(p) for p in model.parameters() if p.requires_grad}


@pytest.mark.unit
def test_group_zero_carries_the_configured_learning_rate():
    """Anything reporting `param_groups[0]["lr"]` -- the trainer's log line -- should show the
    number the config states, not whichever group ended up most discounted."""
    config = TrainerConfig(max_steps=100, batch_size=1, lr=1e-3, layerwise_lr_decay=0.75,
                           patch_embed_lr_mult=0.2)
    groups = build_param_groups(_stack(depth=6), config)

    assert groups[0]["lr"] == pytest.approx(1e-3)
    assert groups[0]["lr"] == pytest.approx(max(g["lr"] for g in groups))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"layerwise_lr_decay": 1.5}, "layerwise_lr_decay"),
        ({"layerwise_lr_decay": 0.0}, "layerwise_lr_decay"),
        ({"patch_embed_lr_mult": 0.0}, "patch_embed_lr_mult"),
        ({"final_weight_decay": -0.1}, "final_weight_decay"),
    ],
)
def test_rejects_out_of_range_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TrainerConfig(max_steps=10, batch_size=1, **kwargs)
