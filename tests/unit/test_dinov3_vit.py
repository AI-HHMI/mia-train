"""Unit tests for the ported DINOv3 vision transformers, 2D and 3D.

The two classes share almost all of their behaviour, so the shared contract is parametrised over
both and only the genuinely rank-specific parts (input shape, `prepare_input`, and the 3D RoPE
choice) are tested separately.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from models.dinov3_vit import DinoVisionTransformer
from models.dinov3_vit3d import DinoVisionTransformer3D
from models.registry import ModelRegistry

# img 32 / patch 8 -> a 4x4 grid (16 patches) in 2D; img 16 / patch 8 -> 2x2x2 (8) in 3D.
TINY_2D: dict[str, Any] = dict(
    img_size=32, patch_size=8, in_chans=3, embed_dim=64, depth=2, num_heads=4,
    pos_embed_rope_dtype="fp32",
)
TINY_3D: dict[str, Any] = dict(
    img_size=16, patch_size=8, in_chans=1, embed_dim=96, depth=2, num_heads=4,
    pos_embed_rope_dtype="fp32",
)


def _model(cls, **overrides: Any):
    kwargs = dict(TINY_2D if cls is DinoVisionTransformer else TINY_3D)
    kwargs.update(overrides)
    model = cls(**kwargs)
    model.init_weights()
    return model.eval()


def _input(cls, batch: int = 2) -> torch.Tensor:
    if cls is DinoVisionTransformer:
        return torch.randn(batch, 3, 32, 32)
    return torch.randn(batch, 1, 16, 16, 16)


def _shape(cls, side: int) -> tuple[int, ...]:
    """One sample's shape at a given side length, as `flops` takes it -- no batch dimension."""
    if cls is DinoVisionTransformer:
        return (3, side, side)
    return (1, side, side, side)


# The side length each tiny model is configured for, and one twice as large: these encoders are
# meant to run at resolutions other than the one they were built for, so the tests need both.
CONFIGURED_SIDE = {DinoVisionTransformer: 32, DinoVisionTransformer3D: 16}

# Forward FLOPs of the tiny models at the configured side and at twice it; derived below.
PINNED_FLOPS = {
    DinoVisionTransformer: {32: 3_883_520, 64: 16_515_584},
    DinoVisionTransformer3D: {16: 4_829_952, 32: 38_290_176},
}

BOTH = pytest.mark.parametrize("cls", [DinoVisionTransformer, DinoVisionTransformer3D])


# ---------------------------------------------------------------- registration


@pytest.mark.unit
def test_registered_under_their_config_names():
    assert ModelRegistry.get("dinov3_vit") is DinoVisionTransformer
    assert ModelRegistry.get("dinov3_vit3d") is DinoVisionTransformer3D


@pytest.mark.unit
@BOTH
def test_buildable_through_the_registry(cls):
    name = "dinov3_vit" if cls is DinoVisionTransformer else "dinov3_vit3d"
    kwargs = TINY_2D if cls is DinoVisionTransformer else TINY_3D
    assert isinstance(ModelRegistry.build(name, **kwargs), cls)


# ---------------------------------------------------------------- shapes


@pytest.mark.unit
@BOTH
def test_patch_grid_and_counts(cls):
    model = _model(cls)
    if cls is DinoVisionTransformer:
        assert model.grid_size == (4, 4)
        assert model.num_patches == 16
    else:
        assert model.grid_size == (2, 2, 2)
        assert model.num_patches == 8


@pytest.mark.unit
@BOTH
def test_forward_features_shapes(cls):
    model = _model(cls, n_storage_tokens=3)
    out = model.forward_features(_input(cls))

    assert out["x_norm_clstoken"].shape == (2, model.embed_dim)
    assert out["x_storage_tokens"].shape == (2, 3, model.embed_dim)
    assert out["x_norm_patchtokens"].shape == (2, model.num_patches, model.embed_dim)
    # The pre-norm stream still carries CLS + storage + patches.
    assert out["x_prenorm"].shape == (2, 1 + 3 + model.num_patches, model.embed_dim)
    assert out["masks"] is None


@pytest.mark.unit
@BOTH
def test_forward_returns_the_class_token(cls):
    model = _model(cls)
    x = _input(cls)
    assert model(x).shape == (2, model.embed_dim)
    assert torch.allclose(model(x), model.forward_features(x)["x_norm_clstoken"], atol=1e-6)


@pytest.mark.unit
@BOTH
def test_is_training_returns_the_full_feature_dict(cls):
    model = _model(cls)
    out = model(_input(cls), is_training=True)
    assert set(out) == {
        "x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens", "x_prenorm", "masks"
    }


@pytest.mark.unit
@BOTH
def test_no_storage_tokens_yields_an_empty_slice(cls):
    model = _model(cls, n_storage_tokens=0)
    out = model.forward_features(_input(cls))
    assert out["x_storage_tokens"].shape == (2, 0, model.embed_dim)


# ---------------------------------------------------------------- multi-crop and masking


@pytest.mark.unit
def test_2d_accepts_crops_of_different_resolutions():
    """One model, two crop sizes in one call -- what rotary position makes possible."""
    model = _model(DinoVisionTransformer)
    outs = model.forward_features([torch.randn(2, 3, 32, 32), torch.randn(2, 3, 16, 16)],
                                  [None, None])
    assert [o["x_norm_patchtokens"].shape[1] for o in outs] == [16, 4]


@pytest.mark.unit
@BOTH
def test_masked_patches_are_replaced_by_the_mask_token(cls):
    model = _model(cls)
    x = _input(cls)
    masks = torch.zeros(2, model.num_patches, dtype=torch.bool)
    masks[:, 0] = True

    with torch.no_grad():
        unmasked = model.forward_features(x)["x_prenorm"]
        masked = model.forward_features(x, masks)["x_prenorm"]

    assert not torch.allclose(masked, unmasked)
    assert masks.equal(model.forward_features(x, masks)["masks"])


@pytest.mark.unit
@BOTH
def test_mask_token_receives_gradient_even_when_unused(cls):
    """`cls_token + 0 * mask_token` keeps it in the graph, so FSDP/DDP do not see an unused
    parameter on ranks that happen to draw no masks."""
    model = _model(cls)
    model.forward_features(_input(cls))["x_norm_clstoken"].sum().backward()
    assert model.mask_token.grad is not None


# ---------------------------------------------------------------- intermediate layers


@pytest.mark.unit
@BOTH
def test_get_intermediate_layers_returns_patch_tokens_only(cls):
    model = _model(cls, n_storage_tokens=2)
    (out,) = model.get_intermediate_layers(_input(cls), n=1)
    assert out.shape == (2, model.num_patches, model.embed_dim)


@pytest.mark.unit
@BOTH
def test_get_intermediate_layers_n_selects_the_last_n_blocks(cls):
    model = _model(cls)
    assert len(model.get_intermediate_layers(_input(cls), n=2)) == 2
    assert len(model.get_intermediate_layers(_input(cls), n=[0])) == 1


@pytest.mark.unit
@BOTH
def test_get_intermediate_layers_can_return_tokens_and_reshape(cls):
    model = _model(cls, n_storage_tokens=2)
    ((patches, cls_token, extras),) = model.get_intermediate_layers(
        _input(cls), n=1, reshape=True, return_class_token=True, return_extra_tokens=True
    )
    grid = model.grid_size
    assert patches.shape == (2, model.embed_dim, *grid), "reshape restores the spatial grid"
    assert cls_token.shape == (2, model.embed_dim)
    assert extras.shape == (2, 2, model.embed_dim)


# ---------------------------------------------------------------- configuration


@pytest.mark.unit
@BOTH
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"norm_layer": "batchnorm"}, "unknown norm_layer"),
        ({"ffn_layer": "gelu"}, "unknown ffn_layer"),
        ({"pos_embed_rope_dtype": "fp8"}, "unknown pos_embed_rope_dtype"),
    ],
)
def test_unknown_config_strings_are_rejected(cls, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(cls, **kwargs)


@pytest.mark.unit
def test_unknown_rope_type_is_rejected():
    """Without this the if/else would leave `rope_embed` unset and fail much later."""
    with pytest.raises(ValueError, match="unknown pos_embed_rope_type"):
        _model(DinoVisionTransformer3D, pos_embed_rope_type="axial")


@pytest.mark.unit
@BOTH
def test_unknown_kwargs_are_rejected_rather_than_ignored(cls):
    # Upstream swallows unrecognised kwargs with a warning; a typo in a config should not train.
    with pytest.raises(TypeError):
        _model(cls, embed_dimension=64)


@pytest.mark.unit
@BOTH
@pytest.mark.parametrize("norm_layer", ["layernorm", "layernormbf16", "rmsnorm"])
@pytest.mark.parametrize("ffn_layer", ["mlp", "swiglu", "swiglu32"])
def test_norm_and_ffn_variants_run(cls, norm_layer, ffn_layer):
    model = _model(cls, norm_layer=norm_layer, ffn_layer=ffn_layer)
    assert model(_input(cls)).shape == (2, model.embed_dim)


@pytest.mark.unit
@BOTH
def test_untied_norms_add_parameters_and_still_run(cls):
    tied = _model(cls)
    untied = _model(cls, untie_cls_and_patch_norms=True, untie_global_and_local_cls_norm=True)

    assert untied.num_parameters() > tied.num_parameters()
    assert untied.cls_norm is not None and untied.local_cls_norm is not None
    assert tied.cls_norm is None and tied.local_cls_norm is None
    assert untied(_input(cls)).shape == (2, untied.embed_dim)


@pytest.mark.unit
def test_3d_superposition_rope_is_selectable():
    from layers.dinov3.rope import RopePositionEmbedding3D, RopePositionEmbedding3DSuperposition

    assert isinstance(_model(DinoVisionTransformer3D).rope_embed, RopePositionEmbedding3D)
    superposed = _model(DinoVisionTransformer3D, pos_embed_rope_type="superposition")
    assert isinstance(superposed.rope_embed, RopePositionEmbedding3DSuperposition)
    assert superposed(_input(DinoVisionTransformer3D)).shape == (2, superposed.embed_dim)


# ---------------------------------------------------------------- BaseModel contract


@pytest.mark.unit
@BOTH
def test_declares_the_methods_fsdp_must_wrap(cls):
    # Neither is reached through `forward`, so FSDP2 would not all-gather around them otherwise.
    assert _model(cls).extra_forward_methods() == ("forward_features", "get_intermediate_layers")


@pytest.mark.unit
@BOTH
def test_flops_grows_with_depth(cls):
    shape = _shape(cls, CONFIGURED_SIDE[cls])
    shallow = _model(cls, depth=2).flops(shape)
    deep = _model(cls, depth=4).flops(shape)
    assert 0 < shallow < deep


@pytest.mark.unit
@BOTH
def test_flops_scale_with_the_input_resolution(cls):
    # Pinned absolute values rather than a bare inequality: the count was wrong by a constant for
    # every resolution but one, and "greater than zero" is exactly what failed to notice. Closed
    # form with p patches, n = p + 1 (CLS, no storage tokens) and depth = 2:
    #     patch_proj = 2*p*patch_volume*d
    #     per_block  = 8*n*d*d + 4*n*n*d + 4*n*d*hidden
    #     total      = patch_proj + depth*per_block
    # 2D at the tiny geometry: patch_volume = 8**2 * 3 = 192, d = 64, hidden = 256, p = 16 or 64.
    # 3D: patch_volume = 8**3 * 1 = 512, d = 96, hidden = 384, p = 8 or 64.
    model = _model(cls)
    for side, expected in PINNED_FLOPS[cls].items():
        assert model.flops(_shape(cls, side)) == expected


@pytest.mark.unit
@BOTH
def test_flops_read_the_grid_off_the_input_not_the_configured_img_size(cls):
    # The regression this pins: `flops` used to derive the grid from `self.num_patches`, so it
    # reported the construction-time cost whatever it was asked about. These encoders have no
    # position-embedding table and SSL runs global and local crops through them in the same step,
    # so the answer has to follow the argument.
    side = CONFIGURED_SIDE[cls]
    small = _model(cls, img_size=side)
    large = _model(cls, img_size=2 * side)
    assert small.flops(_shape(cls, 2 * side)) == large.flops(_shape(cls, 2 * side))
    assert large.flops(_shape(cls, side)) == small.flops(_shape(cls, side))


@pytest.mark.unit
@BOTH
def test_flops_cost_anisotropic_crops(cls):
    # `grid_size` cannot express these -- `img_size` is a single int, so it only describes a
    # cube/square -- which is the other half of why the grid has to come from `input_shape`.
    side = CONFIGURED_SIDE[cls]
    model = _model(cls)
    channels, _, *rest = _shape(cls, side)
    stretched = (channels, 2 * side, *rest)  # one spatial axis doubled, the others left alone
    cube, twice = model.flops(_shape(cls, side)), model.flops(_shape(cls, 2 * side))
    assert cube < model.flops(stretched) < twice


@pytest.mark.unit
@BOTH
def test_flops_use_the_ffn_width_that_was_actually_built(cls):
    # SwiGLU runs three projections of 2/3 the nominal hidden width, so it matches the plain MLP's
    # two full-width ones only while that 2/3 is already aligned. `ffn_ratio=3.5` is chosen so it
    # is not: `swiglu64` then rounds up and the two genuinely differ, which is what the old
    # `int(embed_dim * ffn_ratio)` formula could not see.
    shape = _shape(cls, CONFIGURED_SIDE[cls])
    plain = _model(cls, ffn_layer="mlp", ffn_ratio=3.5)
    gated = _model(cls, ffn_layer="swiglu64", ffn_ratio=3.5)

    plain_hidden = plain.blocks[0].mlp.fc1.out_features
    gated_hidden = gated.blocks[0].mlp.w1.out_features
    assert 3 * gated_hidden != 2 * plain_hidden  # otherwise the two would cost the same by luck

    n = plain.num_patches + 1 + plain.n_storage_tokens
    per_block = 2 * n * plain.embed_dim * (3 * gated_hidden - 2 * plain_hidden)
    assert gated.flops(shape) - plain.flops(shape) == len(plain.blocks) * per_block


@pytest.mark.unit
@BOTH
def test_flops_rejects_a_shape_with_no_spatial_axes(cls):
    # An empty or truncated shape would otherwise silently cost a one-patch grid.
    with pytest.raises(ValueError, match="patch grid"):
        _model(cls).flops(())


@pytest.mark.unit
@BOTH
def test_flops_reject_a_channel_count_the_model_could_not_run(cls):
    # The resolution is free here, but the channel count is fixed by the patch convolution, so a
    # shape naming another one describes a forward pass that raises. Unchecked it was answered
    # anyway, with the configured model's cost: `patch_volume` is linear in `in_chans`, so the
    # patch-projection term came back scaled by the ratio of the two counts and nothing said so.
    model = _model(cls)
    channels, *spatial = _shape(cls, CONFIGURED_SIDE[cls])
    with pytest.raises(ValueError, match="channel axis"):
        model.flops((channels + 1, *spatial))
    # Naming only the spatial axes asserts nothing about channels, and is still costed: the shape
    # has no channel to disagree with, and inventing one to demand would be a different contract.
    assert model.flops(tuple(spatial)) == model.flops((channels, *spatial))


@pytest.mark.unit
@BOTH
def test_no_tensor_parallel_plan(cls):
    assert _model(cls).tensor_parallel_plan() is None


# ---------------------------------------------------------------- prepare_input


@pytest.mark.unit
@BOTH
def test_prepare_input_drops_the_level_axis_and_adds_a_channel(cls):
    model = _model(cls)
    if cls is DinoVisionTransformer:
        assert model.prepare_input(torch.randn(2, 1, 32, 32), "lyx").shape == (2, 1, 32, 32)
        assert model.prepare_input(torch.randn(2, 1, 3, 32, 32), "lcyx").shape == (2, 3, 32, 32)
    else:
        assert model.prepare_input(
            torch.randn(2, 1, 16, 16, 16), "lzyx"
        ).shape == (2, 1, 16, 16, 16)
        assert model.prepare_input(
            torch.randn(2, 1, 1, 16, 16, 16), "lczyx"
        ).shape == (2, 1, 16, 16, 16)


@pytest.mark.unit
@BOTH
def test_prepare_input_requires_a_level_axis(cls):
    axes = "yx" if cls is DinoVisionTransformer else "zyx"
    with pytest.raises(ValueError, match="must contain 'l'"):
        _model(cls).prepare_input(torch.randn(2, 4, 4), axes)


@pytest.mark.unit
@BOTH
def test_prepare_input_rejects_multiple_scale_levels(cls):
    model = _model(cls)
    if cls is DinoVisionTransformer:
        batch, axes = torch.randn(2, 3, 32, 32), "lyx"
    else:
        batch, axes = torch.randn(2, 3, 16, 16, 16), "lzyx"
    with pytest.raises(ValueError, match="single-scale"):
        model.prepare_input(batch, axes)


@pytest.mark.unit
@BOTH
def test_prepare_input_rejects_the_wrong_spatial_rank(cls):
    model = _model(cls)
    # Hand each model the *other* rank's axis order.
    axes = "lzyx" if cls is DinoVisionTransformer else "lyx"
    with pytest.raises(ValueError, match="not usable by a"):
        model.prepare_input(torch.randn(2, 1, 4, 4, 4), axes)


@pytest.mark.unit
@BOTH
def test_prepare_input_rejects_a_batch_that_contradicts_its_axes(cls):
    axes = "lcyx" if cls is DinoVisionTransformer else "lczyx"
    with pytest.raises(ValueError, match="implies a"):
        _model(cls).prepare_input(torch.randn(2, 1, 4), axes)


@pytest.mark.unit
@BOTH
def test_prepare_input_output_feeds_straight_into_the_model(cls):
    model = _model(cls)
    if cls is DinoVisionTransformer:
        batch, axes = torch.randn(2, 1, 3, 32, 32), "lcyx"
    else:
        batch, axes = torch.randn(2, 1, 1, 16, 16, 16), "lczyx"
    assert model(model.prepare_input(batch, axes)).shape == (2, model.embed_dim)


# ---------------------------------------------------------------- initialisation


@pytest.mark.unit
@BOTH
def test_registry_built_model_is_usable_without_an_explicit_init_call(cls):
    """`ModelRegistry.build` is the only production path (src/engine/run.py), and it does not
    call `init_weights`. Upstream relies on its own entry points to do so; nothing here would."""
    name = "dinov3_vit" if cls is DinoVisionTransformer else "dinov3_vit3d"
    kwargs = dict(TINY_2D if cls is DinoVisionTransformer else TINY_3D)
    kwargs.update(layerscale_init=1e-5, n_storage_tokens=2, mask_k_bias=True)

    model = ModelRegistry.build(name, **kwargs).eval()

    for tensor_name, tensor in model.named_parameters():
        assert torch.isfinite(tensor).all(), f"{tensor_name} was left uninitialised"
    # LayerScale at exactly zero would silence every residual branch, making the stack an identity.
    gamma = model.blocks[0].ls1.gamma
    assert torch.allclose(gamma, torch.full_like(gamma, 1e-5))
    # The masked-bias buffer starts as NaN and is only filled by the weight init.
    assert torch.isfinite(model.blocks[0].attn.qkv.bias_mask).all()
    assert torch.isfinite(model(_input(cls))).all()
