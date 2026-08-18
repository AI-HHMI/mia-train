"""Pins low-rank adaptation: that it starts as a no-op, what it freezes, and how it merges.

Every failure mode worth guarding here is silent. An adapter whose delta is not exactly zero at
initialisation means the run does not start from the pretrained function it was chosen to preserve;
a promotion that renamed a parameter means a checkpoint quietly stops loading part of itself; a
freeze
that caught `rope_embed.depth_scale` means a volumetric encoder trains with a two-dimensional sense
of position; and a merge that folds the wrong scaling means a downstream stage silently starts from
the wrong weights. None of these raise on their own, so they are asserted.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from engine.config import LoRAConfig, TrainerConfig
from engine.lora import apply_lora
from engine.optimizer import build_param_groups, wd_scale
from layers.common.lora import LoRAMixin, merge_all, promote
from layers.dinov3.attention import LinearKMaskedBias
from layers.dinov3.config import init_weights_vit
from models.dinov3_vit3d import DinoVisionTransformer3D
from utils.module_ops import named_apply
from utils.pretrained import load_pretrained

# Small enough to build a dozen of, structurally faithful to the released ViT-L in the ways that
# interact with adaptation: LayerScale gains, a masked fused-qkv bias, storage tokens, and the
# superposition rotary embedding whose depth gate must survive the freeze.
TINY = dict(
    img_size=32,
    patch_size=16,
    in_chans=1,
    embed_dim=64,
    depth=2,
    num_heads=4,
    n_storage_tokens=4,
    layerscale_init=1.0e-05,
    mask_k_bias=True,
    pos_embed_rope_type="superposition",
    pos_embed_rope_dtype="fp32",
)


def _model(**overrides) -> DinoVisionTransformer3D:
    torch.manual_seed(0)
    model = DinoVisionTransformer3D(**{**TINY, **overrides})
    return model.eval()


def _config(**overrides) -> LoRAConfig:
    return LoRAConfig(**{"rank": 4, "alpha": 8.0, **overrides})


def _volume() -> torch.Tensor:
    return torch.randn(2, 1, 32, 32, 32, generator=torch.Generator().manual_seed(1))


# ------------------------------------------------------------------ the no-op property


@pytest.mark.unit
def test_adaptation_is_exactly_a_no_op_at_initialisation() -> None:
    """The reason LoRA is the right tool here at all: `B = 0`, so the delta starts at zero.

    Asserted with `torch.equal`, not `allclose`. "Close" would also pass for an adapter initialised
    with small random values in both factors, which is a different and worse mechanism -- the model
    would start somewhere near its pretrained self rather than at it.
    """
    model, volume = _model(), _volume()
    with torch.no_grad():
        before = model.patch_features(volume)[0].clone()
    apply_lora(model, _config())
    with torch.no_grad():
        after = model.patch_features(volume)[0]
    assert torch.equal(before, after)


@pytest.mark.unit
def test_promotion_preserves_every_pretrained_parameter_name() -> None:
    """A superset, never a rename: what keeps `load_pretrained` and DCP working untouched."""
    model = _model()
    before = {name for name, _ in model.named_parameters()}
    buffers_before = {name for name, _ in model.named_buffers()}
    apply_lora(model, _config(targets=("attn_qkv", "attn_proj", "mlp")))
    after = {name for name, _ in model.named_parameters()}

    assert before < after, "promotion must add names without removing or renaming any"
    added = after - before
    assert all(name.endswith((".lora_a", ".lora_b")) for name in added), sorted(added)
    # `bias_mask` is a buffer on the fused qkv Linear; a wrapper-based adapter would have moved it.
    assert buffers_before < {name for name, _ in model.named_buffers()}


@pytest.mark.unit
def test_masked_key_bias_still_masks_after_promotion() -> None:
    """`LinearKMaskedBias.forward` must still run, not be bypassed by the mixin's `super()` call."""
    layer = LinearKMaskedBias(8, 24, bias=True)
    named_apply(init_weights_vit, layer, include_root=True)
    x = torch.randn(3, 8)
    before = layer(x).clone()
    promote(layer, rank=2, alpha=4.0)

    assert isinstance(layer, LinearKMaskedBias), "the base class must remain in the MRO"
    assert torch.equal(before, layer(x))
    assert bool((layer.bias_mask[8:16] == 0).all()), "the key third of the bias must stay masked"


# ------------------------------------------------------------------ what trains, what freezes


@pytest.mark.unit
def test_the_rope_depth_gate_is_trainable_even_with_every_switch_off() -> None:
    """The invariant the model declares, which no config may override.

    `depth_scale` is one scalar that gates *all* depth information in the positional encoding, and
    it arrives at exactly zero. Frozen, the encoder cannot tell one z-slice from another, and
    nothing in a run would report it.
    """
    model = _model()
    apply_lora(model, _config(train_stem=False, train_norms=False, train_tokens=False))
    assert dict(model.named_parameters())["rope_embed.depth_scale"].requires_grad


@pytest.mark.unit
def test_vanilla_rope_declares_no_required_parameter() -> None:
    """The gate is specific to superposition; the axial form gives depth its own channels."""
    assert _model(pos_embed_rope_type="vanilla").lora_required_trainable() == ()
    assert _model().lora_required_trainable() == ("rope_embed.depth_scale",)


@pytest.mark.unit
def test_the_switches_select_exactly_what_they_name() -> None:
    model = _model()
    report = apply_lora(model, _config())
    trainable = set(report.trainable)

    assert any(name.startswith("patch_embed.") for name in trainable)
    assert "cls_token" in trainable and "storage_tokens" in trainable and "mask_token" in trainable
    assert "blocks.0.norm1.weight" in trainable
    assert "blocks.0.ls1.gamma" in trainable, "LayerScale counts as a norm here"
    # The base weights of the adapted projections, and every FFN weight, stay frozen.
    assert "blocks.0.attn.qkv.weight" in set(report.frozen)
    assert "blocks.0.mlp.fc1.weight" in set(report.frozen)
    assert "blocks.0.attn.qkv.lora_a" in trainable


@pytest.mark.unit
def test_switches_off_freeze_their_groups() -> None:
    model = _model()
    report = apply_lora(
        model, _config(train_stem=False, train_norms=False, train_tokens=False)
    )
    frozen = set(report.frozen)
    assert "patch_embed.proj.weight" in frozen
    assert "cls_token" in frozen
    assert "blocks.0.norm1.weight" in frozen
    # Only the adapters and the declared-required gate survive.
    assert set(report.trainable) == {
        name for name, _ in model.named_parameters() if ".lora_" in name
    } | {"rope_embed.depth_scale"}


@pytest.mark.unit
def test_only_named_target_groups_are_adapted() -> None:
    model = _model()
    report = apply_lora(model, _config(targets=("attn_qkv",)))
    assert report.adapted == ["blocks.0.attn.qkv", "blocks.1.attn.qkv"]

    mlp = _model()
    mlp_report = apply_lora(mlp, _config(targets=("mlp",)))
    assert mlp_report.adapted == [
        "blocks.0.mlp.fc1",
        "blocks.0.mlp.fc2",
        "blocks.1.mlp.fc1",
        "blocks.1.mlp.fc2",
    ]


@pytest.mark.unit
def test_swiglu_ffn_is_covered_by_the_mlp_group() -> None:
    """The group is collected by walking for Linears, so it must not be `Mlp`-specific."""
    model = _model(ffn_layer="swiglu", embed_dim=64)
    report = apply_lora(model, _config(targets=("mlp",)))
    assert [name.rsplit(".", 1)[-1] for name in report.adapted[:3]] == ["w1", "w2", "w3"]


# ------------------------------------------------------------------ rejected configurations


@pytest.mark.unit
def test_unknown_target_group_names_the_menu() -> None:
    with pytest.raises(ValueError, match="attn_qkv"):
        apply_lora(_model(), _config(targets=("attention",)))


@pytest.mark.unit
def test_a_model_without_targets_is_rejected_rather_than_silently_frozen() -> None:
    class Bare(DinoVisionTransformer3D):
        def lora_target_groups(self):
            return {}

    torch.manual_seed(0)
    with pytest.raises(ValueError, match="declares no lora_target_groups"):
        apply_lora(Bare(**TINY), _config())


@pytest.mark.unit
def test_alpha_must_be_stated_when_rank_is() -> None:
    with pytest.raises(ValueError, match="alpha must be > 0"):
        LoRAConfig(rank=8)


@pytest.mark.unit
def test_options_without_a_rank_are_rejected() -> None:
    with pytest.raises(ValueError, match="leaves rank at 0"):
        LoRAConfig(alpha=16.0)


@pytest.mark.unit
def test_double_promotion_is_refused() -> None:
    layer = promote(nn.Linear(4, 4), rank=2, alpha=4.0)
    with pytest.raises(ValueError, match="already adapted"):
        promote(layer, rank=2, alpha=4.0)


# ------------------------------------------------------------------ merging


@pytest.mark.unit
def test_merge_reproduces_the_adapted_forward_pass() -> None:
    """Tolerance, not equality -- and the reason is worth recording.

    Merging reassociates the arithmetic: the adapted layer computes `Wx + s*B(Ax)` while the merged
    one computes `(W + s*BA)x`. Those are equal in exact arithmetic and differ at the last bit or
    two in float32, so this is the one property here that cannot be asserted with `torch.equal`.
    """
    model, volume = _model(), _volume()
    apply_lora(model, _config())
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRAMixin):
                module.lora_b.normal_(0.0, 0.05)
        adapted = model.patch_features(volume)[0].clone()

    merged = merge_all(model)
    assert merged, "nothing was merged"
    with torch.no_grad():
        after = model.patch_features(volume)[0]
    torch.testing.assert_close(adapted, after, rtol=0, atol=1e-5)


@pytest.mark.unit
def test_merging_removes_the_adapter_entirely() -> None:
    model = _model()
    before = {name for name, _ in model.named_parameters()}
    apply_lora(model, _config())
    merge_all(model)
    assert {name for name, _ in model.named_parameters()} == before
    assert not any(isinstance(module, LoRAMixin) for module in model.modules())


@pytest.mark.unit
def test_merging_twice_raises_rather_than_folding_again() -> None:
    layer = promote(nn.Linear(4, 4), rank=2, alpha=4.0)
    layer.merge_lora()
    with pytest.raises(AttributeError):
        layer.merge_lora()


# ------------------------------------------------------------------ optimizer integration


@pytest.mark.unit
def test_adapter_factors_are_exempt_from_weight_decay_by_default() -> None:
    """Both factors are 2-D, so rank alone would decay them -- which decays the delta toward zero.

    That is a pull back toward the pretrained weights, i.e. a regularizer toward θ₀ rather than
    toward the origin, and it must be opted into rather than inherited from a key named for
    something else.
    """
    parameter = nn.Parameter(torch.zeros(4, 8))
    default = TrainerConfig(max_steps=10, batch_size=1)
    assert wd_scale("blocks.0.attn.qkv.lora_a", parameter, default) == 0.0
    assert wd_scale("blocks.0.attn.qkv.lora_b", parameter, default) == 0.0
    assert wd_scale("blocks.0.attn.qkv.weight", parameter, default) == 1.0

    opted_in = TrainerConfig(max_steps=10, batch_size=1, zero_weight_decay_on_lora=False)
    assert wd_scale("blocks.0.attn.qkv.lora_a", parameter, opted_in) == 1.0


@pytest.mark.unit
def test_only_trainable_parameters_reach_the_optimizer() -> None:
    model = _model()
    report = apply_lora(model, _config())
    groups = build_param_groups(model, TrainerConfig(max_steps=10, batch_size=1))
    in_optimizer = {id(p) for group in groups for p in group["params"]}

    named = dict(model.named_parameters())
    assert {id(named[name]) for name in report.trainable} == in_optimizer
    assert all(id(named[name]) not in in_optimizer for name in report.frozen)


# ---------------------------------------------------------------- the three-stage checkpoint chain


def _save_as_algorithm(model: nn.Module, path) -> None:
    """Write `model`'s state the way a training run does: the encoder nested under `model.`."""
    torch.save({"model": {f"model.{k}": v for k, v in model.state_dict().items()}}, path)


def _trained_ssl_encoder() -> DinoVisionTransformer3D:
    """An adapted model standing in for the output of a LoRA SSL stage."""
    model = _model()
    apply_lora(model, _config())
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRAMixin):
                module.lora_b.normal_(0.0, 0.05)
        model.rope_embed.depth_scale.fill_(0.3)  # the gate learned something
    return model


@pytest.mark.unit
def test_released_checkpoint_loads_into_an_adapted_model_unchanged(tmp_path) -> None:
    """Stage 1. The adapter tensors are `kept_initial`; nothing is unused and nothing mismatches.

    This is the payoff of name-preserving promotion: `strict = true` and `allow_unused = false`, the
    settings every real config uses, need no relaxation to load a plain checkpoint into an adapted
    model.
    """
    plain = _model()
    checkpoint = tmp_path / "released.pt"
    torch.save(plain.state_dict(), checkpoint)

    adapted = _model()
    apply_lora(adapted, _config())
    report = load_pretrained(adapted, checkpoint, strict=True, allow_unused=False)

    assert report.unused == [] and report.mismatched == []
    assert set(report.kept_initial) == {
        name for name, _ in adapted.named_parameters() if ".lora_" in name
    } | {name for name, _ in adapted.named_buffers() if name.endswith(".lora_scaling")}


@pytest.mark.unit
def test_b1_chain_carries_the_adapter_forward(tmp_path) -> None:
    """Arm B1: the finetune continues the SSL stage's adapters, so its function is inherited."""
    ssl_model = _trained_ssl_encoder()
    volume = _volume()
    with torch.no_grad():
        expected = ssl_model.patch_features(volume)[0].clone()

    checkpoint = tmp_path / "ssl.pt"
    _save_as_algorithm(ssl_model, checkpoint)

    stage2 = _model()
    apply_lora(stage2, _config())
    report = load_pretrained(stage2, checkpoint, prefix="model.", strict=True)
    with torch.no_grad():
        torch.testing.assert_close(stage2.patch_features(volume)[0], expected, rtol=0, atol=0)

    assert report.merged == []
    assert stage2.blocks[0].attn.qkv.lora_b.abs().max() > 0, "the adapter must have carried over"
    gate = dict(stage2.named_parameters())["rope_embed.depth_scale"]
    assert gate.detach().item() == pytest.approx(0.3)


@pytest.mark.unit
def test_b2_chain_rebases_the_adapter(tmp_path) -> None:
    """Arm B2: `merge_lora` folds the SSL adaptation into the prior and re-zeroes the adapter.

    The two properties that make this the arm it is: the model computes what the SSL stage computed,
    and its adapter is back at zero so the finetune spends its whole rank budget on the task.
    """
    ssl_model = _trained_ssl_encoder()
    volume = _volume()
    with torch.no_grad():
        expected = ssl_model.patch_features(volume)[0].clone()

    checkpoint = tmp_path / "ssl.pt"
    _save_as_algorithm(ssl_model, checkpoint)

    stage2 = _model()
    apply_lora(stage2, _config())
    report = load_pretrained(
        stage2, checkpoint, prefix="model.", strict=True, merge_lora=True
    )

    assert len(report.merged) == 4, report.merged  # qkv + proj over two blocks
    assert report.unused == [] and report.mismatched == []
    with torch.no_grad():
        torch.testing.assert_close(stage2.patch_features(volume)[0], expected, rtol=0, atol=1e-5)
    assert stage2.blocks[0].attn.qkv.lora_b.abs().max() == 0, "the new adapter must start at zero"
    gate = dict(stage2.named_parameters())["rope_embed.depth_scale"]
    assert gate.detach().item() == pytest.approx(0.3)


@pytest.mark.unit
def test_merged_checkpoint_loads_into_a_model_that_knows_nothing_about_lora(tmp_path) -> None:
    """The other reason to merge: handing an adapted encoder to an unadapted config, for eval."""
    ssl_model = _trained_ssl_encoder()
    volume = _volume()
    with torch.no_grad():
        expected = ssl_model.patch_features(volume)[0].clone()

    checkpoint = tmp_path / "ssl.pt"
    _save_as_algorithm(ssl_model, checkpoint)

    plain = _model()
    load_pretrained(plain, checkpoint, prefix="model.", strict=True, merge_lora=True)
    with torch.no_grad():
        torch.testing.assert_close(plain.patch_features(volume)[0], expected, rtol=0, atol=1e-5)


@pytest.mark.unit
def test_merge_lora_on_a_checkpoint_without_adapters_is_an_error(tmp_path) -> None:
    """Otherwise the option silently does nothing and the run is not the arm it claims to be."""
    checkpoint = tmp_path / "plain.pt"
    torch.save(_model().state_dict(), checkpoint)
    with pytest.raises(ValueError, match="holds no LoRA adapter"):
        load_pretrained(_model(), checkpoint, strict=True, merge_lora=True)


@pytest.mark.unit
def test_merge_uses_the_checkpoints_own_scaling_not_the_loading_runs(tmp_path) -> None:
    """`lora_scaling` is a persistent buffer precisely so this cannot go wrong silently.

    The SSL stage here trained at alpha/rank = 2.0 and the loading run is configured at 0.5. The
    folded weights must reflect what was trained, not what the new config happens to say.
    """
    ssl_model = _model()
    apply_lora(ssl_model, LoRAConfig(rank=4, alpha=8.0))  # scaling 2.0
    volume = _volume()
    with torch.no_grad():
        for module in ssl_model.modules():
            if isinstance(module, LoRAMixin):
                module.lora_b.normal_(0.0, 0.05)
        expected = ssl_model.patch_features(volume)[0].clone()

    checkpoint = tmp_path / "ssl.pt"
    _save_as_algorithm(ssl_model, checkpoint)

    stage2 = _model()
    apply_lora(stage2, LoRAConfig(rank=4, alpha=2.0))  # scaling 0.5 -- deliberately different
    load_pretrained(stage2, checkpoint, prefix="model.", strict=True, merge_lora=True)
    with torch.no_grad():
        torch.testing.assert_close(stage2.patch_features(volume)[0], expected, rtol=0, atol=1e-5)


# ---------------------------------------- the build order, which the unit tests once missed


def _run_config(tmp_path, lora: bool, init_path=None, merge=False):
    """A RunConfig shaped like a real one, carrying only what `prepare_model` reads."""
    from engine.config import InitConfig, TrainerConfig
    from utils.config import ComponentConfig, RunConfig

    return RunConfig(
        experiment_name="order",
        model=ComponentConfig(name="dinov3_vit3d", kwargs=dict(TINY)),
        algorithm=ComponentConfig(name="simmim", kwargs={}),
        data=ComponentConfig(name="miao_volumes", kwargs={}),
        trainer=TrainerConfig(max_steps=10, batch_size=1),
        init=InitConfig(
            path=str(init_path) if init_path else "",
            prefix="model." if init_path else "",
            strict=bool(init_path),
            merge_lora=merge,
        ),
        lora=_config() if lora else LoRAConfig(),
    )


@pytest.mark.unit
def test_prepare_model_adapts_before_it_loads(tmp_path) -> None:
    """The ordering bug a 20-step smoke run caught and these tests did not.

    Stage 2 of a LoRA arm loads a checkpoint that already carries `lora_a`/`lora_b`/`lora_scaling`.
    With `[lora]` applied *after* `[init]`, the model has nowhere to put them and `load_pretrained`
    refuses -- correctly, and fatally: 288 homeless tensors on a real ViT-L. The tests that existed
    all called `apply_lora` first, which is the right order, so none of them could see it.

    Asserted through `prepare_model`, the function production actually uses, rather than by
    reproducing its steps here -- which is the mistake that let this through the first time.
    """
    from engine.run import prepare_model

    ssl_model = _trained_ssl_encoder()
    checkpoint = tmp_path / "stage_a.pt"
    _save_as_algorithm(ssl_model, checkpoint)

    volume = _volume()
    with torch.no_grad():
        expected = ssl_model.patch_features(volume)[0].clone()

    stage_b = prepare_model(_run_config(tmp_path, lora=True, init_path=checkpoint))
    with torch.no_grad():
        torch.testing.assert_close(stage_b.patch_features(volume)[0], expected, rtol=0, atol=0)
    adapter = stage_b.blocks[0].attn.qkv.lora_b
    assert adapter.abs().max() > 0, "stage a's adapter did not carry over"


@pytest.mark.unit
def test_prepare_model_loads_a_plain_checkpoint_into_an_adapted_model(tmp_path) -> None:
    """The other direction -- stage 1 of a LoRA arm, reading a released checkpoint."""
    from engine.run import prepare_model

    plain = _model()
    checkpoint = tmp_path / "released.pt"
    _save_as_algorithm(plain, checkpoint)

    volume = _volume()
    with torch.no_grad():
        expected = plain.patch_features(volume)[0].clone()

    adapted = prepare_model(_run_config(tmp_path, lora=True, init_path=checkpoint))
    assert any(".lora_a" in name for name, _ in adapted.named_parameters())
    with torch.no_grad():
        # The adapter is a no-op at initialisation, so an adapted model that loaded plain weights
        # computes exactly what the plain model does.
        torch.testing.assert_close(adapted.patch_features(volume)[0], expected, rtol=0, atol=0)


@pytest.mark.unit
def test_prepare_model_merges_when_asked(tmp_path) -> None:
    """`[init].merge_lora` under the production order: fold in, then hand back a zeroed adapter."""
    from engine.run import prepare_model

    ssl_model = _trained_ssl_encoder()
    checkpoint = tmp_path / "stage_a.pt"
    _save_as_algorithm(ssl_model, checkpoint)
    volume = _volume()
    with torch.no_grad():
        expected = ssl_model.patch_features(volume)[0].clone()

    rebased = prepare_model(_run_config(tmp_path, lora=True, init_path=checkpoint, merge=True))
    with torch.no_grad():
        torch.testing.assert_close(rebased.patch_features(volume)[0], expected, rtol=0, atol=1e-5)
    assert rebased.blocks[0].attn.qkv.lora_b.abs().max() == 0


@pytest.mark.unit
def test_prepare_model_without_lora_is_untouched(tmp_path) -> None:
    """A config with no `[lora]` section must produce exactly what it always did."""
    from engine.run import prepare_model

    model = prepare_model(_run_config(tmp_path, lora=False))
    assert not any(".lora_" in name for name, _ in model.named_parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())
