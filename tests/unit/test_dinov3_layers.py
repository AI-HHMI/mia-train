"""Unit tests for the DINOv3 building blocks under `layers/dinov3/` and `layers/common/`."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from layers.common.batched_tokens import cat_keep_shapes, uncat_with_shapes
from layers.common.layer_scale import LayerScale
from layers.common.rms_norm import RMSNorm
from layers.dinov3.attention import LinearKMaskedBias, SelfAttention, rope_apply
from layers.dinov3.block import SelfAttentionBlock
from layers.dinov3.config import ffn_layer_dict, init_weights_vit, norm_layer_dict
from layers.dinov3.ffn import Mlp, SwiGLUFFN
from layers.dinov3.patch_embed import PatchEmbed, PatchEmbed3D
from layers.dinov3.rope import (
    RopePositionEmbedding,
    RopePositionEmbedding3D,
    RopePositionEmbedding3DSuperposition,
)
from utils.module_ops import named_apply

# ---------------------------------------------------------------- batched tokens


@pytest.mark.unit
def test_cat_uncat_roundtrips_tensors_of_different_shapes():
    tensors = [torch.randn(2, 5, 8), torch.randn(3, 1, 8), torch.randn(1, 7, 8)]
    flat, shapes, counts = cat_keep_shapes(tensors)

    assert flat.shape == (2 * 5 + 3 * 1 + 1 * 7, 8)
    restored = uncat_with_shapes(flat, shapes, counts)
    for original, back in zip(tensors, restored, strict=True):
        assert torch.equal(original, back)


@pytest.mark.unit
def test_uncat_follows_a_width_change():
    # The whole point of concatenating is to run a projection once; that projection changes width.
    tensors = [torch.randn(2, 5, 8), torch.randn(3, 1, 8)]
    flat, shapes, counts = cat_keep_shapes(tensors)
    widened = nn.Linear(8, 24)(flat)

    restored = uncat_with_shapes(widened, shapes, counts)
    assert [tuple(t.shape) for t in restored] == [(2, 5, 24), (3, 1, 24)]


# ---------------------------------------------------------------- rope


def _relative_position_spread(rope_tables, head_dim: int) -> float:
    """How much a same-displacement attention logit drifts with absolute position.

    RoPE's defining property is that <rotate(q, i), rotate(k, j)> depends only on i - j. If this
    spread is not ~0 the embedding has stopped being a *relative* position encoding.
    """
    sin, cos = rope_tables
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, head_dim), torch.randn(1, 1, 1, head_dim)

    def logit(i: int, j: int) -> float:
        qi = rope_apply(q, sin[i : i + 1], cos[i : i + 1])
        kj = rope_apply(k, sin[j : j + 1], cos[j : j + 1])
        return float((qi * kj).sum())

    same_offset = [logit(i + 1, i) for i in range(0, 6, 2)]
    return max(same_offset) - min(same_offset)


@pytest.mark.unit
def test_2d_rope_encodes_only_relative_position():
    rope = RopePositionEmbedding(embed_dim=64, num_heads=4, base=100.0, dtype=torch.float32)
    assert _relative_position_spread(rope(H=1, W=8), rope.D_head) < 1e-4


@pytest.mark.unit
@pytest.mark.parametrize(
    ("embed_dim", "num_heads"),
    [
        (96, 4),  # head_dim 24, divisible by 6 -- no leftover channels
        (256, 4),  # head_dim 64 -> 60 rotary + 4 leftover, the case that used to break
        (512, 4),  # head_dim 128 -> 126 rotary + 2 leftover
    ],
)
def test_3d_rope_encodes_only_relative_position(embed_dim, num_heads):
    """Pins the padding fix.

    Upstream pads sin/cos with zeros *after* tiling, which both zeroes the leftover channels and
    slides the second copy of the angles out of alignment with `rope_rotate_half`'s split -- so the
    logit for a fixed displacement drifts with absolute position. Padding the half-angles before
    tiling is what keeps this a relative encoding.
    """
    rope = RopePositionEmbedding3D(
        embed_dim=embed_dim, num_heads=num_heads, base=100.0, dtype=torch.float32
    )
    assert _relative_position_spread(rope(D=1, H=1, W=8), rope.D_head) < 1e-4


@pytest.mark.unit
def test_3d_rope_passes_leftover_channels_through_unchanged():
    # head_dim 64 splits three ways as 60 rotary channels, leaving 4 that carry no position.
    rope = RopePositionEmbedding3D(embed_dim=256, num_heads=4, base=100.0, dtype=torch.float32)
    assert (rope.D_head, rope.D_rope) == (64, 60)

    sin, cos = rope(D=2, H=2, W=2)
    # A zero angle means cos=1 and sin=0, i.e. the identity -- not cos=0, which would erase them.
    positionless = (cos == 1.0) & (sin == 0.0)
    assert int(positionless[0].sum()) == rope.D_head - rope.D_rope

    x = torch.randn(1, 1, 8, rope.D_head)
    rotated = rope_apply(x, sin, cos)
    assert torch.allclose(rotated[..., positionless[0]], x[..., positionless[0]], atol=1e-6)


@pytest.mark.unit
def test_superposition_rope_starts_out_identical_to_2d():
    """`depth_scale` initialises at zero, so a 2D checkpoint transfers with no change in output."""
    rope3d = RopePositionEmbedding3DSuperposition(
        embed_dim=64, num_heads=4, base=100.0, dtype=torch.float32
    )
    rope2d = RopePositionEmbedding(embed_dim=64, num_heads=4, base=100.0, dtype=torch.float32)

    assert rope3d.depth_scale.detach().item() == 0.0
    assert rope3d.periods.shape == rope2d.periods.shape, "buffer footprint must match 2D exactly"

    sin3, cos3 = rope3d(D=1, H=4, W=4)
    sin2, cos2 = rope2d(H=4, W=4)
    assert torch.allclose(sin3, sin2, atol=1e-6)
    assert torch.allclose(cos3, cos2, atol=1e-6)


@pytest.mark.unit
def test_superposition_rope_depth_gate_actually_engages():
    rope = RopePositionEmbedding3DSuperposition(
        embed_dim=64, num_heads=4, base=100.0, dtype=torch.float32
    )
    before = rope(D=4, H=2, W=2)
    with torch.no_grad():
        rope.depth_scale.fill_(1.0)
    after = rope(D=4, H=2, W=2)
    assert not torch.allclose(before[0], after[0])


@pytest.mark.unit
def test_rope_rejects_both_parametrisations_at_once():
    with pytest.raises(ValueError, match="Either `base` or"):
        RopePositionEmbedding(
            embed_dim=64, num_heads=4, base=100.0, min_period=1.0, max_period=10.0
        )
    with pytest.raises(ValueError, match="Either `base` or"):
        RopePositionEmbedding3D(embed_dim=96, num_heads=4, base=None)


@pytest.mark.unit
def test_rope_coordinate_augmentation_is_training_only():
    rope = RopePositionEmbedding(
        embed_dim=64, num_heads=4, base=100.0, shift_coords=0.5, dtype=torch.float32
    )
    rope.eval()
    assert torch.allclose(rope(H=4, W=4)[0], rope(H=4, W=4)[0])

    rope.train()
    assert not torch.allclose(rope(H=4, W=4)[0], rope(H=4, W=4)[0])


# ---------------------------------------------------------------- patch embed


@pytest.mark.unit
def test_patch_embed_2d_shapes():
    embed = PatchEmbed(img_size=32, patch_size=8, in_chans=3, embed_dim=16)
    assert embed.num_patches == 16
    assert embed(torch.randn(2, 3, 32, 32)).shape == (2, 16, 16)

    unflattened = PatchEmbed(img_size=32, patch_size=8, in_chans=3, embed_dim=16,
                             flatten_embedding=False)
    assert unflattened(torch.randn(2, 3, 32, 32)).shape == (2, 4, 4, 16)


@pytest.mark.unit
def test_patch_embed_3d_shapes():
    embed = PatchEmbed3D(img_size=16, patch_size=8, in_chans=1, embed_dim=16)
    assert embed.num_patches == 8
    assert embed(torch.randn(2, 1, 16, 16, 16)).shape == (2, 8, 16)

    unflattened = PatchEmbed3D(img_size=16, patch_size=8, in_chans=1, embed_dim=16,
                               flatten_embedding=False)
    assert unflattened(torch.randn(2, 1, 16, 16, 16)).shape == (2, 2, 2, 2, 16)


@pytest.mark.unit
def test_patch_embed_3d_rejects_a_partial_patch():
    embed = PatchEmbed3D(img_size=16, patch_size=8, in_chans=1, embed_dim=16)
    with pytest.raises(AssertionError, match="not a multiple of patch"):
        embed(torch.randn(1, 1, 12, 16, 16))


@pytest.mark.unit
@pytest.mark.parametrize("cls", [PatchEmbed, PatchEmbed3D])
def test_init_weights_reaches_both_patch_embeddings(cls):
    """Upstream's `init_weights_vit` lists only the 2D class, silently leaving the 3D patch
    convolution on torch's default init. Both must be reached."""
    embed = cls(img_size=16, patch_size=8, in_chans=1, embed_dim=16)
    with torch.no_grad():
        embed.proj.weight.fill_(999.0)
        embed.proj.bias.fill_(999.0)

    # include_root because the patch embedding is the tree being walked here; inside a model it is
    # reached as a child of the root.
    named_apply(init_weights_vit, embed, include_root=True)

    rank = 2 if cls is PatchEmbed else 3
    bound = math.sqrt(1 / (1 * 8**rank))
    assert embed.proj.weight.abs().max() <= bound
    assert embed.proj.bias.abs().max() <= bound


# ---------------------------------------------------------------- ffn, norms, layer scale


@pytest.mark.unit
def test_mlp_and_swiglu_preserve_width():
    for module in (Mlp(in_features=16, hidden_features=32), SwiGLUFFN(in_features=16,
                                                                     hidden_features=32)):
        assert module(torch.randn(2, 5, 16)).shape == (2, 5, 16)


@pytest.mark.unit
def test_swiglu_align_to_rounds_the_hidden_width_up():
    # 2/3 of 100 is 66, which the presets round up to the next multiple of their alignment.
    assert SwiGLUFFN(in_features=16, hidden_features=100).w1.out_features == 72  # align_to=8
    assert ffn_layer_dict["swiglu32"](in_features=16, hidden_features=100).w1.out_features == 96
    assert ffn_layer_dict["swiglu64"](in_features=16, hidden_features=100).w1.out_features == 128


@pytest.mark.unit
def test_forward_list_matches_looping_forward():
    module = Mlp(in_features=8, hidden_features=16).eval()
    crops = [torch.randn(2, 5, 8), torch.randn(3, 2, 8)]
    with torch.no_grad():
        batched = module.forward_list(crops)
        looped = [module(c) for c in crops]
    for a, b in zip(batched, looped, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.unit
def test_rmsnorm_normalises_by_root_mean_square():
    norm = RMSNorm(8)
    out = norm(torch.randn(2, 4, 8))
    assert torch.allclose(out.pow(2).mean(-1).sqrt(), torch.ones(2, 4), atol=1e-3)


@pytest.mark.unit
def test_layer_scale_reset_sets_the_configured_value():
    scale = LayerScale(8, init_values=1e-4)
    scale.reset_parameters()
    assert torch.allclose(scale.gamma, torch.full((8,), 1e-4))
    assert torch.allclose(scale(torch.ones(1, 8)), torch.full((1, 8), 1e-4))


@pytest.mark.unit
def test_norm_layer_presets_differ_in_epsilon():
    assert norm_layer_dict["layernorm"](8).eps == 1e-6
    assert norm_layer_dict["layernormbf16"](8).eps == 1e-5
    assert isinstance(norm_layer_dict["rmsnorm"](8), RMSNorm)


# ---------------------------------------------------------------- attention


@pytest.mark.unit
def test_masked_key_bias_zeroes_the_key_third():
    linear = LinearKMaskedBias(6, 12, bias=True)
    named_apply(init_weights_vit, linear, include_root=True)

    assert torch.equal(linear.bias_mask[0:4], torch.ones(4)), "queries keep their bias"
    assert torch.equal(linear.bias_mask[4:8], torch.zeros(4)), "keys lose theirs"
    assert torch.equal(linear.bias_mask[8:12], torch.ones(4)), "values keep theirs"


@pytest.mark.unit
def test_attention_matches_textbook_attention_without_rope():
    torch.manual_seed(0)
    attn = SelfAttention(16, num_heads=4, qkv_bias=True).eval()
    x = torch.randn(2, 6, 16)

    qkv = attn.qkv(x).reshape(2, 6, 3, 4, 4).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(4)
    expected = torch.softmax(scores, dim=-1) @ v
    expected = attn.proj(expected.transpose(1, 2).reshape(2, 6, 16))

    with torch.no_grad():
        assert torch.allclose(attn(x), expected, atol=1e-5)


@pytest.mark.unit
def test_rope_leaves_the_prefix_tokens_unrotated():
    """CLS and storage tokens sit in front of the patch tokens and have no grid position."""
    attn = SelfAttention(16, num_heads=4).eval()
    rope = RopePositionEmbedding(embed_dim=16, num_heads=4, base=100.0, dtype=torch.float32)
    sin, cos = rope(H=2, W=2)  # 4 patch tokens

    n_prefix = 3
    q = torch.randn(1, 4, n_prefix + 4, 4)
    k = torch.randn(1, 4, n_prefix + 4, 4)
    q_out, k_out = attn.apply_rope(q, k, (sin, cos))

    assert torch.allclose(q_out[:, :, :n_prefix], q[:, :, :n_prefix], atol=1e-6)
    assert torch.allclose(k_out[:, :, :n_prefix], k[:, :, :n_prefix], atol=1e-6)
    assert not torch.allclose(q_out[:, :, n_prefix:], q[:, :, n_prefix:], atol=1e-3)


@pytest.mark.unit
def test_use_fa4_without_the_kernel_fails_with_a_reason():
    from layers.common.attention import flash4_status

    usable, _ = flash4_status()
    if usable:
        pytest.skip("FlashAttention-4 is usable here, so construction is expected to succeed")
    with pytest.raises(ValueError, match="FlashAttention-4 is unusable"):
        SelfAttention(16, num_heads=4, use_fa4=True)


# ---------------------------------------------------------------- block


@pytest.mark.unit
def test_block_single_tensor_and_singleton_list_agree():
    block = SelfAttentionBlock(dim=16, num_heads=4).eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        assert torch.allclose(block(x), block([x])[0], atol=1e-6)


@pytest.mark.unit
def test_block_handles_crops_of_different_lengths():
    block = SelfAttentionBlock(dim=16, num_heads=4).eval()
    crops = [torch.randn(2, 9, 16), torch.randn(2, 4, 16)]
    with torch.no_grad():
        out = block(crops, [None, None])
    assert [tuple(t.shape) for t in out] == [(2, 9, 16), (2, 4, 16)]


@pytest.mark.unit
def test_block_stochastic_depth_is_training_only():
    torch.manual_seed(0)
    block = SelfAttentionBlock(dim=16, num_heads=4, drop_path=0.5)
    x = torch.randn(8, 5, 16)

    block.eval()
    with torch.no_grad():
        assert torch.allclose(block(x), block(x))

    block.train()
    with torch.no_grad():
        assert not torch.allclose(block(x), block(x))


@pytest.mark.unit
def test_layerscale_is_only_added_when_configured():
    assert isinstance(SelfAttentionBlock(dim=16, num_heads=4).ls1, nn.Identity)
    assert isinstance(SelfAttentionBlock(dim=16, num_heads=4, init_values=1e-4).ls1, LayerScale)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("augmentation", "value"),
    # shift is an additive offset in coordinate units; jitter and rescale are multiplicative
    # factors whose log sets a symmetric range, so they must be greater than 1.
    [("shift_coords", 0.5), ("jitter_coords", 2.0), ("rescale_coords", 2.0)],
)
def test_superposition_rope_survives_train_mode_augmentation(augmentation, value):
    """The depth path is 1-D [DHW] while the spatial path is [DHW, 2], so it needs a differently
    shaped augmentation operand. Upstream indexes both the same way and raises on the first
    training step."""
    rope = RopePositionEmbedding3DSuperposition(
        embed_dim=64, num_heads=4, base=100.0, dtype=torch.float32, **{augmentation: value}
    )
    rope.train()
    sin, cos = rope(D=3, H=4, W=5)
    assert sin.shape == (3 * 4 * 5, rope.D_head)


@pytest.mark.unit
@pytest.mark.parametrize(
    "rope_cls", [RopePositionEmbedding3D, RopePositionEmbedding3DSuperposition]
)
def test_3d_rope_tables_always_span_the_whole_head(rope_cls):
    """Anything narrower than the head dimension fails inside `rope_apply`'s multiply."""
    for embed_dim, num_heads in [(96, 4), (256, 4), (512, 4), (64, 4), (128, 8)]:
        rope = rope_cls(embed_dim=embed_dim, num_heads=num_heads, base=100.0, dtype=torch.float32)
        sin, cos = rope(D=2, H=2, W=2)
        assert sin.shape[-1] == cos.shape[-1] == rope.D_head, f"{embed_dim}/{num_heads}"
        rope_apply(torch.randn(1, 1, 8, rope.D_head), sin, cos)  # must not raise


@pytest.mark.unit
def test_3d_rope_rejects_an_odd_head_dimension():
    # Rotation acts on channel pairs, so an odd head cannot be rotated; the 2D classes exclude
    # this via their own assertion, but the vanilla 3D one accepts more shapes.
    with pytest.raises(ValueError, match="odd head dimension"):
        RopePositionEmbedding3D(embed_dim=36, num_heads=4, base=100.0)
