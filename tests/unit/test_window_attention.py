"""Windowed and pooled-key attention must compute exactly what their definitions say.

`layers.common.window_attention` only moves tokens, and every way of moving them wrongly is silent:
a window built from the wrong tokens, a prefix token missing from a window's keys, a pooled key
rotated somewhere other than the centre of its box. Each still returns a tensor of the right shape
and a loss that trains. So the checks here are equalities against an explicit reference -- full
attention under the block mask the pattern stands for, or keys averaged box by box by hand -- in
float64, so that agreement means the algebra is right rather than that two roundings coincided.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from layers.common import window_attention
from layers.common.attention import SelfAttention as CommonSelfAttention
from layers.common.rope import AxialRotaryEmbedding
from layers.common.window_attention import (
    PREFIX_GROUP,
    attention_pairs,
    block_attention_layout,
    merge_windows,
    partition_windows,
    pool_grid,
    prefix_copies,
    resolve_window,
    sliding_block_mask,
    sliding_tile,
    sliding_window_attention,
    window_start,
    windowed_attention,
)
from layers.dinov3.attention import SelfAttention as Dinov3SelfAttention
from layers.dinov3.attention import rope_apply_to_suffix
from layers.dinov3.rope import RopePositionEmbedding3D, RopePositionEmbedding3DSuperposition
from models.dinov3_vit3d import DinoVisionTransformer3D
from models.vit import ViT3D

# Anisotropic on purpose, so an axis swapped anywhere changes which tokens share a window.
GRID = (4, 6, 2)
WINDOW = (2, 3, 1)
CELLS = math.prod(GRID)


def _worst(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute difference, as a number."""
    return float((a - b).detach().abs().max())


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The reference kernel, in the helpers' (B, N, heads, head_dim) layout."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def _grid_coords(grid: tuple[int, ...]) -> torch.Tensor:
    """Row-major patch indices, (1, prod(grid), rank) -- the layout the models' tokens are in."""
    axes = [torch.arange(g, dtype=torch.float64) for g in grid]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, len(grid))


def _window_ids(grid: tuple[int, ...], window: tuple[int, ...]) -> torch.Tensor:
    """Each grid token's window, numbered row-major over the windows."""
    coords = _grid_coords(grid)[0].long()
    ids = torch.zeros(coords.shape[0], dtype=torch.long)
    for axis, (g, w) in enumerate(zip(grid, window, strict=True)):
        ids = ids * (g // w) + coords[:, axis] // w
    return ids


def _block_mask(grid: tuple[int, ...], window: tuple[int, ...], prefix: int) -> torch.Tensor:
    """(N, N) booleans: a grid token sees its own window and the prefix; the prefix sees all."""
    ids = _window_ids(grid, window)
    allowed = torch.ones(prefix + ids.numel(), prefix + ids.numel(), dtype=torch.bool)
    allowed[prefix:, prefix:] = ids[:, None] == ids[None, :]
    return allowed


def _boxes(grid: tuple[int, ...], pool: int) -> list[torch.Tensor]:
    """The grid tokens each pooled token averages, row-major over the pooled grid."""
    coords = _grid_coords(grid)[0].long()
    box_of = torch.zeros(coords.shape[0], dtype=torch.long)
    for axis, g in enumerate(grid):
        box_of = box_of * (g // pool) + coords[:, axis] // pool
    return [torch.nonzero(box_of == box).flatten() for box in range(int(box_of.max()) + 1)]


def _qkv(prefix: int, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    shape = (2, prefix + CELLS, 2, 4)
    return tuple(torch.randn(shape, generator=generator, dtype=dtype) for _ in range(3))


# ------------------------------------------------------------------ the helpers


@pytest.mark.unit
def test_merging_the_windows_back_is_the_identity():
    x = torch.randn(2, CELLS, 3, 5)
    windows = partition_windows(x, GRID, WINDOW)
    assert windows.shape == (2 * 8, 6, 3, 5)
    assert torch.equal(merge_windows(windows, GRID, WINDOW, batch=2), x)


@pytest.mark.unit
def test_each_window_holds_its_box_of_the_grid_in_row_major_order():
    index = torch.arange(CELLS).reshape(1, CELLS)
    windows = partition_windows(index, GRID, WINDOW)
    ids = _window_ids(GRID, WINDOW)
    for window in range(windows.shape[0]):
        assert windows[window].tolist() == torch.nonzero(ids == window).flatten().tolist()


@pytest.mark.unit
@pytest.mark.parametrize("prefix", [0, 3])
def test_windowed_attention_is_full_attention_under_the_block_mask(prefix):
    q, k, v = _qkv(prefix)
    windowed = windowed_attention(q, k, v, grid=GRID, window=WINDOW, attend=_sdpa, prefix=prefix)
    mask = _block_mask(GRID, WINDOW, prefix)
    reference = F.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in (q, k, v)), attn_mask=mask
    ).transpose(1, 2)
    assert _worst(windowed, reference) < 1e-12


@pytest.mark.unit
def test_a_window_covering_the_grid_is_global_attention():
    q, k, v = _qkv(prefix=2, dtype=torch.float32)
    covering = windowed_attention(q, k, v, grid=GRID, window=(8, 8, 8), attend=_sdpa, prefix=2)
    assert resolve_window(GRID, (8, 8, 8)) == GRID
    assert torch.equal(covering, _sdpa(q, k, v))


@pytest.mark.unit
def test_a_window_that_does_not_tile_the_grid_is_refused():
    with pytest.raises(ValueError, match="does not tile"):
        resolve_window(GRID, (3, 3, 1))


@pytest.mark.unit
def test_windowed_attention_refuses_tokens_that_are_not_the_grid():
    q, k, v = _qkv(prefix=0)
    with pytest.raises(ValueError, match="whole grid"):
        windowed_attention(q[:, 1:], k[:, 1:], v[:, 1:], grid=GRID, window=WINDOW, attend=_sdpa)


@pytest.mark.unit
def test_pooling_averages_each_box():
    x = torch.randn(2, CELLS, 3, dtype=torch.float64)
    expected = torch.stack([x[:, box].mean(dim=1) for box in _boxes(GRID, 2)], dim=1)
    assert torch.allclose(pool_grid(x, GRID, 2), expected, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="does not tile"):
        pool_grid(x, GRID, 4)


@pytest.mark.unit
@pytest.mark.parametrize("prefix", [0, 5])
def test_attention_pairs_count_what_each_pattern_scores(prefix):
    tokens = prefix + CELLS
    assert attention_pairs(GRID, prefix) == tokens * tokens
    assert attention_pairs(GRID, prefix, WINDOW) == int(_block_mask(GRID, WINDOW, prefix).sum())
    assert attention_pairs(GRID, prefix, (9, 9, 9)) == tokens * tokens
    assert attention_pairs(GRID, prefix, kv_pool=2) == tokens * (prefix + CELLS // 8)


@pytest.mark.unit
def test_layout_windows_every_block_but_the_global_ones():
    assert block_attention_layout(4, 2, [1, 3], 2, 3) == [
        ((2, 2, 2), 1), (None, 2), ((2, 2, 2), 1), (None, 2),
    ]
    assert block_attention_layout(3, None, (), 1, 3) == [(None, 1)] * 3
    assert block_attention_layout(2, [4, 4, 2], (), 1, 3)[0] == ((4, 4, 2), 1)
    # Pooling with no windows pools every block.
    assert block_attention_layout(2, None, (), 4, 3) == [(None, 4)] * 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (dict(window=None, global_blocks=[0]), "only means something"),
        (dict(window=0), "positive"),
        (dict(window=[2, 2]), "positive"),
        (dict(global_blocks=[4]), "distinct block indices"),
        (dict(global_blocks=[1, 1]), "distinct block indices"),
        (dict(global_kv_pool=0), "positive int"),
    ],
)
def test_layout_refuses_settings_that_cannot_mean_anything(overrides, match):
    settings: dict[str, Any] = dict(window=2, global_blocks=(), global_kv_pool=1)
    settings.update(overrides)
    with pytest.raises(ValueError, match=match):
        block_attention_layout(
            4, settings["window"], settings["global_blocks"], settings["global_kv_pool"], 3
        )


# ------------------------------------------------------------------ the two attention layers


@pytest.mark.unit
def test_a_layer_is_windowed_or_pooled_not_both():
    with pytest.raises(ValueError, match="not both"):
        CommonSelfAttention(16, 2, window=WINDOW, kv_pool=2)
    with pytest.raises(ValueError, match="not both"):
        Dinov3SelfAttention(16, num_heads=2, window=WINDOW, kv_pool=2)


@pytest.mark.unit
def test_windowed_and_pooled_layers_need_the_grid():
    x = torch.randn(1, CELLS, 16)
    for layer in (CommonSelfAttention(16, 2, window=WINDOW), CommonSelfAttention(16, 2, kv_pool=2)):
        with pytest.raises(ValueError, match="grid"):
            layer(x)
    for layer in (
        Dinov3SelfAttention(16, num_heads=2, window=WINDOW),
        Dinov3SelfAttention(16, num_heads=2, kv_pool=2),
    ):
        with pytest.raises(ValueError, match="grid"):
            layer(x)


def _common_layer(**kwargs: Any) -> tuple[CommonSelfAttention, AxialRotaryEmbedding]:
    torch.manual_seed(0)
    layer = CommonSelfAttention(16, 2, backend="sdpa", **kwargs).double()
    return layer, AxialRotaryEmbedding(8, 3).double()


@pytest.mark.unit
def test_common_windowed_layer_is_full_attention_under_the_block_mask():
    layer, rotary = _common_layer(window=WINDOW)
    coords = _grid_coords(GRID)
    rope = rotary(coords)
    x = torch.randn(2, CELLS, 16, dtype=torch.float64)

    q, k, v = layer.qkv(x).reshape(2, CELLS, 3, 2, 8).unbind(2)
    q, k = rope(q), rope(k)
    attended = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=_block_mask(GRID, WINDOW, 0), scale=layer.scale,
    ).transpose(1, 2)
    expected = layer.proj(attended.reshape(2, CELLS, 16))

    actual = layer(x, rope=rope, grid=GRID)
    assert _worst(actual, expected) < 1e-12


@pytest.mark.unit
def test_common_pooled_layer_attends_to_box_means_rotated_at_their_centres():
    layer, rotary = _common_layer(kv_pool=2)
    coords = _grid_coords(GRID)
    boxes = _boxes(GRID, 2)
    x = torch.randn(2, CELLS, 16, dtype=torch.float64)

    q, k, v = layer.qkv(x).reshape(2, CELLS, 3, 2, 8).unbind(2)
    pooled_k, pooled_v = (torch.stack([t[:, box].mean(1) for box in boxes], 1) for t in (k, v))
    centres = torch.stack([coords[:, box].mean(1) for box in boxes], 1)
    q, pooled_k = rotary(coords)(q), rotary(centres)(pooled_k)
    attended = F.scaled_dot_product_attention(
        q.transpose(1, 2), pooled_k.transpose(1, 2), pooled_v.transpose(1, 2), scale=layer.scale
    ).transpose(1, 2)
    expected = layer.proj(attended.reshape(2, CELLS, 16))

    actual = layer(
        x, rope=rotary(coords), grid=GRID, pooled_rope=rotary(pool_grid(coords, GRID, 2))
    )
    assert _worst(actual, expected) < 1e-12


def _dinov3_rope(cls=RopePositionEmbedding3D, **kwargs: Any):
    rope = cls(16, num_heads=2, dtype=torch.float64, **kwargs)
    rope._init_weights()
    return rope


@pytest.mark.unit
def test_dinov3_windowed_layer_keeps_the_prefix_tokens_global():
    prefix = 3
    torch.manual_seed(0)
    layer = Dinov3SelfAttention(16, num_heads=2, qkv_bias=True, window=WINDOW).double()
    rope = _dinov3_rope()(D=GRID[0], H=GRID[1], W=GRID[2])
    x = torch.randn(2, prefix + CELLS, 16, dtype=torch.float64)

    q, k, v = layer.qkv(x).reshape(2, prefix + CELLS, 3, 2, 8).unbind(2)
    q, k = layer.apply_rope(q.transpose(1, 2), k.transpose(1, 2), rope)
    attended = F.scaled_dot_product_attention(
        q, k, v.transpose(1, 2), attn_mask=_block_mask(GRID, WINDOW, prefix), scale=layer.scale
    ).transpose(1, 2)
    expected = layer.proj(attended.reshape(2, prefix + CELLS, 16))

    actual = layer(x, rope=rope, grid=GRID)
    assert _worst(actual, expected) < 1e-12


@pytest.mark.unit
def test_dinov3_pooled_layer_attends_to_box_means_and_the_unpooled_prefix():
    prefix = 3
    torch.manual_seed(0)
    layer = Dinov3SelfAttention(16, num_heads=2, qkv_bias=True, kv_pool=2).double()
    embed = _dinov3_rope()
    rope = embed(D=GRID[0], H=GRID[1], W=GRID[2])
    pooled_rope = embed(D=GRID[0] // 2, H=GRID[1] // 2, W=GRID[2] // 2)
    x = torch.randn(2, prefix + CELLS, 16, dtype=torch.float64)

    q, k, v = layer.qkv(x).reshape(2, prefix + CELLS, 3, 2, 8).unbind(2)
    pooled_k, pooled_v = (
        torch.cat(
            [
                t[:, :prefix],
                torch.stack([t[:, prefix + box].mean(1) for box in _boxes(GRID, 2)], 1),
            ],
            dim=1,
        )
        for t in (k, v)
    )
    q = rope_apply_to_suffix(q.transpose(1, 2), rope)
    pooled_k = rope_apply_to_suffix(pooled_k.transpose(1, 2), pooled_rope)
    attended = F.scaled_dot_product_attention(
        q, pooled_k, pooled_v.transpose(1, 2), scale=layer.scale
    ).transpose(1, 2)
    expected = layer.proj(attended.reshape(2, prefix + CELLS, 16))

    actual = layer(x, rope=rope, grid=GRID, pooled_rope=pooled_rope)
    assert _worst(actual, expected) < 1e-12


@pytest.mark.unit
@pytest.mark.parametrize("cls", [RopePositionEmbedding3D, RopePositionEmbedding3DSuperposition])
@pytest.mark.parametrize("mode", ["separate", "max", "min"])
def test_dinov3_pooled_grid_tables_sit_at_the_centres_of_their_boxes(cls, mode):
    """What lets a pooled layer rotate its keys with the pooled grid's own rope tables.

    Angles are linear in the coordinates, so a key at the centre of a box has the mean of the
    box's angles. Every period is stretched so every angle stays inside (-pi, pi) and `atan2`
    recovers it exactly; otherwise wrapped angles could not be averaged.
    """
    embed = _dinov3_rope(cls, normalize_coords=mode)
    with torch.no_grad():
        for name, buffer in embed.named_buffers():
            if "periods" in name:
                buffer.mul_(1000.0)
        if cls is RopePositionEmbedding3DSuperposition:
            embed.depth_scale.fill_(1.0)  # zero at init, which would hide depth from the check

    fine = torch.atan2(*embed(D=GRID[0], H=GRID[1], W=GRID[2]))
    pooled = torch.atan2(*embed(D=GRID[0] // 2, H=GRID[1] // 2, W=GRID[2] // 2))
    expected = pool_grid(fine.unsqueeze(0), GRID, 2)[0]
    assert _worst(pooled, expected) < 1e-12


# ------------------------------------------------------------------ the models


def _vit(**overrides: Any) -> ViT3D:
    kwargs: dict[str, Any] = dict(
        img_size=(32, 32, 16), patch_size=(8, 8, 8), in_channels=1,
        embed_dim=32, depth=4, num_heads=4, attention_backend="sdpa",
    )
    kwargs.update(overrides)
    return ViT3D(**kwargs).eval()


def _dino(**overrides: Any) -> DinoVisionTransformer3D:
    kwargs: dict[str, Any] = dict(
        img_size=32, patch_size=8, in_chans=1, embed_dim=32, depth=4, num_heads=2,
        n_storage_tokens=2, layerscale_init=1.0e-5, pos_embed_rope_dtype="fp32",
    )
    kwargs.update(overrides)
    return DinoVisionTransformer3D(**kwargs).eval()


def _counted_forward_flops(model: torch.nn.Module, volume: torch.Tensor) -> int:
    # SDPA's math backend, because the flop counter sees attention only as the matmuls it
    # decomposes into on CPU.
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter, torch.no_grad():
        model.patch_features(volume)
    return counter.get_total_flops()


PATTERNS = [
    dict(attn_window=2),
    dict(attn_window=2, attn_global_blocks=[1, 3]),
    dict(attn_window=[2, 2, 1], attn_global_blocks=[0], attn_global_kv_pool=2),
    dict(attn_global_kv_pool=2),
]


@pytest.mark.unit
def test_vit3d_with_windows_covering_its_grid_is_the_global_model():
    reference = _vit()
    covering = _vit(attn_window=[4, 4, 2], attn_global_blocks=[1])
    assert covering.state_dict().keys() == reference.state_dict().keys()
    covering.load_state_dict(reference.state_dict())
    volume = torch.randn(2, 1, 32, 32, 16)
    assert torch.equal(covering.patch_features(volume)[0], reference.patch_features(volume)[0])


@pytest.mark.unit
@pytest.mark.parametrize("attention", PATTERNS)
def test_vit3d_patterns_train_and_change_the_features(attention):
    reference = _vit()
    patterned = _vit(**attention)
    patterned.load_state_dict(reference.state_dict())
    volume = torch.randn(2, 1, 32, 32, 16)
    features = patterned.patch_features(volume)[0]
    assert not torch.allclose(features, reference.patch_features(volume)[0])
    features.square().mean().backward()
    assert all(p.grad is not None for p in patterned.parameters())


@pytest.mark.unit
@pytest.mark.parametrize("attention", PATTERNS)
def test_vit3d_flops_follow_the_attention_pattern(attention):
    reference, patterned = _vit(), _vit(**attention)
    volume = torch.randn(1, 1, 32, 32, 16)
    saved = _counted_forward_flops(reference, volume) - _counted_forward_flops(patterned, volume)
    shape = (1, 32, 32, 16)
    assert saved > 0
    assert patterned.flops(shape) == reference.flops(shape) - saved


@pytest.mark.unit
def test_vit3d_masked_encoding_refuses_windowed_blocks_but_still_serves_global_ones():
    tokens, coords = _vit().embed(torch.randn(1, 1, 32, 32, 16))
    with pytest.raises(ValueError, match="Masked encoding"):
        _vit(attn_window=2).encode(tokens[:, :10], coords[:, :10])
    assert _vit().encode(tokens[:, :10], coords[:, :10]).shape == (1, 10, 32)


@pytest.mark.unit
def test_vit3d_refuses_windows_and_pools_that_do_not_tile_its_grid():
    with pytest.raises(ValueError, match="does not tile"):
        _vit(attn_window=3)
    with pytest.raises(ValueError, match="does not tile"):
        _vit(attn_global_kv_pool=4)  # the grid is 4 x 4 x 2


@pytest.mark.unit
def test_dinov3_with_windows_covering_its_grid_is_the_global_model():
    reference = _dino()
    covering = _dino(attn_window=4, attn_global_blocks=[2])
    assert covering.state_dict().keys() == reference.state_dict().keys()
    covering.load_state_dict(reference.state_dict())
    volume = torch.randn(2, 1, 32, 32, 32)
    assert torch.equal(covering.patch_features(volume)[0], reference.patch_features(volume)[0])


@pytest.mark.unit
@pytest.mark.parametrize("attention", PATTERNS)
def test_dinov3_patterns_train_and_change_the_features(attention):
    reference = _dino()
    patterned = _dino(**attention)
    patterned.load_state_dict(reference.state_dict())
    volume = torch.randn(2, 1, 32, 32, 32)
    features = patterned.patch_features(volume)[0]
    assert not torch.allclose(features, reference.patch_features(volume)[0])
    features.square().mean().backward()
    missing = [name for name, p in patterned.named_parameters() if p.grad is None]
    # The mask token only reaches the loss through a zero-weighted term, as in the global model.
    assert missing in ([], ["mask_token"])


@pytest.mark.unit
@pytest.mark.parametrize("attention", PATTERNS)
def test_dinov3_flops_follow_the_attention_pattern(attention):
    reference, patterned = _dino(), _dino(**attention)
    volume = torch.randn(1, 1, 32, 32, 32)
    saved = _counted_forward_flops(reference, volume) - _counted_forward_flops(patterned, volume)
    shape = (1, 32, 32, 32)
    assert saved > 0
    assert patterned.flops(shape) == reference.flops(shape) - saved


@pytest.mark.unit
def test_dinov3_multi_crop_path_windows_each_crop_on_its_own_grid():
    # 32 -> a 4^3 grid, which 2-windows really split; 16 -> 2^3, which one window covers.
    model = _dino(attn_window=2, attn_global_blocks=[3], attn_global_kv_pool=2)
    large, small = torch.randn(1, 1, 32, 32, 32), torch.randn(1, 1, 16, 16, 16)
    together = model.forward_features([large, small], [None, None])
    alone = [model.forward_features(large), model.forward_features(small)]
    for joint, single in zip(together, alone, strict=True):
        assert torch.allclose(
            joint["x_norm_patchtokens"], single["x_norm_patchtokens"], rtol=0, atol=1e-5
        )


@pytest.mark.unit
def test_dinov3_intermediate_layers_run_the_same_pattern():
    model = _dino(attn_window=2, attn_global_blocks=[3], attn_global_kv_pool=2)
    volume = torch.randn(1, 1, 32, 32, 32)
    (last,) = model.get_intermediate_layers(volume, n=1)
    assert torch.allclose(last, model.patch_features(volume)[0], rtol=0, atol=1e-5)


@pytest.mark.unit
def test_dinov3_patterns_run_under_stochastic_depth():
    # A batch of 4 at drop_path 0.5 takes the subset path, which reaches attention through
    # `forward_list` rather than `forward`.
    model = _dino(
        attn_window=2, attn_global_blocks=[3], attn_global_kv_pool=2, drop_path_rate=0.5
    ).train()
    model.forward_features(torch.randn(4, 1, 32, 32, 32))["x_norm_patchtokens"].mean().backward()


@pytest.mark.unit
def test_dinov3_pooling_refuses_rope_coordinate_augmentation():
    with pytest.raises(ValueError, match="coordinate augmentation"):
        _dino(attn_global_kv_pool=2, pos_embed_rope_rescale_coords=2.0)
    _dino(attn_window=2, pos_embed_rope_rescale_coords=2.0)  # windows alone are unaffected


@pytest.mark.unit
def test_dinov3_refuses_a_crop_its_windows_cannot_tile():
    model = _dino(attn_window=3)
    with pytest.raises(ValueError, match="does not tile"):
        model.patch_features(torch.randn(1, 1, 32, 32, 32))
    with pytest.raises(ValueError, match="does not tile"):
        model.flops((1, 32, 32, 32))


# ------------------------------------------------------------------ sliding windows
#
# FlexAttention's CPU path is an unfused reference that applies the per-pair predicate everywhere
# and ignores the block lists, while the fused CUDA kernel trusts the lists: it skips a key block
# that is not listed and does not evaluate the predicate in one listed as full. So correctness on
# the GPU needs both halves checked -- the predicate against a reference here, the lists against
# the predicate here -- and the fused kernel itself in `test_window_attention_gpu.py`.

# Grids the preferred 8 x 8 x 4 tile divides, so the kernel works on whole 256-token blocks. The
# larger has more key blocks (16) than any query tile's box holds (6): the case that exposed
# BlockMask's one-column-per-key-block layout.
SLIDING_CASES = [
    ((8, 8, 16), (4, 4, 8), 0),
    ((8, 8, 16), (4, 4, 8), 3),
    ((8, 8, 16), (3, 5, 6), 3),  # odd and even sizes, and none of them tiles the grid
    ((8, 8, 16), (8, 2, 16), 0),  # the whole extent on two axes: slides along one only
    ((8, 16, 32), (4, 4, 8), 3),
    ((8, 16, 32), (3, 5, 6), 0),
]


def _centred_mask(grid: tuple[int, ...], window: tuple[int, ...]) -> torch.Tensor:
    """(cells, cells) booleans, row-major: the window centred on each token, shifted inward at the
    faces -- written out independently of the module's helpers."""
    coords = _grid_coords(grid)[0].long()
    allowed = torch.ones(coords.shape[0], coords.shape[0], dtype=torch.bool)
    for axis, (extent, size) in enumerate(zip(grid, window, strict=True)):
        start = (coords[:, axis] - size // 2).clamp(0, extent - size)
        allowed &= (coords[None, :, axis] >= start[:, None]) & (
            coords[None, :, axis] < start[:, None] + size
        )
    return allowed


def _sliding_mask(grid: tuple[int, ...], window: tuple[int, ...], prefix: int) -> torch.Tensor:
    """(N, N) booleans, row-major: a token sees every key tile that the centred window of any token
    in its own tile touches, and the prefix; the prefix sees everything.

    Derived from the per-token windows by grouping them by tile, rather than from the module's
    per-axis box arithmetic; only the tile shape is taken from the module.
    """
    centred = _centred_mask(grid, window)
    coords = _grid_coords(grid)[0].long()
    tile = sliding_tile(grid)
    tile_of = torch.zeros(coords.shape[0], dtype=torch.long)
    for axis, (extent, t) in enumerate(zip(grid, tile, strict=True)):
        tile_of = tile_of * (extent // t) + coords[:, axis] // t
    tiles = int(tile_of.max()) + 1
    rows, columns = (i.expand_as(centred)[centred] for i in (tile_of[:, None], tile_of[None, :]))
    touched = torch.zeros(tiles, tiles, dtype=torch.bool)
    touched[rows, columns] = True
    full = torch.ones(prefix + coords.shape[0], prefix + coords.shape[0], dtype=torch.bool)
    full[prefix:, prefix:] = touched[tile_of[:, None], tile_of[None, :]]
    return full


@pytest.mark.unit
def test_windows_are_centred_and_shifted_inward_at_the_faces():
    assert [window_start(i, 8, 4) for i in range(8)] == [0, 0, 0, 1, 2, 3, 4, 4]
    assert [window_start(i, 8, 3) for i in range(8)] == [0, 0, 1, 2, 3, 4, 5, 5]


@pytest.mark.unit
@pytest.mark.parametrize("group", [PREFIX_GROUP, 1])  # one prefix copy per tile: several copies
@pytest.mark.parametrize(("grid", "window", "prefix"), SLIDING_CASES)
def test_sliding_window_attention_is_full_attention_under_the_tile_mask(
    grid, window, prefix, group, monkeypatch
):
    monkeypatch.setattr(window_attention, "PREFIX_GROUP", group)
    generator = torch.Generator().manual_seed(0)
    shape = (1, prefix + math.prod(grid), 2, 4)
    q, k, v = (torch.randn(shape, generator=generator, dtype=torch.float64) for _ in range(3))
    actual = sliding_window_attention(
        q, k, v, grid=grid, window=window, attend=_sdpa, scale=4**-0.5, prefix=prefix
    )
    reference = F.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in (q, k, v)), attn_mask=_sliding_mask(grid, window, prefix)
    ).transpose(1, 2)
    assert _worst(actual, reference) < 1e-12


def _listed(counts: torch.Tensor, indices: torch.Tensor, columns: int) -> torch.Tensor:
    """(query blocks, key blocks) booleans: which key blocks each query block's list names."""
    counts, indices = counts[0, 0], indices[0, 0]
    listed = torch.zeros(counts.shape[0], columns, dtype=torch.bool)
    for row, count in enumerate(counts.tolist()):
        listed[row, indices[row, :count].long()] = True
    return listed


@pytest.mark.unit
@pytest.mark.parametrize("group", [PREFIX_GROUP, 1])  # one prefix copy per tile: several copies
@pytest.mark.parametrize(("grid", "window", "prefix"), SLIDING_CASES)
def test_sliding_block_lists_are_exactly_what_the_predicate_needs(grid, window, prefix, group):
    """Unlisted blocks hold no visible pair, full blocks nothing else, partial ones some of each."""
    mask = sliding_block_mask(grid, window, prefix, torch.device("cpu"), group)
    cells, volume = math.prod(grid), math.prod(sliding_tile(grid))
    copies = prefix_copies(grid, prefix, group)
    keys = copies * volume + cells
    dense = mask.mask_mod(0, 0, torch.arange(cells)[:, None], torch.arange(keys)[None, :])
    blocks = dense.reshape(cells // volume, volume, keys // volume, volume).transpose(1, 2)
    columns = keys // volume
    # One column per key block, which BlockMask's transposition for the backward pass relies on.
    assert mask.kv_indices.shape[-1] == mask.full_kv_indices.shape[-1] == columns
    partial = _listed(mask.kv_num_blocks, mask.kv_indices, columns)
    full = _listed(mask.full_kv_num_blocks, mask.full_kv_indices, columns)

    assert not (partial & full).any()
    assert torch.equal(partial | full, blocks.any(-1).any(-1))
    assert torch.equal(full, blocks.all(-1).all(-1))
    # Every grid block is full; the only one the kernel evaluates the predicate in is the copy of
    # the prefix block that the query tile's group attends to.
    only_prefix = torch.zeros_like(partial)
    if prefix:
        only_prefix[torch.arange(partial.shape[0]), torch.arange(partial.shape[0]) // group] = True
    assert torch.equal(partial, only_prefix)


@pytest.mark.unit
@pytest.mark.parametrize(("grid", "window", "prefix"), SLIDING_CASES)
def test_every_token_sees_at_least_its_centred_window_and_pairs_are_counted(grid, window, prefix):
    sliding = _sliding_mask(grid, window, prefix)
    centred = _centred_mask(grid, window)
    assert not (centred & ~sliding[prefix:, prefix:]).any()
    assert attention_pairs(grid, prefix, window, window_mode="sliding") == int(sliding.sum())


@pytest.mark.unit
def test_a_sliding_window_covering_the_grid_is_global_attention():
    q, k, v = (t.float() for t in _qkv(prefix=2))
    covering = sliding_window_attention(
        q, k, v, grid=GRID, window=(9, 9, 9), attend=_sdpa, scale=4**-0.5, prefix=2
    )
    assert torch.equal(covering, _sdpa(q, k, v))


@pytest.mark.unit
def test_vit3d_with_sliding_windows_covering_its_grid_is_the_global_model():
    reference = _vit()
    covering = _vit(attn_window=[4, 4, 2], attn_window_mode="sliding", attn_global_blocks=[1])
    covering.load_state_dict(reference.state_dict())
    volume = torch.randn(2, 1, 32, 32, 16)
    assert torch.equal(covering.patch_features(volume)[0], reference.patch_features(volume)[0])


@pytest.mark.unit
def test_sliding_windows_need_not_tile_the_grid():
    with pytest.raises(ValueError, match="does not tile"):
        _vit(attn_window=3)
    model = _vit(attn_window=3, attn_window_mode="sliding")
    with torch.no_grad():
        assert model.patch_features(torch.randn(1, 1, 32, 32, 16))[0].shape == (1, 32, 32)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("build", "volume", "prefix"),
    [(_vit, (1, 1, 32, 32, 16), 0), (_dino, (1, 1, 32, 32, 32), 3)],  # _dino: CLS + 2 storage
)
def test_sliding_models_run_and_cost_what_their_windows_attend_to(build, volume, prefix):
    layout: dict[str, Any] = dict(attn_window=2, attn_global_blocks=[3], attn_global_kv_pool=2)
    sliding = build(**layout, attn_window_mode="sliding")
    blocky = build(**layout)
    blocky.load_state_dict(sliding.state_dict())
    # The three windowed blocks attend to their tiles' boxes rather than to 2^3 blocks.
    grid = tuple(side // 8 for side in volume[-3:])
    extra = attention_pairs(grid, prefix, (2, 2, 2), window_mode="sliding") - attention_pairs(
        grid, prefix, (2, 2, 2)
    )
    assert sliding.flops(volume[1:]) - blocky.flops(volume[1:]) == 3 * 4 * extra * 32
    x = torch.randn(volume)
    with torch.no_grad():
        features = sliding.patch_features(x)[0]
        assert features.shape == blocky.patch_features(x)[0].shape
        assert not torch.allclose(features, blocky.patch_features(x)[0])


@pytest.mark.unit
def test_an_unknown_window_mode_is_refused():
    with pytest.raises(ValueError, match="window mode"):
        _vit(attn_window=2, attn_window_mode="shifted")
    with pytest.raises(ValueError, match="window mode"):
        _dino(attn_window=2, attn_window_mode="shifted")
