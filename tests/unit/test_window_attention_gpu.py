"""Sliding-window attention on the fused FlexAttention kernel, which only a CUDA device runs.

The CPU tests check the per-pair predicate and the block lists separately; the fused kernel is
where they meet -- it skips unlisted key blocks and does not evaluate the predicate in full ones,
which with windows sliding tile by tile is every grid block -- and it is also the only place the
backward pass runs. So both directions are checked here against
full attention under the equivalent dense mask.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from layers.common import window_attention
from layers.common.window_attention import PREFIX_GROUP, sliding_tile, sliding_window_attention
from models.dinov3_vit3d import DinoVisionTransformer3D

pytestmark = [
    pytest.mark.gpu_dist,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

# The larger grid has more key blocks than any query block has candidates, which is what BlockMask's
# one-column-per-key-block layout is about; the smaller one cannot show it.
CASES = [
    ((8, 8, 16), (4, 4, 8), 0),
    ((8, 8, 16), (4, 4, 8), 5),
    ((8, 8, 16), (3, 5, 6), 5),
    ((8, 16, 32), (4, 4, 8), 5),
    ((8, 16, 32), (3, 5, 6), 0),
]


def _dense_mask(grid: tuple[int, ...], window: tuple[int, ...], prefix: int) -> torch.Tensor:
    """Row-major booleans: a token sees every key tile the centred window of any token in its own
    tile touches, plus the prefix; derived from the per-token windows, grouped by tile."""
    cells = math.prod(grid)
    axes = [torch.arange(g) for g in grid]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(grid))
    centred = torch.ones(cells, cells, dtype=torch.bool)
    for axis, (extent, size) in enumerate(zip(grid, window, strict=True)):
        start = (coords[:, axis] - size // 2).clamp(0, extent - size)
        centred &= (coords[None, :, axis] >= start[:, None]) & (
            coords[None, :, axis] < start[:, None] + size
        )
    tile_of = torch.zeros(cells, dtype=torch.long)
    for axis, (extent, t) in enumerate(zip(grid, sliding_tile(grid), strict=True)):
        tile_of = tile_of * (extent // t) + coords[:, axis] // t
    tiles = int(tile_of.max()) + 1
    rows, columns = (i.expand_as(centred)[centred] for i in (tile_of[:, None], tile_of[None, :]))
    touched = torch.zeros(tiles, tiles, dtype=torch.bool)
    touched[rows, columns] = True
    full = torch.ones(prefix + cells, prefix + cells, dtype=torch.bool)
    full[prefix:, prefix:] = touched[tile_of[:, None], tile_of[None, :]]
    return full.cuda()


@pytest.mark.parametrize("group", [PREFIX_GROUP, 1])  # one prefix copy per tile: several copies
@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 2e-3), (torch.bfloat16, 3e-2)])
@pytest.mark.parametrize(("grid", "window", "prefix"), CASES)
def test_fused_kernel_matches_masked_attention_forward_and_backward(
    grid, window, prefix, dtype, tolerance, group, monkeypatch
):
    monkeypatch.setattr(window_attention, "PREFIX_GROUP", group)
    torch.manual_seed(0)
    shape = (2, prefix + math.prod(grid), 4, 64)
    q, k, v = (torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3))
    scale = 64**-0.5

    def attend(query, key, value):
        query, key, value = (t.transpose(1, 2) for t in (query, key, value))
        return F.scaled_dot_product_attention(query, key, value, scale=scale).transpose(1, 2)

    actual = sliding_window_attention(
        q, k, v, grid=grid, window=window, attend=attend, scale=scale, prefix=prefix
    )
    reference = F.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in (q, k, v)),
        attn_mask=_dense_mask(grid, window, prefix),
        scale=scale,
    ).transpose(1, 2)
    assert (actual - reference).abs().max().item() < tolerance

    upstream = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (q, k, v), upstream)
    expected = torch.autograd.grad(reference, (q, k, v), upstream)
    for name, got, want in zip("qkv", grads, expected, strict=True):
        worst = (got - want).abs().max().item()
        assert worst < tolerance * max(1.0, want.abs().max().item()), f"d{name} off by {worst}"


def test_a_sliding_model_compiles_to_what_it_computes_eagerly():
    """The enclosing `torch.compile` traces the block mask and the kernel into its own graph.

    A 16^3 grid of patches: 16 key blocks of 8 x 8 x 4 tokens, more than a tile's box holds (12).
    """
    torch.manual_seed(0)
    model = DinoVisionTransformer3D(
        img_size=128, patch_size=8, in_chans=1, embed_dim=64, depth=2, num_heads=2,
        n_storage_tokens=2, pos_embed_rope_dtype="fp32",
        attn_window=4, attn_window_mode="sliding", attn_global_blocks=[1], attn_global_kv_pool=2,
    ).cuda()
    volume = torch.randn(2, 1, 128, 128, 128, device="cuda")
    eager = model.patch_features(volume)[0]
    compiled = torch.compile(model.patch_features)(volume)[0]
    assert (eager - compiled).abs().max().item() < 1e-3
    compiled.square().mean().backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if n != "mask_token")
