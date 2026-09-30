"""Attention over a patch grid for less than the square of its size: local windows, pooled keys.

Global self-attention over a D x H x W patch grid scores every pair of tokens, (DHW)^2 per layer,
and at a 1024-cube crop of 16-voxel patches that is ~83% of a ViT-L training step
(`experiments/large_inputs`). This module holds the two cheaper patterns the patch-grid ViTs here
can run instead, shared by `layers.common.attention.SelfAttention` (ViT3D) and
`layers.dinov3.attention.SelfAttention` (the DINOv3 port):

  - **Windows.** The grid is cut into non-overlapping `window`-shaped boxes and each grid token
    attends only to the tokens of its own box, so a layer costs DHW x prod(window) -- linear in the
    volume. ViTDet and SAM's image encoder run pretrained global-attention ViTs this way, with a
    few global blocks left in to carry information between windows.
    Windows come in two kinds (`WINDOW_MODES`). **Block** windows tile the grid, so a token at a
    window's edge sees context on one side only and the model sees fixed seams between windows;
    they are a reshape around the layer's own kernel. **Sliding** windows move tile by tile: every
    8x8x4 tile of tokens attends to the box of tiles that the windows centred on its tokens cover --
    itself and about `window / 2` tokens either side -- so no token sits at the edge of its own
    window and there are no fixed seams. They run on FlexAttention (its FLASH backend where
    FlashAttention-4 is installed), where boxes of whole tiles make every key block wholly inside or
    outside a window, so the kernel runs at dense speed. (Windows
    centred on each token instead leave most key blocks partly inside and cost ~18x as much;
    `experiments/large_inputs`.)
  - **Pooled keys and values.** Every token attends to the whole grid, but through keys and values
    averaged over `pool`-sided boxes (PVT's spatial-reduction attention), which divides a global
    layer's cost by pool^rank.

Tokens in front of the grid -- DINOv3's CLS and storage tokens -- are the *prefix*. They have no
place on the grid, so no window contains them: in a windowed layer they attend to every token, and
every grid token attends to them as well as to its own window. That keeps them a global channel
through every layer for the price of a few extra keys per window.

Nothing here computes attention. The kernel is the caller's `attend`, which takes and returns
(B, N, heads, head_dim), so each layer keeps its own SDPA / FlashAttention-4 choice and
`engine.mfu`'s switch onto a countable kernel reaches windowed layers exactly as it reaches global
ones. Rotary position embeddings are the caller's too, and are applied *before* the tokens are
windowed: each token is rotated for its place on the whole grid, so attention inside a window sees
the same relative positions it would have seen globally.

Tokens are in the patch embedding's row-major order throughout -- the order
`BaseModel.patch_features` promises -- with the prefix, if any, in front.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

#: The attention kernel a layer hands in: (B, N, heads, head_dim) query, key and value -> output.
Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

#: `"block"`: windows tile the grid. `"sliding"`: windows centred on each tile, moving tile by tile.
WINDOW_MODES = ("block", "sliding")


def block_attention_layout(
    depth: int,
    window: int | Sequence[int] | None,
    global_blocks: Sequence[int],
    global_kv_pool: int,
    spatial_rank: int,
) -> list[tuple[tuple[int, ...] | None, int]]:
    """Each block's `(window, kv_pool)`, from a model's three attention settings.

    `window` unset leaves every block global, which is the default and what every config written
    before this module means. Set, it makes every block attend within `window` (an int for a cube,
    or one extent per axis, in patches) except the blocks listed in `global_blocks`. Global blocks
    -- all of them when `window` is unset -- average their keys and values over
    `global_kv_pool`-sided boxes when that is above 1.

    Shared by every model that exposes these settings so they validate and mean the same thing
    everywhere; the checks are on configuration, which arrives straight from a file.
    """
    if window is None:
        if global_blocks:
            raise ValueError(
                f"attn_global_blocks {list(global_blocks)} names blocks to keep global, which only "
                "means something when attn_window is set; without it every block is already global"
            )
        shape = None
    else:
        shape = (window,) * spatial_rank if isinstance(window, int) else tuple(window)
        if len(shape) != spatial_rank or any(not isinstance(w, int) or w < 1 for w in shape):
            raise ValueError(
                f"attn_window must be a positive int or {spatial_rank} positive ints (patches per "
                f"axis), got {window!r}"
            )
    if not isinstance(global_kv_pool, int) or global_kv_pool < 1:
        raise ValueError(f"attn_global_kv_pool must be a positive int, got {global_kv_pool!r}")
    listed = list(global_blocks)
    if len(set(listed)) != len(listed) or any(not 0 <= index < depth for index in listed):
        raise ValueError(
            f"attn_global_blocks must be distinct block indices in [0, {depth}), got {listed}"
        )
    return [
        (None, global_kv_pool) if shape is None or index in listed else (shape, 1)
        for index in range(depth)
    ]


def resolve_window(grid: Sequence[int], window: Sequence[int]) -> tuple[int, ...]:
    """The window used on `grid`: clamped to the grid on every axis, and required to tile it.

    Clamped because a window at least as large as the crop can only mean "the whole crop". That is
    what makes a 16^3-token window on a 256-cube crop of 16-voxel patches exactly global attention,
    and one config valid at every size of a sweep. Required to tile because a partial window at the
    far face would have to be dropped or padded with tokens that attend to nothing, and either
    changes what the layer computes without changing a shape.
    """
    if len(window) != len(grid):
        raise ValueError(f"window {tuple(window)} and token grid {tuple(grid)} differ in rank")
    resolved = tuple(min(w, g) for w, g in zip(window, grid, strict=True))
    if any(g % w for w, g in zip(resolved, grid, strict=True)):
        raise ValueError(
            f"attention window {tuple(window)} does not tile the token grid {tuple(grid)}: every "
            "axis of the grid must be a whole number of windows (or no larger than one)"
        )
    return resolved


def partition_windows(x: torch.Tensor, grid: Sequence[int], window: Sequence[int]) -> torch.Tensor:
    """(B, prod(grid), *rest) -> (B * windows, prod(window), *rest), one row per window.

    Windows come out batch-major (all of sample 0's, then sample 1's), each window's tokens in
    row-major order within it.
    """
    batch, rank, rest = x.shape[0], len(grid), x.shape[2:]
    counts = [g // w for g, w in zip(grid, window, strict=True)]
    # (B, n_1, w_1, ..., n_rank, w_rank, *rest) -> (B, n_1, ..., n_rank, w_1, ..., w_rank, *rest)
    x = x.reshape(batch, *(n for pair in zip(counts, window, strict=True) for n in pair), *rest)
    order = [0, *range(1, 2 * rank, 2), *range(2, 2 * rank + 1, 2), *range(2 * rank + 1, x.dim())]
    return x.permute(order).reshape(batch * math.prod(counts), math.prod(window), *rest)


def merge_windows(
    x: torch.Tensor, grid: Sequence[int], window: Sequence[int], batch: int
) -> torch.Tensor:
    """The inverse of `partition_windows`.

    (B * windows, prod(window), *rest) -> (B, prod(grid), *rest).
    """
    rank, rest = len(grid), x.shape[2:]
    counts = [g // w for g, w in zip(grid, window, strict=True)]
    # (B, n_1, ..., n_rank, w_1, ..., w_rank, *rest) -> (B, n_1, w_1, ..., n_rank, w_rank, *rest)
    x = x.reshape(batch, *counts, *window, *rest)
    order = [0, *(i for axis in range(rank) for i in (1 + axis, 1 + rank + axis))]
    order += list(range(1 + 2 * rank, x.dim()))
    return x.permute(order).reshape(batch, math.prod(grid), *rest)


def windowed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    grid: Sequence[int],
    window: Sequence[int],
    attend: Attend,
    prefix: int = 0,
) -> torch.Tensor:
    """Attention in which each grid token sees only its own window, plus the prefix tokens.

    (B, prefix + prod(grid), heads, head_dim) query, key and value -> the same shape. The prefix
    tokens see everything. Exactly full attention under the corresponding block mask, computed as
    one kernel call over the batch of windows rather than as a masked (N, N) problem.
    """
    window = resolve_window(grid, window)
    if query.shape[1] != prefix + math.prod(grid):
        raise ValueError(
            f"{query.shape[1]} tokens cannot be {prefix} prefix tokens followed by the token grid "
            f"{tuple(grid)}: windowed attention needs the whole grid, in row-major order"
        )
    if window == tuple(grid):
        # One window holding the whole grid is global attention, and is run as such: the same
        # single kernel call a layer without a window makes, not a reshaped copy of it.
        return attend(query, key, value)

    batch = query.shape[0]
    grid_query, grid_key, grid_value = (
        partition_windows(t[:, prefix:], grid, window) for t in (query, key, value)
    )
    if prefix:
        windows = grid_query.shape[0] // batch
        # Every window's keys and values start with the prefix tokens'.
        grid_key, grid_value = (
            torch.cat((t[:, :prefix].repeat_interleave(windows, dim=0), windowed), dim=1)
            for t, windowed in ((key, grid_key), (value, grid_value))
        )
    attended = merge_windows(attend(grid_query, grid_key, grid_value), grid, window, batch)
    if not prefix:
        return attended
    # The prefix tokens belong to no window: they read the whole sequence, as in a global layer.
    return torch.cat((attend(query[:, :prefix], key, value), attended), dim=1)


def check_pool(grid: Sequence[int], pool: int) -> None:
    """Refuse a pool whose boxes do not tile `grid`.

    A partial box at the far face would average fewer tokens than the rest and put its pooled key
    somewhere other than the centre the rotary tables assume, with no shape to show it.
    """
    if any(g % pool for g in grid):
        raise ValueError(
            f"key/value pool {pool} does not tile the token grid {tuple(grid)}: every axis of "
            "the grid must be a whole number of pooling boxes"
        )


def check_window_mode(mode: str) -> None:
    """Refuse a window mode this module does not implement; it arrives straight from a config."""
    if mode not in WINDOW_MODES:
        raise ValueError(f"window mode must be one of {WINDOW_MODES}, got {mode!r}")


# ------------------------------------------------------------------ sliding windows

#: The tile of the grid a sliding window moves by, per spatial rank: 256 tokens, one FlexAttention
#: block of queries and one of keys. 256 because FlexAttention's FLASH backend -- FlashAttention-4
#: underneath, ~1.8x its Triton kernel on a B300 here -- takes block sizes in multiples of 256, and
#: 256 is not a cube, so one axis is shorter.
PREFERRED_TILE = {1: (256,), 2: (16, 16), 3: (8, 8, 4)}

#: Query tiles that share one copy of the prefix's key block. Every query attends to the prefix, and
#: the backward pass accumulates a key block's gradient over the query blocks that attend to it one
#: after another, so a single prefix block attended by every tile is a serial tail -- ~+40% on a
#: layer at a 64^3 grid. One copy per this many tiles attends like any other key block.
PREFIX_GROUP = 64


def sliding_tile(grid: Sequence[int]) -> tuple[int, ...]:
    """The tile the sliding kernel cuts `grid` into: the preferred tile, shrunk where it would not
    divide the grid. The shrunk tiles only arise on grids too small to need windows at all."""
    preferred = PREFERRED_TILE[len(grid)]
    return tuple(math.gcd(g, t) for g, t in zip(grid, preferred, strict=True))


def window_start(position: torch.Tensor | int, extent: int, size: int) -> torch.Tensor | int:
    """First token of the `size`-wide window centred on `position`, on an axis of `extent`.

    Centred on the token -- `size // 2` before it, the rest after -- and shifted inward at the
    faces, so every window holds exactly `size` tokens, as neighbourhood attention defines it.
    Takes a tensor of positions (the kernel's predicate) or one int (the block lists).
    """
    if isinstance(position, int):
        return min(max(position - size // 2, 0), extent - size)
    return torch.clamp(position - size // 2, min=0, max=extent - size)


def _axis_tile_boxes(extent: int, size: int, tile: int) -> list[tuple[int, int]]:
    """For every tile along one axis, the first and last key tile of its window.

    A tile's window is the union of the `size`-wide windows centred on each of its tokens
    (`window_start`), rounded out to whole tiles. `window_start` never decreases along an axis, so
    the union is [start(first token), start(last token) + size): the tile itself and about
    `size // 2` tokens either side, fewer at a face. Plain ints, from the geometry alone, so nothing
    here depends on a tensor's value when it is traced.
    """
    boxes = []
    for first in range(0, extent, tile):
        start_first = window_start(first, extent, size)
        start_last = window_start(first + tile - 1, extent, size)
        assert isinstance(start_first, int) and isinstance(start_last, int)
        boxes.append((start_first // tile, (start_last + size - 1) // tile))
    return boxes


def _tile_box_mask_mod(
    grid: tuple[int, ...],
    window: tuple[int, ...],
    tile: tuple[int, ...],
    prefix: int,
    copies: int,
    group: int,
) -> Callable[..., torch.Tensor]:
    """FlexAttention's per-pair predicate: may this grid query see this key?

    Queries are the grid tokens in tile order (`partition_windows(x, grid, tile)`). Keys are
    `copies` blocks of the prefix -- its tokens, then padding -- followed by the grid tokens in tile
    order. A grid query sees the prefix tokens in its group's copy, never the padding, and every key
    in its tile's box. The fused kernel evaluates this only in the prefix blocks, since every grid
    block it is handed is wholly inside a box; on the unfused CPU path, which ignores block lists,
    it is the definition.
    """
    counts = [g // t for g, t in zip(grid, tile, strict=True)]
    volume = math.prod(tile)
    slots = copies * volume

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001, ANN202 - FlexAttention's signature
        query_tile, key_tile = q_idx // volume, (kv_idx - slots) // volume
        own_copy = (kv_idx // volume == query_tile // group) & (kv_idx % volume < prefix)
        inside = None
        for axis in reversed(range(len(grid))):  # tile indices are row-major: last axis fastest
            q, k = query_tile % counts[axis], key_tile % counts[axis]
            query_tile, key_tile = query_tile // counts[axis], key_tile // counts[axis]
            extent, size, t = grid[axis], window[axis], tile[axis]
            first = window_start(q * t, extent, size) // t
            last = (window_start(q * t + t - 1, extent, size) + size - 1) // t
            on_axis = (k >= first) & (k <= last)
            inside = on_axis if inside is None else inside & on_axis
        if slots:
            inside = torch.where(kv_idx < slots, own_copy, inside)
        return inside

    return mask_mod


def _packed(
    selected: torch.Tensor, index: torch.Tensor, blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row, the `index` entries where `selected`, packed to the left: (counts, indices).

    Padded with zeros to `blocks` columns, one per key block. That is BlockMask's layout, not a
    formality: `from_kv_blocks` transposes the lists for the backward pass through a dense
    (query blocks, columns) matrix indexed by key block, so a narrower list -- a row only as wide
    as its candidates -- indexes past its end. The padding is zero rather than whatever the
    candidate grid held there, so no entry past a row's count names a block that does not exist.
    """
    order = torch.argsort((~selected).to(torch.int8), dim=1, stable=True)
    counts = selected.sum(dim=1)
    packed = index.gather(1, order)
    used = torch.arange(packed.shape[1], device=packed.device) < counts[:, None]
    packed = F.pad(torch.where(used, packed, 0), (0, blocks - packed.shape[1]))
    return counts.to(torch.int32), packed.to(torch.int32)


def prefix_copies(grid: Sequence[int], prefix: int, group: int = PREFIX_GROUP) -> int:
    """How many copies of the prefix's key block the sliding kernel attends to (`PREFIX_GROUP`)."""
    if not prefix:
        return 0
    tiles = math.prod(g // t for g, t in zip(grid, sliding_tile(grid), strict=True))
    return -(-tiles // group)


@functools.lru_cache(maxsize=32)
def sliding_block_mask(
    grid: tuple[int, ...],
    window: tuple[int, ...],
    prefix: int,
    device: torch.device,
    group: int = PREFIX_GROUP,
) -> BlockMask:
    """The FlexAttention block mask of `window`s sliding tile by tile over `grid`.

    `prefix` tokens, when there are any, sit in front of the grid.

    Every key tile in a query tile's box is listed as *full*, so the kernel computes those blocks
    without evaluating any predicate, at the speed of dense attention. The prefix, when there is
    one, is `prefix_copies` *partial* blocks in front of the grid's, one per `group` query tiles,
    whose padding the predicate masks out.
    Built from the geometry rather than by evaluating the predicate on every (query, key) pair --
    `create_block_mask`'s way, which at a 2048-cube crop would be 4e12 evaluations -- and cached,
    because it depends only on its arguments, which a run repeats every layer and every step.
    """
    tile = sliding_tile(grid)
    volume = math.prod(tile)
    counts = [g // t for g, t in zip(grid, tile, strict=True)]
    copies = prefix_copies(grid, prefix, group)

    # Every (query tile, candidate key tile) pair as one dense product over the axes: axis a holds
    # the query tile's coordinate on dimension a and the candidate's offset on dimension rank + a.
    rank = len(grid)
    key_tile = torch.zeros((), dtype=torch.long, device=device)
    inside = torch.ones((), dtype=torch.bool, device=device)
    for axis, (g, w, t) in enumerate(zip(grid, window, tile, strict=True)):
        boxes = _axis_tile_boxes(g, w, t)
        width = max(last - first for first, last in boxes) + 1
        query_shape, offset_shape = [1] * (2 * rank), [1] * (2 * rank)
        query_shape[axis], offset_shape[rank + axis] = -1, -1
        first, last = (
            torch.tensor(column, device=device).view(query_shape)
            for column in zip(*boxes, strict=True)
        )
        candidate = first + torch.arange(width, device=device).view(offset_shape)
        inside = inside & (candidate <= last)
        key_tile = key_tile * counts[axis] + candidate
    rows = math.prod(counts)
    key_tile = key_tile.reshape(rows, -1) + copies
    # A query tile's box never holds more tiles than exist: on each axis they are consecutive.
    blocks = copies + rows
    full_counts, full_indices = _packed(inside.reshape(rows, -1), key_tile, blocks)
    # Each query tile's one partial block, when there is a prefix, is its group's copy of it.
    partial_counts = torch.full((rows,), 1 if prefix else 0, dtype=torch.int32, device=device)
    partial_indices = torch.zeros(rows, blocks, dtype=torch.int32, device=device)
    if prefix:
        partial_indices[:, 0] = torch.arange(rows, device=device) // group
    cells = math.prod(grid)
    return BlockMask.from_kv_blocks(
        partial_counts.view(1, 1, rows),
        partial_indices.view(1, 1, rows, blocks),
        full_counts.view(1, 1, rows),
        full_indices.view(1, 1, rows, blocks),
        BLOCK_SIZE=volume,
        mask_mod=_tile_box_mask_mod(tuple(grid), tuple(window), tile, prefix, copies, group),
        seq_lengths=(cells, copies * volume + cells),
    )


@functools.cache
def _compiled_flex_attention() -> Callable[..., torch.Tensor]:
    # Eager FlexAttention runs an unfused reference that materialises every score; the fused
    # kernel is what `torch.compile` generates. Inside an already-compiled region the plain
    # function is traced into the enclosing graph instead.
    return torch.compile(flex_attention, dynamic=False)


@torch.compiler.assume_constant_result
@functools.cache
def _flash_backend_usable() -> bool:
    """Whether FlexAttention's FLASH backend can run here: FlashAttention-4 installed and usable.

    Imported where it is used, not at module level, because `layers.common.attention` imports this
    module; `flash4_status` is the one place the repo decides whether FA4 can run. A constant to
    `torch.compile`, which would otherwise trace the import and the device query.
    """
    from layers.common.attention import flash4_status

    return flash4_status()[0]


#: What FlashAttention-4, and so FlexAttention's FLASH backend, takes. Training runs in bf16 under
#: autocast; float32 (the tests' reference precision) goes to Triton.
_FLASH_DTYPES = (torch.float16, torch.bfloat16)


def _kernel_options(query: torch.Tensor, volume: int) -> dict[str, str] | None:
    """FLASH where it can take this input and block size; otherwise Triton, the default."""
    usable = query.is_cuda and query.dtype in _FLASH_DTYPES and volume % 256 == 0
    return {"BACKEND": "FLASH"} if usable and _flash_backend_usable() else None


def sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    grid: Sequence[int],
    window: Sequence[int],
    attend: Attend,
    scale: float,
    prefix: int = 0,
) -> torch.Tensor:
    """Attention in which each grid token sees its tile's sliding window, plus the prefix.

    (B, prefix + prod(grid), heads, head_dim) query, key and value -> the same shape. A tile's
    window is the box of tiles the `window`-wide windows centred on its tokens cover
    (`_axis_tile_boxes`), so every token sees at least its own centred window. The prefix tokens
    see everything, through `attend`; the grid tokens run on FlexAttention, which is fused on
    CUDA and an unfused reference elsewhere (forward only: FlexAttention has no CPU backward).
    """
    window = tuple(min(w, g) for w, g in zip(window, grid, strict=True))
    if query.shape[1] != prefix + math.prod(grid):
        raise ValueError(
            f"{query.shape[1]} tokens cannot be {prefix} prefix tokens followed by the token grid "
            f"{tuple(grid)}: sliding-window attention needs the whole grid, in row-major order"
        )
    if window == tuple(grid):
        return attend(query, key, value)  # every window is the whole grid: global attention

    tile = sliding_tile(grid)
    volume = math.prod(tile)
    if prefix > volume:
        raise ValueError(
            f"{prefix} prefix tokens do not fit the {volume}-token key block reserved for them"
        )
    group = PREFIX_GROUP  # read per call, not bound as a default, so it stays one knob
    copies = prefix_copies(grid, prefix, group)
    batch, heads, head_dim = query.shape[0], query.shape[2], query.shape[3]

    def tiled(t: torch.Tensor) -> torch.Tensor:  # grid tokens -> (B, heads, cells, head_dim)
        return partition_windows(t, grid, tile).reshape(batch, -1, heads, head_dim).transpose(1, 2)

    grid_query, grid_key, grid_value = (tiled(t[:, prefix:]) for t in (query, key, value))
    if prefix:
        # The prefix keys and values fill a block of their own, padded to its size, repeated once
        # per `group` query tiles; the mask never lets a query see the padding, and autograd sums
        # the copies' gradients back onto the prefix.
        def with_prefix(full: torch.Tensor, grid_part: torch.Tensor) -> torch.Tensor:
            block = F.pad(full[:, :prefix].transpose(1, 2), (0, 0, 0, volume - prefix))
            return torch.cat((block.repeat(1, 1, copies, 1), grid_part), dim=2)

        grid_key, grid_value = with_prefix(key, grid_key), with_prefix(value, grid_value)

    compiling = torch.compiler.is_compiling()
    build = sliding_block_mask.__wrapped__ if compiling else sliding_block_mask
    block_mask = build(tuple(grid), window, prefix, query.device, group)
    kernel = _compiled_flex_attention() if query.is_cuda and not compiling else flex_attention
    options = _kernel_options(grid_query, volume)
    attended = kernel(
        grid_query, grid_key, grid_value, block_mask=block_mask, scale=scale, kernel_options=options
    )
    attended = attended.transpose(1, 2).reshape(-1, volume, heads, head_dim)
    attended = merge_windows(attended, grid, tile, batch)
    if not prefix:
        return attended
    return torch.cat((attend(query[:, :prefix], key, value), attended), dim=1)


# ------------------------------------------------------------------ pooled keys


def pool_grid(x: torch.Tensor, grid: Sequence[int], pool: int) -> torch.Tensor:
    """(B, prod(grid), *rest) -> (B, prod(grid) / pool^rank, *rest): means over pool-sided boxes.

    The pooled tokens are in row-major order over the pooled grid `[g // pool for g in grid]`.
    Works on anything laid out per token -- keys, values, coordinates.
    """
    if x.shape[1] != math.prod(grid):
        raise ValueError(
            f"{x.shape[1]} tokens are not the token grid {tuple(grid)}: pooling needs the whole "
            "grid, in row-major order"
        )
    check_pool(grid, pool)
    batch, rank, rest = x.shape[0], len(grid), x.shape[2:]
    x = x.reshape(batch, *(n for g in grid for n in (g // pool, pool)), *rest)
    return x.mean(dim=tuple(range(2, 2 * rank + 1, 2))).reshape(batch, -1, *rest)


def attention_pairs(
    grid: Sequence[int],
    prefix: int,
    window: Sequence[int] | None = None,
    kv_pool: int = 1,
    window_mode: str = "block",
) -> int:
    """(query, key) pairs one self-attention layer scores over `prefix` tokens plus `grid`.

    Global attention scores every pair, `(prefix + prod(grid))^2`. A layer's attention FLOPs are
    `4 * pairs * embed_dim` whatever its pattern -- q.k and the weighted sum of v, a multiply and an
    add per channel each -- so this is the one number a model's `flops()` needs from its layout.
    A sliding window attends to its tile's box (`_axis_tile_boxes`), ~2.8x a block window of the
    same size for 16^3 windows on 8x8x4 tiles, and needs no tiling of the grid by the window.
    """
    cells = math.prod(grid)
    tokens = prefix + cells
    if window is not None and window_mode == "sliding":
        window = tuple(min(w, g) for w, g in zip(window, grid, strict=True))
        if window == tuple(grid):
            return tokens * tokens
        tile = sliding_tile(grid)
        # A box is a product over the axes, so its size summed over the tiles factorises too.
        boxes = math.prod(tile) * math.prod(
            t * sum(last - first + 1 for first, last in _axis_tile_boxes(g, w, t))
            for g, w, t in zip(grid, window, tile, strict=True)
        )
        return boxes + cells * prefix + prefix * tokens  # every grid query sees the prefix once
    if window is not None:
        window = resolve_window(grid, window)
        if window == tuple(grid):
            return tokens * tokens
        return cells * (math.prod(window) + prefix) + prefix * tokens
    if kv_pool > 1:
        check_pool(grid, kv_pool)
        return tokens * (prefix + cells // kv_pool ** len(grid))
    return tokens * tokens
