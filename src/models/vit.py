from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from layers.common.blocks import TransformerBlock
from layers.common.window_attention import (
    attention_pairs,
    block_attention_layout,
    check_pool,
    check_window_mode,
    resolve_window,
)

from .base import BaseModel, single_scale_volumes
from .registry import ModelRegistry

SPATIAL_RANK = 3


@ModelRegistry.register("vit3d")
class ViT3D(BaseModel):
    """Plain 3D ViT over volumetric patches.

    `embed` and `encode` are deliberately separate: masked autoencoding embeds every patch,
    discards most of the tokens, then encodes only what remains, so the encoder must accept an
    arbitrary token count rather than a fixed grid.

    Position is carried by axial rotary embeddings on patch coordinates, applied inside attention,
    rather than by a learned table added to the tokens. Two reasons this suits a masked encoder:
    rotary attention encodes the *displacement* between two patches instead of their absolute slots,
    which is the relationship a 3D grid actually has; and there is no per-position table to keep
    aligned when most of the tokens are thrown away, only coordinates that travel with the tokens
    that survive.

    Attention is global in every block by default. `attn_window` makes blocks attend within
    windows of the patch grid -- blocks of it, or windows sliding tile by tile when
    `attn_window_mode = "sliding"` -- except those listed in `attn_global_blocks`, and
    `attn_global_kv_pool` averages the global blocks' keys and values over boxes of the grid; see
    `layers.common.window_attention`. Both need the whole grid in row-major order, which the dense
    path (`patch_features`) has and masked encoding does not, so a model configured with either
    serves the dense path only.

    `attn_max_score` bounds every attention score in every block by capping the lengths of queries
    and keys (`layers.common.attention.cap_lengths`). Off by default; it adds no parameters, so
    every checkpoint loads either way, and it changes nothing while the scores stay below it.
    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (64, 64, 64),
        patch_size: tuple[int, int, int] = (8, 8, 8),
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        attention_backend: str = "auto",
        rotary_base: float = 10000.0,
        attn_window: int | Sequence[int] | None = None,
        attn_global_blocks: Sequence[int] = (),
        attn_global_kv_pool: int = 1,
        attn_window_mode: str = "block",
        attn_max_score: float | None = None,
    ) -> None:
        super().__init__()
        img_size = tuple(img_size)  # type: ignore[assignment]
        patch_size = tuple(patch_size)  # type: ignore[assignment]
        if len(img_size) != 3 or len(patch_size) != 3:
            raise ValueError(f"img_size and patch_size must be 3D, got {img_size} {patch_size}")
        for size, patch in zip(img_size, patch_size, strict=True):
            if size % patch != 0:
                raise ValueError(
                    f"img_size {img_size} must be divisible by patch_size {patch_size} on "
                    "every axis; a partial patch would silently crop the volume"
                )

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.attention_backend = attention_backend
        self.grid_size = tuple(s // p for s, p in zip(img_size, patch_size, strict=True))

        # (window, kv_pool) per block. Checked against the grid here rather than per forward pass:
        # `embed` admits exactly one volume, so this is the only grid the model will ever see.
        self.attention_layout = block_attention_layout(
            depth, attn_window, attn_global_blocks, attn_global_kv_pool, SPATIAL_RANK
        )
        check_window_mode(attn_window_mode)
        self.attn_window_mode = attn_window_mode
        for window, kv_pool in self.attention_layout:
            if window is not None and attn_window_mode == "block":
                resolve_window(self.grid_size, window)  # sliding windows need not tile the grid
            check_pool(self.grid_size, kv_pool)

        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, attention_backend,
                spatial_rank=SPATIAL_RANK, rotary_base=rotary_base,
                window=window, kv_pool=kv_pool, window_mode=attn_window_mode,
                max_score=attn_max_score,
            )
            for window, kv_pool in self.attention_layout
        )
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def num_patches(self) -> int:
        return int(math.prod(self.grid_size))

    @property
    def patch_volume(self) -> int:
        """Number of values in one patch, i.e. the width of a per-patch reconstruction."""
        return int(math.prod(self.patch_size)) * self.in_channels

    def checkpointable_modules(self) -> tuple[nn.Module, ...]:
        """The transformer blocks: repeated, sequence-length-sized, and cheap to rerun."""
        return tuple(self.blocks)

    def prepare_input(self, batch: torch.Tensor, axes: str) -> torch.Tensor:
        """(B, *axes) -> (B, C, D, H, W). Single-scale: exactly one level per sample; see
        `models.base.single_scale_volumes`, the contract every single-scale 3D encoder shares.
        """
        return single_scale_volumes(self, batch, axes)

    def patch_coords(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Coordinates of every patch on the grid -> (B, num_patches, 3).

        Plain patch indices, one unit per patch, in the same row-major order the patches come out of
        the convolution. No centring or rescaling: rotary attention sees only the difference between
        two coordinates, so the origin is arbitrary, and a grid of at most a few dozen per axis is
        nowhere near the precision limits that make `MuViT3D` recentre its physical coordinates.
        """
        axes = [
            torch.arange(count, dtype=torch.float32, device=device) for count in self.grid_size
        ]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return grid.reshape(1, -1, SPATIAL_RANK).expand(batch_size, -1, -1)

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, C, D, H, W) -> tokens (B, num_patches, embed_dim) and coordinates (B, N, 3).

        Coordinates come back with the tokens because position now lives in attention rather than in
        the token values, so anything that reorders or drops tokens has to carry them along in step.
        Returning them together makes that hard to forget, and matches `MuViT3D.embed`.
        """
        if x.shape[1:] != (self.in_channels, *self.img_size):
            raise ValueError(
                f"expected input (B, {self.in_channels}, {', '.join(map(str, self.img_size))}), "
                f"got {tuple(x.shape)}"
            )
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        return tokens, self.patch_coords(x.shape[0], x.device)

    def encode(
        self, tokens: torch.Tensor, coords: torch.Tensor, grid: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Run the transformer over any number of tokens -> (B, N, embed_dim).

        The token count is free -- masked autoencoding passes a visible subset -- but every token
        must bring its coordinate, so `coords` is required rather than optional.

        `grid` says the tokens are the whole patch grid in row-major order, as `embed` returns
        them. Windowed and pooled blocks need that and refuse to run without it; a model whose
        blocks are all global ignores it.
        """
        if coords.shape[:2] != tokens.shape[:2]:
            raise ValueError(
                f"every token needs a coordinate: got {tokens.shape[1]} tokens but "
                f"{coords.shape[1]} coordinates"
            )
        if grid is None and any(
            window is not None or kv_pool > 1 for window, kv_pool in self.attention_layout
        ):
            raise ValueError(
                "this ViT3D has windowed or pooled attention blocks, which need the whole patch "
                "grid in row-major order, and encode() was given no grid. Masked encoding passes "
                "a subset of the tokens, so it needs every block global: leave attn_window unset "
                "and attn_global_kv_pool at 1."
            )
        for block in self.blocks:
            tokens = block(tokens, coords, grid)
        return self.norm(tokens)

    def patch_features(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """(B, C, D, H, W) -> every patch's encoded feature (B, num_patches, embed_dim).

        The whole grid, unlike the masked path `embed`/`encode` serve: a dense head needs a
        feature for every patch, so nothing is dropped here.
        """
        tokens, coords = self.embed(x)
        return self.encode(tokens, coords, self.grid_size), self.grid_size

    def patchify(self, volumes: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) -> (B, num_patches, patch_volume), matching the encoder's grid.

        Lives on the model rather than in the algorithm because it has to agree with `embed`'s patch
        order, which is the model's own layout. `MuViT3D.patchify` has the same signature over its
        extra level axis, so an algorithm can call it without knowing which encoder it holds.
        """
        pd, ph, pw = self.patch_size
        gd, gh, gw = self.grid_size
        batch, channels = volumes.shape[0], volumes.shape[1]
        x = volumes.reshape(batch, channels, gd, pd, gh, ph, gw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(batch, gd * gh * gw, channels * pd * ph * pw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pooled volume embedding, for downstream heads."""
        tokens, coords = self.embed(x)
        return self.encode(tokens, coords, self.grid_size).mean(dim=1)

    def extra_forward_methods(self) -> tuple[str, ...]:
        """`embed` and `encode` are called directly by masked autoencoding, not through forward."""
        return ("embed", "encode")

    def flops(self, input_shape: tuple[int, ...]) -> int:
        """Rough forward FLOPs for one sample: patch embedding, attention, and MLPs.

        `input_shape` is validated against the configured geometry rather than used to re-derive a
        grid. The token count is a function of the input volume in general, but `embed` admits
        exactly one volume -- `img_size` -- so the only shape this model can be asked about is the
        one whose grid is already `self.grid_size`, and deriving it again would just be the same
        arithmetic on a value that has to be equal anyway. Disagreement is a caller error, and it
        is checked rather than assumed away because the failure is otherwise silent and this is an
        MFU denominator: quietly answering for `img_size` when the caller asked about a different
        volume reports a plausible utilisation for a forward pass that would have raised.

        The FFN term uses `mlp_ratio`, which is what `TransformerBlock` sizes its hidden layer
        with; a fixed 4x here would have been right only for the default and off by tens of
        percent for every other setting, in the direction that flatters a narrow model.
        """
        if input_shape[-SPATIAL_RANK:] != self.img_size:
            raise ValueError(
                f"input_shape {tuple(input_shape)} does not describe an input this model can run: "
                f"its last {SPATIAL_RANK} axes must be the configured img_size {self.img_size}"
            )
        # The channel axis is optional -- a caller naming only the volume has not asserted anything
        # about channels, and there is nothing to check -- but when it is there it belongs to the
        # same contract as `img_size` and is just as load-bearing: `patch_volume` is linear in
        # `in_channels`, so costing a 3-channel shape on a 1-channel model would answer with a
        # patch-projection term three times too small, for a tensor `embed` rejects outright. That
        # is precisely the silent wrong integer the img_size check above exists to prevent, so
        # leaving the channel unchecked contradicted its own rationale.
        if len(input_shape) > SPATIAL_RANK and input_shape[-SPATIAL_RANK - 1] != self.in_channels:
            raise ValueError(
                f"input_shape {tuple(input_shape)} does not describe an input this model can run: "
                f"the axis before its last {SPATIAL_RANK} is the channel axis and must be the "
                f"configured in_channels {self.in_channels}"
            )
        n = self.num_patches
        d = self.embed_dim
        depth = len(self.blocks)
        hidden = int(d * self.mlp_ratio)
        patch_proj = 2 * n * self.patch_volume * d
        per_block = 2 * (4 * n * d * d) + 2 * (2 * n * d * hidden)
        # Attention is costed per block from the pairs it actually scores, which is `n^2` for a
        # global block and far fewer for a windowed or pooled one.
        pairs = sum(
            attention_pairs(self.grid_size, 0, window, kv_pool, self.attn_window_mode)
            for window, kv_pool in self.attention_layout
        )
        return int(patch_proj + depth * per_block + 4 * pairs * d)
