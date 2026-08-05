from __future__ import annotations

import math

import torch
import torch.nn as nn

from .attention import SelfAttention
from .base import BaseModel
from .registry import ModelRegistry


class TransformerBlock(nn.Module):
    """Pre-norm transformer block.

    Public because it is shared: MAE builds its decoder from the same block, so this is part of
    this module's interface rather than an internal detail.
    """

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float, attention_backend: str = "auto"
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, backend=attention_backend)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


@ModelRegistry.register("vit3d")
class ViT3D(BaseModel):
    """Plain 3D ViT over volumetric patches.

    `embed` and `encode` are deliberately separate: masked autoencoding embeds every patch,
    discards most of the tokens, then encodes only what remains, so the encoder must accept an
    arbitrary token count rather than a fixed grid.

    Positional embeddings are learned rather than fixed sin-cos. For a 3D grid the sin-cos
    construction buys little once the encoder also sees a variable token subset, and a learned
    table keeps the masking bookkeeping to one gather.
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
        self.attention_backend = attention_backend
        self.grid_size = tuple(s // p for s, p in zip(img_size, patch_size, strict=True))

        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, mlp_ratio, attention_backend)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def num_patches(self) -> int:
        return int(math.prod(self.grid_size))

    @property
    def patch_volume(self) -> int:
        """Number of values in one patch, i.e. the width of a per-patch reconstruction."""
        return int(math.prod(self.patch_size)) * self.in_channels

    def prepare_input(self, batch: torch.Tensor, axes: str) -> torch.Tensor:
        """(B, *axes) -> (B, C, D, H, W). Single-scale: exactly one level per sample.

        A plain ViT has one patch grid at one resolution, so it is single-scale by construction.
        miao's scale levels share a centre but cover different physical extents, which makes them
        neither pixel-aligned (so they cannot be channels) nor interchangeable with independent
        samples (so folding them into the batch would quietly redefine `batch_size`). Rather than
        pick one of those for you, this requires a single level and says how to configure it. A
        multi-scale encoder overrides this and consumes the level axis itself.
        """
        if "l" not in axes:
            raise ValueError(f"axis order must contain 'l' (scale level), got {axes!r}")

        # After the level axis is dropped, what is left has to be readable as (C, D, H, W) or
        # (D, H, W). A trailing channel such as "lzyxc" would otherwise put a spatial axis where
        # the channel belongs, and no downstream shape check would catch it.
        remainder = axes.replace("l", "", 1)
        if not (len(remainder) == 3 or (len(remainder) == 4 and remainder[0] == "c")):
            raise ValueError(
                f"axis order {axes!r} is not usable by a 3D encoder: after the level axis it "
                'must be three spatial axes, optionally preceded by \'c\' (e.g. "lzyx" or '
                f'"lcxyz"), got {remainder!r}'
            )

        expected_dims = len(axes) + 1
        if batch.dim() != expected_dims:
            raise ValueError(
                f"axis order {axes!r} implies a {expected_dims}-D batch (batch + {len(axes)} "
                f"axes), got {tuple(batch.shape)}; it must match the dataset's output_axes"
            )

        level_dim = axes.index("l") + 1
        levels = batch.shape[level_dim]
        if levels != 1:
            raise ValueError(
                f"{type(self).__name__} is single-scale, but this batch carries {levels} scale "
                f"levels on axis 'l' (shape {tuple(batch.shape)}). Configure the dataset for one "
                "level per sample: give `resolutions` a single entry, or use "
                "`resolution_sampling` with `n_scales = 1`, which still varies the resolution but "
                "draws it independently per sample. A multi-scale encoder should override "
                "prepare_input and consume the level axis itself."
            )

        volumes = batch.squeeze(level_dim)
        if volumes.dim() == 4:  # axis order declared no channel; add a singleton
            volumes = volumes.unsqueeze(1)
        return volumes

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) -> (B, num_patches, embed_dim), positional embedding included."""
        if x.shape[1:] != (self.in_channels, *self.img_size):
            raise ValueError(
                f"expected input (B, {self.in_channels}, {', '.join(map(str, self.img_size))}), "
                f"got {tuple(x.shape)}"
            )
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        return tokens + self.pos_embed

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the transformer over any number of tokens -> (B, N, embed_dim)."""
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pooled volume embedding, for downstream heads."""
        return self.encode(self.embed(x)).mean(dim=1)

    def extra_forward_methods(self) -> tuple[str, ...]:
        """`embed` and `encode` are called directly by masked autoencoding, not through forward."""
        return ("embed", "encode")

    def flops(self, input_shape: tuple[int, ...]) -> int:
        """Rough forward FLOPs for one sample: patch embedding, attention, and MLPs."""
        n = self.num_patches
        d = self.embed_dim
        depth = len(self.blocks)
        patch_proj = 2 * n * self.patch_volume * d
        per_block = 2 * (4 * n * d * d) + 2 * (2 * n * n * d) + 2 * (2 * n * d * int(d * 4.0))
        return int(patch_proj + depth * per_block)
