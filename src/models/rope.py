"""Axial rotary position embedding over continuous coordinates.

Rotary embeddings (RoPE) make an attention logit depend on the *difference* between two tokens'
positions rather than their absolute values. Written out here rather than taken from a library
because the interesting property for multi-resolution models is that positions need not be integer
sequence indices: feed physical coordinates and two patches describing the same place get the same
rotation, whatever resolution they came from.

"Axial" means each spatial axis owns a slice of the head dimension and rotates it by its own
coordinate, so the axes stay separable. The alternative -- concatenating every axis's angles and
rotating the head dimension as one block -- mixes axes at the halfway split and is not what
axis-wise RoPE means.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def split_rope_dims(head_dim: int, spatial_rank: int) -> tuple[int, ...]:
    """Divide a head's channels among spatial axes, each an even count.

    Even because rotation acts on pairs of channels. The remainder goes to the first axis instead
    of being dropped, so the whole head dimension is used when it divides unevenly.
    """
    if head_dim < 2 * spatial_rank:
        raise ValueError(
            f"head_dim {head_dim} cannot cover {spatial_rank} spatial axes: each axis needs at "
            f"least one channel pair, so head_dim must be at least {2 * spatial_rank}. Raise "
            "embed_dim or lower num_heads."
        )
    per_axis = 2 * (head_dim // spatial_rank // 2)
    dims = [per_axis] * spatial_rank
    dims[0] += 2 * ((head_dim - sum(dims)) // 2)
    return tuple(dims)


@dataclass(frozen=True)
class RotaryTables:
    """Precomputed cos/sin per axis, ready to rotate a (B, N, heads, head_dim) tensor.

    A callable rather than a method on the embedding module so attention can apply rotation
    without knowing anything about how the angles were produced -- the attention layer stays a
    generic kernel with a swappable position encoding.
    """

    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    axis_dims: tuple[int, ...]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate (B, N, heads, head_dim) in place of its position-free self."""
        rotated = []
        start = 0
        for (cos, sin), width in zip(self.tables, self.axis_dims, strict=True):
            chunk = x[..., start : start + width]
            first, second = chunk.chunk(2, dim=-1)
            # The standard 2D rotation, applied to (first, second) as one complex pair per
            # frequency. cos/sin arrive as (B, N, 1, width/2) and broadcast over heads.
            rotated.append(
                torch.cat(
                    (first * cos - second * sin, second * cos + first * sin), dim=-1
                )
            )
            start += width

        if start < x.shape[-1]:
            # Channels past the axis allocations carry no position, which is the usual RoPE
            # arrangement when head_dim does not divide evenly.
            rotated.append(x[..., start:])
        # Angles are computed in fp32 for precision; casting back keeps a bf16 forward in bf16,
        # which also matters because a silently promoted dtype would disable the FA4 kernel.
        return torch.cat(rotated, dim=-1).to(x.dtype)


class AxialRotaryEmbedding(nn.Module):
    """Turns per-token coordinates into rotation tables.

    Frequencies are learnable and initialised to the usual geometric progression, matching MuViT:
    every layer owns its own copy, so a layer can widen or narrow the range of distances its
    attention is sensitive to instead of inheriting one fixed schedule.
    """

    def __init__(self, head_dim: int, spatial_rank: int, base: float = 10000.0) -> None:
        super().__init__()
        if base <= 1.0:
            raise ValueError(f"rotary base must be greater than 1, got {base}")
        self.axis_dims = split_rope_dims(head_dim, spatial_rank)
        self.spatial_rank = spatial_rank
        # theta_k = coordinate / base^(2k/d), stored as the reciprocal so the forward pass is a
        # multiply. One Parameter per axis, since axes may have different widths.
        self.inv_freqs = nn.ParameterList(
            nn.Parameter(1.0 / (base ** (torch.arange(0, width, 2).float() / width)))
            for width in self.axis_dims
        )

    @property
    def rotary_dim(self) -> int:
        """How many of a head's channels carry position."""
        return sum(self.axis_dims)

    def forward(self, coords: torch.Tensor) -> RotaryTables:
        """(B, N, spatial_rank) coordinates -> tables for rotating queries and keys."""
        if coords.shape[-1] != self.spatial_rank:
            raise ValueError(
                f"expected coordinates with {self.spatial_rank} components on the last axis, got "
                f"{tuple(coords.shape)}"
            )

        tables = []
        for axis, inv_freq in enumerate(self.inv_freqs):
            # fp32 regardless of autocast: the angle is a product of a possibly large coordinate
            # and a small frequency, and half precision loses the distinctions that make nearby
            # positions different.
            angles = coords[..., axis].float().unsqueeze(-1) * inv_freq.float()
            angles = angles.unsqueeze(-2)  # (B, N, 1, width/2), broadcasting over heads
            tables.append((angles.cos(), angles.sin()))
        return RotaryTables(tuple(tables), self.axis_dims)
