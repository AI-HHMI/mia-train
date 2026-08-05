"""The transformer block shared by every architecture here.

Both `ViT3D` and `MuViT3D` position their tokens with axial rotary embeddings on per-token
coordinates, which makes their blocks the same object: a pre-norm block whose attention is rotated
by whatever coordinates the caller supplies. What differs between the two models is only what the
coordinates *mean* -- patch indices on a single grid, or physical positions in a world frame shared
across resolution levels -- and that is the model's business, not the block's.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import SelfAttention
from .rope import AxialRotaryEmbedding


class TransformerBlock(nn.Module):
    """Pre-norm transformer block, positioned by rotary embeddings on token coordinates.

    Each block owns its own rotary frequencies rather than sharing one schedule across the stack,
    so a layer can specialise in short- or long-range structure. That is MuViT's design, and it
    costs a handful of parameters per layer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention_backend: str = "auto",
        spatial_rank: int = 3,
        rotary_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, backend=attention_backend)
        self.rotary = AxialRotaryEmbedding(dim // num_heads, spatial_rank, base=rotary_base)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """(B, N, dim) tokens at (B, N, spatial_rank) coordinates -> (B, N, dim)."""
        x = x + self.attn(self.norm1(x), rope=self.rotary(coords))
        return x + self.mlp(self.norm2(x))
