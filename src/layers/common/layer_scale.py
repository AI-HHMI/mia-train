"""Per-channel learnable rescaling of a residual branch, as used by DINOv3.

Ported from the DINOv3 reference implementation. Applied to an attention or FFN branch's output
before it is added to the residual stream, so each channel can be damped independently -- which is
what lets very deep ViT stacks train stably without a warmup schedule tuned per depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self) -> None:
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
