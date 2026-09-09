"""Normalization over the channel axis of a channels-first tensor.

`nn.LayerNorm` normalizes over the *last* dimensions, which is what a token sequence wants and
exactly what a `(B, C, *spatial)` feature map does not: applied there it would normalize across
space and leave the channels alone. Convolutional stems and necks need the other one, so it is
written out once here rather than as an inline `movedim` sandwich at each of the places that want
it -- the sandwich is easy to write and easy to get backwards, and getting it backwards is a
normalization that trains without complaint.

`nn.GroupNorm(1, C)` is *not* an alternative: it normalizes over channels and space together, so
its statistics move with the spatial extent of the crop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelLayerNorm(nn.LayerNorm):
    """LayerNorm over dim 1 of `(B, C, *spatial)`, for any spatial rank.

    A subclass rather than a wrapper so the parameters keep `nn.LayerNorm`'s names (`weight`,
    `bias`) and its initialisation, which is what makes a checkpoint written by one readable by
    the other.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.movedim(1, -1)).movedim(-1, 1)
