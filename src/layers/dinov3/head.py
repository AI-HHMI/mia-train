"""The projection head DINOv3 puts between a backbone and its self-supervised objective.

Ported from the DINOv3 reference implementation. An MLP down to a narrow bottleneck, an L2
normalization, then one bias-free linear layer onto a large bank of "prototypes". The objective
never compares features directly -- it compares the distribution each view induces over those
prototypes -- which is what lets the loss be a plain cross-entropy over `out_dim` classes that
nobody ever labelled.

Note this is *not* DINOv2's head: there the prototype layer is weight-normalized with a frozen
gain, and here it is a plain `nn.Linear`. The `force_weight_norm` key still present in the
reference configs is dead, read by nothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
    ) -> None:
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def init_weights(self) -> None:
        """Fill every linear layer, including the prototypes.

        Upstream leaves this to the trainer and calls it explicitly. Here it is called from the
        algorithm that owns the head, for the same reason the ViT models call their own: nothing
        in this repo would otherwise, and an unfilled head trains without complaint.
        """
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        # fp16's smallest normal is around 6e-5, so an eps of 1e-12 would flush to zero and make
        # the normalization a divide-by-zero.
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _build_mlp(
    nlayers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: int | None = None,
    use_bn: bool = False,
    bias: bool = True,
) -> nn.Module:
    """in_dim -> hidden -> ... -> bottleneck_dim, with `nlayers` linear layers in total."""
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    if hidden_dim is None:
        raise ValueError("hidden_dim is required when nlayers > 1")

    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.GELU())
    for _ in range(nlayers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers)
