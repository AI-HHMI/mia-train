"""Flatten a list of token tensors into one batch and put it back afterwards.

DINOv3-style training feeds several crops of different sizes through the same encoder at once --
global crops and local crops have different token counts, so they cannot be stacked into a single
tensor. Rather than looping the elementwise ops (norms, projections, MLPs) once per crop, the
layers concatenate every crop's tokens along the token axis, apply the op once, and split the
result back. The heavy per-crop work that genuinely cannot be shared -- attention, whose cost is
quadratic within a crop -- still runs per crop.

Kept in `common/` rather than inside `dinov3/` because nothing about it is DINOv3-specific: it is
a plain shape-preserving concat/split over a list of tensors that agree on their last dimension.
"""

from __future__ import annotations

import torch


def cat_keep_shapes(
    x_list: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Size], list[int]]:
    """Concatenate tensors that share a last dimension -> (flat, shapes, token counts).

    Each input is flattened to (tokens, features) before concatenation, so inputs may differ in
    both rank and leading extents. `shapes` and `num_tokens` are what `uncat_with_shapes` needs to
    invert this exactly.
    """
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(
    flattened: torch.Tensor, shapes: list[torch.Size], num_tokens: list[int]
) -> list[torch.Tensor]:
    """Inverse of `cat_keep_shapes`, allowing the feature dimension to have changed.

    The last dimension is taken from `flattened` rather than from `shapes`, so an op that changes
    width -- a projection from `dim` to `3 * dim`, say -- still splits back correctly.
    """
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    return [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted, strict=True)]
