"""Prompt tokens plus an image embedding -> several candidate masks and a score for each.

Segment Anything's mask decoder, restated for volumes. The shape of the thing is unchanged: a
two-way transformer mixes a handful of tokens with the image, one token per candidate mask is
turned into a vector by a small hypernetwork, and the mask is the dot product of that vector with
an upscaled feature volume. Two substitutions matter in 3D.

**The upscaling is a sub-pixel expansion, not a stack of transposed convolutions.** Both give each
token a learned readout of its own block; the difference is where the arithmetic happens. Measured
in this repo at a 256-cube with patch 16, cuDNN has no tuned kernel for a 3D transposed convolution
at `kernel == stride` and falls back a hardware generation, costing 152 ms of a 344 ms step for
0.42 ms of arithmetic -- so `layers.common.dense_heads.SubPixelHead` evaluates the same function as
a matmul instead. Reusing it also means the mask head and the affinity head share one construction.

**The upscaled volume is produced once per prompt and dotted `K` times, not upscaled `K` times.**
That is what keeps a large `K` cheap: the wide arithmetic stays on the patch grid and only the
final dot product runs at mask resolution.

Masks come out at `patch_size / upscale` of the input, which at the reference's 4 is the same ratio
it uses (256-pixel masks for a 1024-pixel image). Nothing here upsamples them further: in 2D that
last step is nearly free and in 3D it is a factor of `stride^3` on the largest tensor in the step,
so it belongs to whoever actually needs voxel resolution rather than to every training step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.common.dense_heads import SubPixelHead
from layers.common.two_way import ROPE_MAX_PERIOD, ROPE_MIN_PERIOD, TwoWayTransformer

SPATIAL_RANK = 3


class MLP(nn.Module):
    """`depth` linear layers with ReLU between, as the reference's hypernetworks and IoU head."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        widths = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:], strict=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = nn.functional.relu(x)
        return x


class MaskDecoder3D(nn.Module):
    """`(P, N, dim)` image tokens and `(P, K, dim)` prompt tokens -> masks and predicted IoUs.

    `num_multimask_outputs` candidate masks answer an ambiguous prompt, plus one more token used
    when the prompt is unambiguous. That extra token is the reference's and it is not a spare: with
    several prompts the three candidates collapse onto each other, and scoring three near-identical
    masks with a min-reduction gives the winner a third of the gradient it should have. Three is
    the reference's count, on the grounds that nesting in natural images is at most three deep
    (whole, part, subpart); the same holds here (cell, organelle, sub-compartment).

    `upscale` is how much finer than the patch grid the masks are emitted, so the output stride is
    `patch_size / upscale`. `mask_feature_dim` is the width the dot product runs at, and it is the
    number to watch for cost: every tensor after the expansion carries it over the whole upscaled
    volume.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        num_multimask_outputs: int = 3,
        upscale: int = 4,
        mask_feature_dim: int = 32,
        upscale_hidden: int | None = None,
        refine_depth: int = 2,
        iou_hidden: int = 256,
        iou_depth: int = 3,
        attention_backend: str = "auto",
        rope_min_period: float = ROPE_MIN_PERIOD,
        rope_max_period: float = ROPE_MAX_PERIOD,
    ) -> None:
        super().__init__()
        if num_multimask_outputs < 1:
            raise ValueError(
                f"num_multimask_outputs must be at least 1, got {num_multimask_outputs}"
            )
        if upscale < 1:
            raise ValueError(f"upscale must be at least 1, got {upscale}")

        self.dim = dim
        self.upscale = upscale
        self.num_mask_tokens = num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, dim)

        self.transformer = TwoWayTransformer(
            depth=depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_backend=attention_backend,
            spatial_rank=SPATIAL_RANK,
            rope_min_period=rope_min_period,
            rope_max_period=rope_max_period,
        )

        # `zero_init_output=False`: a zeroed final projection would make the mask features
        # identically zero, and the hypernetwork that dots against them would receive no gradient
        # at all -- the head could never open. That is the opposite of the situation the flag is
        # for, where a zeroed head protects an encoder that already solves the task.
        self.upscaler = SubPixelHead(
            dim,
            (upscale,) * SPATIAL_RANK,
            mask_feature_dim,
            hidden=upscale_hidden if upscale_hidden is not None else dim,
            readout=mask_feature_dim,
            refine_depth=refine_depth,
            zero_init_output=False,
        )
        self.hypernetworks = nn.ModuleList(
            MLP(dim, dim, mask_feature_dim, depth=3) for _ in range(self.num_mask_tokens)
        )
        self.iou_head = MLP(dim, iou_hidden, self.num_mask_tokens, depth=iou_depth)

    def forward(
        self,
        image: torch.Tensor,
        image_coords: torch.Tensor,
        grid: tuple[int, ...],
        sparse_tokens: torch.Tensor,
        sparse_coords: torch.Tensor,
        dense: torch.Tensor,
        multimask: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> mask logits `(P, K, *grid * upscale)` and predicted IoUs `(P, K)`.

        `K` is `num_multimask_outputs` when `multimask`, else 1. `image` is `(P, N, dim)`: the same
        crop repeated across the prompt batch, which is what lets one encoder pass serve every
        prompt. `image_coords` may be `(1, N, rank)`, since every prompt reads the same crop.
        """
        prompts = sparse_tokens.shape[0]
        if image.shape[0] != prompts:
            raise ValueError(
                f"{image.shape[0]} image embeddings against {prompts} prompts; the embedding is "
                "expanded to the prompt batch so that one encoder pass serves them all"
            )

        # The dense prompt (a previous mask, or the learned "no mask" state) is added to the image
        # rather than concatenated: it is registered with the grid, so it is a property of each
        # position rather than another thing to attend to.
        image = image + dense.flatten(2).transpose(1, 2)

        output = torch.cat([self.iou_token.weight, self.mask_tokens.weight]).to(image.dtype)
        tokens = torch.cat([output.unsqueeze(0).expand(prompts, -1, -1), sparse_tokens], dim=1)
        # Output tokens have no location, and the identity rotation is exactly coordinate 0 -- see
        # `layers.common.prompt` for why the origin is the right place to put them.
        token_coords = torch.cat(
            [sparse_coords.new_zeros(prompts, output.shape[0], sparse_coords.shape[-1]),
             sparse_coords],
            dim=1,
        )

        tokens, image = self.transformer(tokens, image, token_coords, image_coords)
        iou_token = tokens[:, 0]
        mask_tokens = tokens[:, 1 : 1 + self.num_mask_tokens]

        size = tuple(extent * self.upscale for extent in grid)
        features = self.upscaler(image.transpose(1, 2).reshape(prompts, self.dim, *grid), size)

        vectors = torch.stack(
            [network(mask_tokens[:, i]) for i, network in enumerate(self.hypernetworks)], dim=1
        )
        masks = torch.einsum("pkc,pcv->pkv", vectors, features.flatten(2)).reshape(
            prompts, self.num_mask_tokens, *size
        )
        scores = self.iou_head(iou_token)

        # Slot 0 is the unambiguous answer and 1.. are the ambiguous ones, so an unambiguous prompt
        # is not scored against candidates that were trained to disagree with each other.
        selection = slice(1, None) if multimask else slice(0, 1)
        return masks[:, selection], scores[:, selection]
