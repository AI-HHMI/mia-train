"""Turning a segmentation prompt into tokens a mask decoder can attend over.

A prompt says *what to segment* without saying how. This module maps the four kinds this repo
supports -- foreground/background points, a box, a previous mask, and a semantic class -- onto one
uniform representation: a short sequence of `(token, coordinate)` pairs plus one dense feature map
added to the image embedding.

**Position is carried by the coordinate, not by the token.** This is the one substantial departure
from the Segment Anything reference, where a point's embedding *is* its position -- a fixed random
Fourier projection of its normalised location, added to a learned type embedding, and re-added to
the queries at every attention layer. Here the token carries only the prompt's *type* and the
coordinate travels beside it, to be applied as a rotation inside attention (`layers.common.rope`).
Three things follow:

  * A point prompt and the patch token containing it are positioned by the same function
    (`rope.voxel_coords` and `rope.patch_grid_coords` agree by construction), so the decoder can
    express "this patch is where the click was" as a zero displacement rather than having to learn
    that two unrelated encodings mean the same place.
  * Attention logits depend on the displacement between a prompt and a patch rather than on both
    absolute positions, so a prompt means the same thing at any crop size.
  * The reference's `get_dense_pe()` and its per-layer re-addition of the prompt embedding both
    disappear. They exist to keep an *additive* encoding alive through a residual stream; a
    rotation is reapplied at every layer for free.

**Tokens without a position sit at coordinate 0.** Padding and class tokens have no location. The
identity rotation is exactly coordinate 0, so nothing special has to happen for them -- and under
this normalisation the origin is the centre of the crop, which is a sensible anchor for a token
whose job is global. This is a choice, not an accident: such a token's attention to a patch then
depends on that patch's offset from the crop centre.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .norms import ChannelLayerNorm

#: `point_labels` vocabulary. A box is two points, which is how the reference represents it
#: internally too: it needs no separate path, it batches with the points, and it keeps the decoder
#: from having to know which of its tokens came from where.
PAD = -1
BACKGROUND = 0
FOREGROUND = 1
BOX_NEAR = 2
BOX_FAR = 3
NUM_POINT_TYPES = 4


class PromptEncoder3D(nn.Module):
    """Prompts -> `(sparse tokens, their coordinates, dense embedding)`.

    `mask_downscale` is how much coarser the token grid is than a mask prompt, and must be a power
    of two. It is `patch_size / mask_stride` for whatever stride the decoder emits masks at, so
    feeding a round's own output back as the next round's prompt needs no resampling. The
    reference's value is 4.

    `num_classes` opts into semantic prompting. The table holds `num_classes + 1` rows; the last is
    "no class named", so a run that mixes class-prompted and purely geometric prompts stays
    rectangular. Left at 0, no class token is produced at all and the parameter does not exist --
    the corpus this was built for has a closed ~35-class organelle ontology and no free text, so
    this is deliberately an embedding table rather than a text encoder. A text encoder would slot
    in here, producing tokens in the same space.
    """

    def __init__(
        self,
        embed_dim: int,
        mask_downscale: int = 4,
        mask_hidden: int | None = None,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        if mask_downscale < 1 or mask_downscale & (mask_downscale - 1):
            raise ValueError(
                f"mask_downscale must be a power of two, got {mask_downscale}: the stem reaches "
                "the token grid by stride-2 convolutions, and a non-power-of-two factor has no "
                "exact decomposition into them"
            )
        if num_classes < 0:
            raise ValueError(f"num_classes must not be negative, got {num_classes}")

        self.embed_dim = embed_dim
        self.mask_downscale = mask_downscale
        self.num_classes = num_classes

        self.point_embed = nn.Embedding(NUM_POINT_TYPES, embed_dim)
        self.not_a_point = nn.Embedding(1, embed_dim)
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim) if num_classes else None

        # Broadcast over every grid position when no mask prompt is given, so "no mask" is a
        # learned state rather than a zero the decoder has to distinguish from a genuinely empty
        # one.
        self.no_mask = nn.Embedding(1, embed_dim)

        hidden = mask_hidden if mask_hidden is not None else max(embed_dim // 4, 1)
        steps = int(math.log2(mask_downscale))
        stem: list[nn.Module] = []
        channels = 1
        for _ in range(steps):
            stem += [
                nn.Conv3d(channels, hidden, kernel_size=2, stride=2),
                ChannelLayerNorm(hidden),
                nn.GELU(),
            ]
            channels = hidden
        stem.append(nn.Conv3d(channels, embed_dim, kernel_size=1))
        self.mask_stem = nn.Sequential(*stem)

    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        grid: tuple[int, ...],
        classes: torch.Tensor | None = None,
        mask_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(P, K, 3)` coordinates and `(P, K)` labels -> tokens, coordinates, dense embedding.

        `P` is the prompt batch: samples times masks-per-sample, already flattened, because every
        prompt is decoded independently against the same image embedding.

        Coordinates are in [-1, 1] and must have been produced by `rope.voxel_coords` against the
        crop the encoder saw -- this module cannot check that and would be positioning prompts in a
        different frame from the image tokens if it were not so.
        """
        if point_coords.shape[:-1] != point_labels.shape:
            raise ValueError(
                f"point_coords {tuple(point_coords.shape)} and point_labels "
                f"{tuple(point_labels.shape)} disagree about the prompt batch or token count"
            )

        prompts = point_labels.shape[0]
        padded = (point_labels == PAD).unsqueeze(-1)
        # `clamp_min` keeps the padding rows in range for the lookup; `where` then discards them.
        tokens = torch.where(
            padded, self.not_a_point.weight.to(point_coords.dtype),
            self.point_embed(point_labels.clamp_min(0)),
        )
        coords = torch.where(padded, torch.zeros_like(point_coords), point_coords)

        if self.class_embed is not None:
            named = (
                classes
                if classes is not None
                else point_labels.new_full((prompts,), self.num_classes)
            )
            tokens = torch.cat([tokens, self.class_embed(named).unsqueeze(1)], dim=1)
            coords = torch.cat([coords, torch.zeros_like(coords[:, :1])], dim=1)
        elif classes is not None:
            raise ValueError(
                "classes were supplied but this prompt encoder was built with num_classes=0, so "
                "there is no embedding table to look them up in"
            )

        dense = self._dense(mask_input, prompts, grid, tokens.dtype, tokens.device)
        return tokens, coords, dense

    def _dense(
        self,
        mask_input: torch.Tensor | None,
        prompts: int,
        grid: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """`(P, embed_dim, *grid)`, from a mask prompt or from the learned "no mask" state."""
        if mask_input is None:
            return self.no_mask.weight.to(dtype=dtype, device=device).reshape(
                1, self.embed_dim, *(1,) * len(grid)
            ).expand(prompts, self.embed_dim, *grid)

        expected = tuple(extent * self.mask_downscale for extent in grid)
        if tuple(mask_input.shape[2:]) != expected:
            raise ValueError(
                f"mask prompt is {tuple(mask_input.shape[2:])} but a token grid of {grid} at "
                f"mask_downscale={self.mask_downscale} expects {expected}. A mask prompt is fed "
                "back at the stride the decoder emits, so these agree by construction unless the "
                "stride changed between rounds."
            )
        return self.mask_stem(mask_input)
