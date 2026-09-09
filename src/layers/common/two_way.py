"""The two-way transformer a promptable segmenter decodes with.

An ordinary transformer decoder reads from an encoder and never writes back. This one runs both
directions in every layer: the prompt tokens attend to the image, and then the image attends to
the prompt tokens. That second direction is what makes the construction work at all -- the mask is
finally produced by dotting a token against the image features, so those features have to have
been told which object is being asked about. It is Segment Anything's design, restated with two
substitutions.

**Rotation instead of an additive position encoding.** The reference adds a dense positional
encoding to the image keys and re-adds the prompt tokens' own (positional) embedding to the
queries at every one of the four attention calls -- eight additions per layer whose only purpose is
to keep an additive encoding from being washed out by the residual stream. Here position is a
rotation of queries and keys, reapplied inside each attention call at no cost, so all eight
additions and the reference's `skip_first_layer_pe` flag disappear. The logits then depend on the
*displacement* between a prompt and a patch rather than on two absolute encodings the network has
to learn to compare.

**Pre-norm instead of post-norm,** matching every other block in this repo
(`layers.common.blocks.TransformerBlock`, `algorithms.muvit_mae.MuViTDecoderLayer`) rather than the
reference. At two layers the difference is not numerically important; consistency is, because a
reader who knows one block here knows them all.

One rotary module per layer, shared by both sequences. Two schedules would put the prompt
coordinates and the patch coordinates on different frequency scales, and the displacement between
a click and the patch under it would stop meaning anything -- the same reason `MuViTDecoderLayer`
shares one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossAttention, SelfAttention
from .rope import AxialRotaryEmbedding

#: Rotary wavelengths at initialisation, in the [-1, 1] coordinate frame prompts and patches share
#: (`layers.common.rope.voxel_coords`), so a period of 2.0 is exactly one turn across the crop.
#:
#: `0.02` is a little under three voxels of a 256-crop: prompts arrive at voxel resolution and the
#: decoder has to tell apart two clicks inside one patch, which is how "part versus whole" is asked
#: for at all, so the fastest channel must vary appreciably below the token spacing (`2 / grid`,
#: 0.125 at a 16-token grid). `4.0` is two turns' worth of crop, so the slowest channel is
#: monotone across the whole volume -- a usable absolute signal for the tokens that sit at the
#: origin and have no displacement of their own.
#:
#: Both are initialisations. `AxialRotaryEmbedding` learns its frequencies, so a layer that wants a
#: different range can move there.
ROPE_MIN_PERIOD = 0.02
ROPE_MAX_PERIOD = 4.0


class TwoWayAttentionBlock(nn.Module):
    """One layer: token self-attention, token->image, token MLP, image->token."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention_backend: str = "auto",
        spatial_rank: int = 3,
        rope_min_period: float = ROPE_MIN_PERIOD,
        rope_max_period: float = ROPE_MAX_PERIOD,
    ) -> None:
        super().__init__()
        self.rotary = AxialRotaryEmbedding(
            dim // num_heads,
            spatial_rank,
            base=None,
            min_period=rope_min_period,
            max_period=rope_max_period,
        )

        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, backend=attention_backend)

        self.norm_token_query = nn.LayerNorm(dim)
        self.norm_image_context = nn.LayerNorm(dim)
        self.token_to_image = CrossAttention(dim, num_heads, backend=attention_backend)

        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        self.norm_image_query = nn.LayerNorm(dim)
        self.norm_token_context = nn.LayerNorm(dim)
        self.image_to_token = CrossAttention(dim, num_heads, backend=attention_backend)

    def forward(
        self,
        tokens: torch.Tensor,
        image: torch.Tensor,
        token_coords: torch.Tensor,
        image_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(P, K, dim)` tokens and `(P, N, dim)` image tokens -> both, updated.

        `image_coords` may be `(1, N, rank)` rather than `(P, N, rank)`: every prompt in the batch
        reads the same crop, so its rotation tables broadcast over the prompt axis instead of being
        materialised `P` times.
        """
        token_rope = self.rotary(token_coords)
        image_rope = self.rotary(image_coords)

        tokens = tokens + self.self_attn(self.norm_self(tokens), rope=token_rope)
        tokens = tokens + self.token_to_image(
            self.norm_token_query(tokens),
            self.norm_image_context(image),
            rope=token_rope,
            context_rope=image_rope,
        )
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        image = image + self.image_to_token(
            self.norm_image_query(image),
            self.norm_token_context(tokens),
            rope=image_rope,
            context_rope=token_rope,
        )
        return tokens, image


class TwoWayTransformer(nn.Module):
    """`depth` two-way layers, then one last token->image attention.

    The trailing attention is the reference's, and it is there because the layer loop ends by
    updating the *image*: without it the tokens that go on to produce the mask would be one
    cross-attention out of date with the features they are about to be dotted against.
    """

    def __init__(
        self,
        depth: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention_backend: str = "auto",
        spatial_rank: int = 3,
        rope_min_period: float = ROPE_MIN_PERIOD,
        rope_max_period: float = ROPE_MAX_PERIOD,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(
                dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                attention_backend=attention_backend,
                spatial_rank=spatial_rank,
                rope_min_period=rope_min_period,
                rope_max_period=rope_max_period,
            )
            for _ in range(depth)
        )
        self.final_rotary = AxialRotaryEmbedding(
            dim // num_heads,
            spatial_rank,
            base=None,
            min_period=rope_min_period,
            max_period=rope_max_period,
        )
        self.norm_final_query = nn.LayerNorm(dim)
        self.norm_final_context = nn.LayerNorm(dim)
        self.final_token_to_image = CrossAttention(dim, num_heads, backend=attention_backend)
        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: torch.Tensor,
        image: torch.Tensor,
        token_coords: torch.Tensor,
        image_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            tokens, image = layer(tokens, image, token_coords, image_coords)

        token_rope = self.final_rotary(token_coords)
        image_rope = self.final_rotary(image_coords)
        tokens = tokens + self.final_token_to_image(
            self.norm_final_query(tokens),
            self.norm_final_context(image),
            rope=token_rope,
            context_rope=image_rope,
        )
        return self.norm_out(tokens), image
