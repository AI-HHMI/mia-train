"""Self-attention for DINOv3, with rotary position embeddings applied inside.

Ported from the DINOv3 reference implementation. Distinct from `layers.common.attention` -- which
the from-scratch ViT3D/MuViT3D use and which takes a rotation callable -- in three ways that the
DINOv3 architecture depends on:

  - RoPE arrives as an explicit (sin, cos) pair and is applied only to the *suffix* of the token
    sequence, so the CLS and storage tokens that sit in front of the patch tokens stay unrotated.
  - `mask_k_bias` swaps in a Linear whose key-half bias is forced to zero, which DINOv3 needs to
    load weights from models trained with an untied key bias.
  - `forward_list` runs the shared projections once across crops of different sizes.

The FlashAttention-4 path reuses `layers.common.attention.flash4_status` rather than re-probing
the import, so both attention implementations agree on when the kernel is usable and give the same
diagnostics when it is not.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.common.attention import flash4_status
from layers.common.batched_tokens import cat_keep_shapes, uncat_with_shapes


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    """A fused QKV Linear whose key-half bias is masked out.

    A bias on the keys cancels in the softmax, so it is redundant; DINOv3 keeps the parameter for
    checkpoint compatibility but multiplies it by a fixed 0/1 mask. The mask is a buffer set by
    `init_weights_vit`, not a constant, so it travels with the state dict.
    """

    bias_mask: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        masked_bias = (
            self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        )
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        use_fa4: bool = False,
        device=None,
    ) -> None:
        super().__init__()

        if use_fa4:
            usable, reason = flash4_status()
            if not usable:
                raise ValueError(
                    f"use_fa4=True was requested but FlashAttention-4 is unusable: {reason}. "
                    "Set use_fa4=False to use torch's scaled_dot_product_attention instead."
                )
        self.use_fa4 = use_fa4

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # All operations use the dtype of rope; the output is cast back to the dtype of q and k.
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        # The rope tables cover only the patch tokens, so the leading CLS and storage tokens are
        # left unrotated -- they have no position on the grid.
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: torch.Tensor, attn_bias=None, rope=None) -> torch.Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> list[torch.Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list, strict=True)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: torch.Tensor, attn_bias=None, rope=None) -> torch.Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)

        q, k = (t.transpose(1, 2) for t in [q, k])
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        if self.use_fa4:
            from flash_attn.cute import flash_attn_func

            # FA4 reads (batch, seq, heads, head_dim); v is already in that layout. The scale is
            # passed explicitly even though the default matches, so the two paths are comparable
            # by construction rather than by coincidence.
            q, k = (t.transpose(1, 2) for t in [q, k])
            x, _ = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=False)
        else:
            # fall back on F.sdpa()
            v = v.transpose(1, 2)
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            x = x.transpose(1, 2)

        return x.reshape([B, N, C])
