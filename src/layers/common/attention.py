"""Multi-head self-attention with a swappable kernel.

Written out rather than delegating to `nn.MultiheadAttention` so the attention call itself is one
named line that can be pointed at a different kernel. That is what makes FlashAttention-4 usable
where the hardware supports it, and what leaves room to try other attention implementations
without touching the surrounding transformer.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKENDS = ("auto", "flash4", "sdpa")

# FlashAttention-4 ships CuTeDSL kernels for Hopper and Blackwell only, and they take half
# precision or narrower — fp32 activations have to go through SDPA instead.
_FLASH4_MIN_CAPABILITY = (9, 0)
_FLASH4_DTYPES = (torch.float16, torch.bfloat16)


def flash4_status() -> tuple[bool, str]:
    """Whether FlashAttention-4 can run on this process's device, and why not when it cannot.

    The reason is returned rather than logged so an explicit request for the flash4 backend can
    fail with something actionable instead of silently degrading to SDPA.
    """
    try:
        from flash_attn.cute import flash_attn_func  # noqa: F401
    except ImportError:
        # An optional dependency: absent is a normal state, not an error. Anything other than
        # ImportError means a broken install and is left to propagate.
        return False, "flash-attn-4 is not installed"

    if not torch.cuda.is_available():
        return False, "no CUDA device available"

    capability = torch.cuda.get_device_capability()
    if capability < _FLASH4_MIN_CAPABILITY:
        needed = ".".join(str(part) for part in _FLASH4_MIN_CAPABILITY)
        have = ".".join(str(part) for part in capability)
        return False, f"compute capability {have} is below {needed} (Hopper)"

    return True, ""


@contextlib.contextmanager
def counting_kernels(module: nn.Module) -> Iterator[None]:
    """Route every attention layer under `module` through SDPA for the duration.

    For counting FLOPs, not for training. The arithmetic of attention does not depend on which
    kernel performs it, but `torch.utils.flop_counter` can only see arithmetic that reaches the
    dispatcher, and FlashAttention-4 does not: `flash_attn.cute.flash_attn_func` is a plain Python
    function wrapping a CuTeDSL kernel, not a registered torch op, so nothing reaches
    `__torch_dispatch__`. The counter does not warn about this — it silently attributes zero FLOPs
    to attention. Measured on an H100 at B=2, N=1024, d=768: SDPA counts 16.106 GF, matching the
    closed form exactly, while FA4 counts 9.664 GF, which is precisely the qkv and output
    projections with the 6.442 GF attention term missing. At long context that error is most of
    the step.

    So a counted step runs SDPA and the real steps run whatever the config chose. `backend` is
    restored alongside the usability flag because `_attend` deliberately raises when
    `backend="flash4"` cannot use its kernel, and that guard would fire here.
    """
    saved = [
        (layer, layer.backend, layer._flash4_usable)
        for layer in module.modules()
        if isinstance(layer, _AttentionBase)
    ]
    for layer, _, _ in saved:
        layer.backend = "sdpa"
        layer._flash4_usable = False
    try:
        yield
    finally:
        for layer, backend, usable in saved:
            layer.backend = backend
            layer._flash4_usable = usable


class _AttentionBase(nn.Module):
    """Kernel selection and dispatch, shared by self- and cross-attention.

    Only the projections differ between the two: self-attention packs q, k and v into one matmul,
    which cross-attention cannot do because its keys and values come from somewhere else. Picking
    the kernel, applying it, and explaining why it could not run are identical, so they live here.

    `backend` selects the kernel: "auto" prefers FlashAttention-4 whenever it can actually run
    and falls back to `scaled_dot_product_attention`, "flash4" demands it and fails loudly if it
    is unavailable, and "sdpa" always uses torch. Failing loudly matters for benchmarking — a run
    that silently used the slower kernel would be easy to misread as a kernel that did not help.
    """

    def __init__(self, dim: int, num_heads: int, backend: str = "auto") -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.backend = backend

        usable, reason = flash4_status()
        if backend == "flash4" and not usable:
            raise ValueError(f"backend='flash4' was requested but is unusable: {reason}")
        self._flash4_usable = usable and backend != "sdpa"

    def selected_backend(self, dtype: torch.dtype, on_cuda: bool) -> str:
        """Which kernel a forward pass with these inputs would use: "flash4" or "sdpa".

        Exposed because the choice is otherwise invisible from outside, and the two kernels can
        agree numerically to the bit on small inputs — so output alone cannot tell you which one
        ran, and a benchmark that quietly used SDPA looks like a kernel that did not help.
        """
        if self._flash4_usable and on_cuda and dtype in _FLASH4_DTYPES:
            return "flash4"
        return "sdpa"

    def _attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, on_cuda: bool
    ) -> torch.Tensor:
        """(B, N, heads, head_dim) inputs -> (B, N, heads, head_dim) attended output."""
        if self.selected_backend(query.dtype, on_cuda) == "flash4":
            return self._flash4_attention(query, key, value)
        if self.backend == "flash4":
            raise ValueError(
                f"backend='flash4' cannot run on {query.dtype} inputs on "
                f"{'cuda' if on_cuda else 'cpu'}: its kernels take "
                f"{' or '.join(str(d).removeprefix('torch.') for d in _FLASH4_DTYPES)} on a CUDA "
                "device. Set precision to bf16, or use the auto backend to fall back to SDPA."
            )
        return self._sdpa_attention(query, key, value)

    def _flash4_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        from flash_attn.cute import flash_attn_func

        # `softmax_scale` is passed explicitly even though its default matches, so the two
        # backends are numerically comparable by construction rather than by coincidence.
        # The kernel returns (out, lse); its README shows a bare tensor, but the wrapper hands
        # back whatever its autograd Function returns, which is the pair.
        attended, _ = flash_attn_func(
            query, key, value, softmax_scale=self.scale, causal=False
        )
        return attended

    def _sdpa_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        # (B, N, H, Dh) -> (B, H, N, Dh) for SDPA, and back again so both backends return the
        # same layout to the caller.
        query, key, value = (t.transpose(1, 2) for t in (query, key, value))
        attended = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
        return attended.transpose(1, 2)


class SelfAttention(_AttentionBase):
    """Full (unmasked) multi-head self-attention over a token sequence."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        backend: str = "auto",
        qkv_bias: bool = True,
    ) -> None:
        super().__init__(dim, num_heads, backend)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        rope: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """(B, N, dim) -> (B, N, dim).

        `rope` is any callable that rotates a (B, N, heads, head_dim) tensor in place of its
        position-free self, applied to queries and keys but not values. Typed as a plain callable
        so this layer stays independent of which position encoding is in use.
        """
        batch, tokens, _ = x.shape

        # (B, N, 3, H, head_dim). FlashAttention-4 reads (batch, seq, heads, dim) and so consumes
        # this directly; SDPA wants heads ahead of tokens and needs the transpose below.
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)

        if rope is not None:
            # Values carry content, not position, so they are left alone; rotating them would
            # make the attention output itself position-dependent rather than the weights.
            query, key = rope(query), rope(key)

        attended = self._attend(query, key, value, x.is_cuda)
        return self.proj(attended.reshape(batch, tokens, -1))


class CrossAttention(_AttentionBase):
    """Multi-head attention from one sequence onto another.

    Queries come from `x`, keys and values from `context`, so the two may differ in length. Used by
    masked autoencoding to let the tokens being reconstructed read from the encoder's visible
    tokens. Keys and values share one projection because they always come from the same tensor;
    queries need their own, which is the whole reason this cannot reuse `SelfAttention`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        backend: str = "auto",
        qkv_bias: bool = True,
    ) -> None:
        super().__init__(dim, num_heads, backend)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope: Callable[[torch.Tensor], torch.Tensor] | None = None,
        context_rope: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """(B, N, dim) queries against (B, M, dim) context -> (B, N, dim).

        Both position encodings are applied when given: rotating only the keys would make the
        logits depend on the keys' absolute positions instead of on the displacement between a
        query and a key, which is the only thing a rotary encoding is meant to express.
        """
        batch, tokens, _ = x.shape
        context_len = context.shape[1]

        query = self.to_q(x).reshape(batch, tokens, self.num_heads, self.head_dim)
        kv = self.to_kv(context).reshape(
            batch, context_len, 2, self.num_heads, self.head_dim
        )
        key, value = kv.unbind(dim=2)

        if rope is not None:
            query = rope(query)
        if context_rope is not None:
            key = context_rope(key)

        attended = self._attend(query, key, value, x.is_cuda)
        return self.proj(attended.reshape(batch, tokens, -1))
