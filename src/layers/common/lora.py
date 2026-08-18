"""Low-rank adaptation of a linear layer: a frozen base weight plus a trainable rank-`r` delta.

From "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al.). An adapted layer computes
`base(x) + (alpha / r) * B @ A @ x`, with `A` drawn from the same fan-in-scaled uniform torch uses
for a Linear and `B` initialised to **zero** -- so the delta is exactly zero at construction and an
adapted model starts out computing bit-for-bit what the unadapted one did. That property is the
whole point when the weights being adapted are a pretrained prior worth preserving, and
`tests/unit/test_lora.py` pins it as an equality rather than a tolerance.

**Adaptation is in place, by class promotion, and that is deliberate.** `promote` reassigns
`__class__` on the existing module rather than building a replacement and copying into it, so the
`weight` and `bias` Parameters, every registered buffer, and above all every *parameter name* stay
the objects and names they already were. The alternative -- a wrapper module holding the original as
a `base_layer` child -- renames `blocks.0.attn.qkv.weight` to `blocks.0.attn.qkv.base_layer.weight`,
and that rename would have to be taught to `utils.pretrained.load_pretrained`, to DCP resume, and to
every downstream config that loads an adapted checkpoint. Keeping the names makes an adapted state
dict a strict *superset* of a plain one: a released DINOv3 checkpoint loads into an adapted model
under `strict = true` with no special casing, and the adapter tensors simply arrive as
`kept_initial`.

The promoted class is synthesised per base class and cached. Deriving it rather than listing the
Linear variants keeps this module free of any dependency on a particular architecture's layers --
`layers/common/` is shared across architectures and must not reach into `layers/dinov3/` -- and it
is also what makes the adapter compose *correctly* with a Linear subclass that overrides `forward`:
`LoRAMixin` precedes the base class in the MRO, so its `super().forward(...)` is that subclass's own
forward. DINOv3's `LinearKMaskedBias`, whose forward applies a 0/1 mask to the fused qkv bias, is
adapted by this without knowing anything about it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAMixin:
    """The low-rank delta, mixed into an `nn.Linear` subclass ahead of it in the MRO."""

    # Declared for type checking: the mixin reads what its Linear half provides.
    in_features: int
    out_features: int
    weight: nn.Parameter
    lora_a: nn.Parameter
    lora_b: nn.Parameter
    lora_rank: int
    lora_scaling: float

    def init_lora(self, rank: int, alpha: float) -> None:
        """Add the adapter's parameters. Called by `promote`, once, after class reassignment."""
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.lora_rank = rank
        options = {"device": self.weight.device, "dtype": self.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, **options))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, **options))
        # A **persistent buffer**, not a float, so the scaling travels in the state dict.
        # `alpha / rank` is not recoverable from the stored tensors -- `rank` is `lora_a.shape[0]`
        # but `alpha` is nowhere -- and folding an adapter into its base weight later needs it
        # exactly. Without this, merging a checkpoint would have to take the scaling from whatever
        # `[lora].alpha` the *merging* run happened to set, and an alpha that had changed since the
        # adapter was trained would rescale the delta with nothing to show it. Registering it makes
        # the checkpoint self-describing instead. It is one scalar per adapter.
        self.register_buffer(
            "lora_scaling", torch.tensor(alpha / rank, **options), persistent=True
        )
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        """`A` fan-in-scaled uniform, `B` exactly zero.

        Deliberately not the `trunc_normal_(std=0.02)` that `layers.dinov3.config.init_weights_vit`
        applies to every `nn.Linear.weight` it walks past: these are factors of a delta, not a
        weight matrix, and `B` has to be *exactly* zero rather than merely small. The two schemes
        never collide, because that function initialises attributes it knows by name and does not
        know these -- but that is why the adapter initialises itself here rather than relying on it.
        """
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def lora_delta(self) -> torch.Tensor:
        """The adapter's contribution to the effective weight, shaped like `weight`."""
        return self.lora_scaling * (self.lora_b @ self.lora_a)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Two thin matmuls rather than materialising `lora_delta()` and adding it to `weight`: at
        # rank 16 into a 1024 -> 3072 projection the factored form is ~48x less arithmetic, and it
        # keeps the frozen base weight out of the autograd graph.
        base = super().forward(input)  # type: ignore[misc]  # provided by the Linear half
        return base + F.linear(F.linear(input, self.lora_a), self.lora_b) * self.lora_scaling

    @torch.no_grad()
    def merge_lora(self) -> None:
        """Fold the delta into `weight` and drop the adapter, leaving a plain layer behind.

        Turns an adapted layer into exactly the unadapted layer it is equivalent to, which is what
        lets an adapted checkpoint be consumed by a config that knows nothing about LoRA. It
        reverses `promote` completely -- class and parameter set both -- so calling it twice raises
        rather than folding the same delta in a second time.
        """
        self.weight.add_(self.lora_delta().to(self.weight.dtype))
        del self.lora_a
        del self.lora_b
        del self.lora_rank
        del self.lora_scaling
        self.__class__ = _base_class(type(self))


# Synthesised promoted classes, keyed by the base class they adapt. Cached because `__class__`
# assignment compares types by identity: building a fresh class per layer would make every adapted
# layer its own type, so `isinstance(module, LoRAMixin)` would still work but nothing else would
# recognise two adapted qkv projections as the same kind of thing.
_PROMOTED: dict[type[nn.Linear], type[nn.Linear]] = {}


def _promoted_class(base: type[nn.Linear]) -> type[nn.Linear]:
    if base not in _PROMOTED:
        _PROMOTED[base] = type(f"LoRA{base.__name__}", (LoRAMixin, base), {})
    return _PROMOTED[base]


def _base_class(promoted: type) -> type[nn.Linear]:
    """The class `promoted` was derived from, read back off its MRO."""
    return promoted.__bases__[1]


def promote(linear: nn.Linear, *, rank: int, alpha: float) -> nn.Linear:
    """Give `linear` a low-rank adapter, in place. Returns it, for use in an expression.

    Exact on the forward pass at the moment it returns, since `B` is zero.
    """
    if isinstance(linear, LoRAMixin):
        raise ValueError(
            f"{type(linear).__name__} is already adapted; promoting it twice would stack a second "
            "adapter on the first and silently double the delta"
        )
    if not isinstance(linear, nn.Linear):
        raise TypeError(
            f"LoRA adapts nn.Linear layers; got {type(linear).__name__}. A convolutional stem is "
            "adapted by training it outright, not by a low-rank factorisation of its kernel."
        )
    linear.__class__ = _promoted_class(type(linear))
    linear.init_lora(rank, alpha)  # type: ignore[attr-defined]  # class was just reassigned
    return linear


def adapted_modules(root: nn.Module) -> list[tuple[str, LoRAMixin]]:
    """Every adapted layer under `root`, by qualified name."""
    return [
        (name, module) for name, module in root.named_modules() if isinstance(module, LoRAMixin)
    ]


def merge_all(root: nn.Module) -> list[str]:
    """Fold every adapter under `root` into its base weight. Returns the names merged."""
    merged = []
    for name, module in adapted_modules(root):
        module.merge_lora()
        merged.append(name)
    return merged
