"""Config-string to layer-class lookups, and the shared DINOv3 weight init.

The DINOv3 model classes take their norm, FFN and RoPE dtype choices as strings so a run is fully
described by a config file. These tables are what turn those strings into classes, and they are
shared by the 2D and 3D models -- which is why they live here rather than in either model module.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from layers.common.layer_scale import LayerScale
from layers.common.rms_norm import RMSNorm

from .attention import LinearKMaskedBias
from .ffn import Mlp, SwiGLUFFN
from .patch_embed import PatchEmbed, PatchEmbed3D

ffn_layer_dict: dict[str, Callable[..., Mlp | SwiGLUFFN]] = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict: dict[str, Callable[..., nn.Module]] = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = "") -> None:
    """Per-module weight init, applied across the tree by `utils.module_ops.named_apply`.

    Modules that own a `reset_parameters` are asked to run it, rather than being initialised from
    here, so each layer keeps its own scheme in one place.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, LinearKMaskedBias) and module.bias is not None:
        # `LinearKMaskedBias` fuses q, k and v; zero the mask over the key third so its bias
        # cannot contribute, and leave q and v enabled.
        o = module.out_features
        module.bias_mask.fill_(1)
        module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    # PatchEmbed3D is checked alongside PatchEmbed: upstream lists only the 2D one, which silently
    # left the 3D patch convolution on PyTorch's default init instead of the fan-in-scaled uniform
    # both classes define in `reset_parameters`.
    if isinstance(module, PatchEmbed | PatchEmbed3D):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()
