"""DINOv3's 2D vision transformer.

Ported from the DINOv3 reference implementation so the architecture lives here in full: nothing
imports the `dinov3` package at runtime, and the building blocks come from `layers/dinov3/`.

What makes this a DINOv3 ViT rather than a plain one:
  - **No position embedding table.** Position enters through rotary embeddings inside attention,
    computed from the patch grid on every forward pass, so a single model handles crops of
    different sizes and aspect ratios. Coordinates are augmented during training
    (shift/jitter/rescale), which is what makes that robustness hold.
  - **Storage (register) tokens.** Extra learned tokens with no position, prepended alongside CLS.
    They give attention somewhere to park global information that would otherwise be dumped into
    high-norm patch tokens and corrupt dense features.
  - **A mask token**, so masked-image modelling can replace patch embeddings in place.
  - **Optionally untied norms** for CLS versus patch tokens, and for global versus local crops --
    the two token populations have different statistics once training is under way.

The self-supervised training machinery itself (DINO head, teacher/student EMA, multi-crop
augmentation) is deliberately not ported: this is the backbone only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn as nn

from layers.dinov3.block import SelfAttentionBlock
from layers.dinov3.config import dtype_dict, ffn_layer_dict, init_weights_vit, norm_layer_dict
from layers.dinov3.ffn import SwiGLUFFN
from layers.dinov3.patch_embed import PatchEmbed
from layers.dinov3.rope import RopePositionEmbedding
from utils.module_ops import named_apply

from .base import BaseModel
from .registry import ModelRegistry

SPATIAL_RANK = 2


@ModelRegistry.register("dinov3_vit")
class DinoVisionTransformer(BaseModel):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float | None = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        use_fa4: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
    ):
        super().__init__()

        # Validated rather than allowed to fail deep inside a dict lookup, because these come
        # straight from a config file.
        if norm_layer not in norm_layer_dict:
            raise ValueError(
                f"unknown norm_layer {norm_layer!r}; expected one of {sorted(norm_layer_dict)}"
            )
        if ffn_layer not in ffn_layer_dict:
            raise ValueError(
                f"unknown ffn_layer {ffn_layer!r}; expected one of {sorted(ffn_layer_dict)}"
            )
        if pos_embed_rope_dtype not in dtype_dict:
            raise ValueError(
                f"unknown pos_embed_rope_dtype {pos_embed_rope_dtype!r}; expected one of "
                f"{sorted(dtype_dict)}"
            )

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dim, device=device)
            )

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )

        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                use_fa4=use_fa4,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm: nn.Module | None = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm: nn.Module | None = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        # Upstream leaves this to the caller -- every DINOv3 entry point runs `init_weights()`
        # right after constructing the backbone. mia-train has no such hook: `ModelRegistry.build`
        # is the only production path and it just calls the constructor, so an uninitialised model
        # would train on whatever `torch.empty` returned. Sibling models (`ViT3D`, `MuViT3D`)
        # likewise finish initialising inside `__init__`.
        self.init_weights()

    def init_weights(self) -> None:
        """Fill every parameter this architecture leaves uninitialised in `__init__`.

        Called from `__init__`; public because it is upstream's API and because re-running it is
        the way to reset a model in place.
        """
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    @property
    def grid_size(self) -> tuple[int, int]:
        return (self.img_size // self.patch_size, self.img_size // self.patch_size)

    @property
    def num_patches(self) -> int:
        return int(math.prod(self.grid_size))

    def prepare_tokens_with_masks(
        self, x: torch.Tensor, masks=None
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)  # B H*W D

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token: torch.Tensor = self.cls_token
        else:
            # Keeps `mask_token` in the autograd graph even when unused, so DDP/FSDP do not trip
            # over a parameter that receives no gradient on some ranks.
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens: torch.Tensor = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1, 0, cls_token.shape[-1], dtype=cls_token.dtype, device=cls_token.device
            )

        x = torch.cat([cls_token.expand(B, -1, -1), storage_tokens.expand(B, -1, -1), x], dim=1)

        return x, (H, W)

    def forward_features_list(
        self, x_list: list[torch.Tensor], masks_list: list[torch.Tensor | None]
    ) -> list[dict[str, torch.Tensor]]:
        tokens_list = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list, strict=True):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            tokens_list.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for _ in rope]
            tokens_list = blk(tokens_list, rope_sincos)
        all_x = tokens_list
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list, strict=True)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    assert self.local_cls_norm is not None  # implied by the flag
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                if self.untie_cls_and_patch_norms:
                    assert self.cls_norm is not None  # implied by the flag
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self,
        x: torch.Tensor | list[torch.Tensor],
        masks: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        if isinstance(x, torch.Tensor):
            assert not isinstance(masks, list), "a single input takes a single mask, not a list"
            return self.forward_features_list([x], [masks])[0]
        assert isinstance(masks, list), "a list of crops needs a matching list of masks"
        return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(
        self, x: torch.Tensor, n: int | Sequence[int] = 1
    ) -> list[torch.Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: int | Sequence[int] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], ...]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    assert self.cls_norm is not None  # implied by the flag
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens, strict=True))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens, strict=True))
        else:
            return tuple(zip(outputs, class_tokens, extra_tokens, strict=True))

    def forward(self, *args, is_training: bool = False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        # The list form is only ever used during training, where `is_training` is set.
        assert isinstance(ret, dict), "forward() over a list of crops requires is_training=True"
        return self.head(ret["x_norm_clstoken"])

    def patch_features(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """(B, C, H, W) -> normalised patch tokens (B, N, embed_dim) and their grid.

        CLS and storage tokens are dropped: they carry no position on the grid, so a dense head
        has nowhere to put them. The grid is read off the input rather than taken from
        `grid_size`, so a crop at a resolution other than the configured `img_size` still folds
        back correctly.
        """
        out = self.forward_features(x)
        assert isinstance(out, dict)  # a single tensor in gives the dict form back
        grid = tuple(s // self.patch_size for s in x.shape[-SPATIAL_RANK:])
        return out["x_norm_patchtokens"], grid  # type: ignore[return-value]

    def checkpointable_modules(self) -> tuple[nn.Module, ...]:
        """The transformer blocks: repeated, sequence-length-sized, and cheap to rerun."""
        return tuple(self.blocks)

    def prepare_input(self, batch: torch.Tensor, axes: str) -> torch.Tensor:
        """(B, *axes) -> (B, C, H, W). Single-scale, single-plane.

        Mirrors `ViT3D.prepare_input` one spatial rank down: miao's scale levels are not
        interchangeable with independent samples, so rather than guess how to fold them this
        requires exactly one level and says how to configure it.
        """
        if "l" not in axes:
            raise ValueError(f"axis order must contain 'l' (scale level), got {axes!r}")

        remainder = axes.replace("l", "", 1)
        if not (
            len(remainder) == SPATIAL_RANK
            or (len(remainder) == SPATIAL_RANK + 1 and remainder[0] == "c")
        ):
            raise ValueError(
                f"axis order {axes!r} is not usable by a 2D encoder: after the level axis it "
                "must be two spatial axes, optionally preceded by 'c' (e.g. \"lyx\" or "
                f'"lcyx"), got {remainder!r}'
            )

        expected_dims = len(axes) + 1
        if batch.dim() != expected_dims:
            raise ValueError(
                f"axis order {axes!r} implies a {expected_dims}-D batch (batch + {len(axes)} "
                f"axes), got {tuple(batch.shape)}; it must match the dataset's output_axes"
            )

        level_dim = axes.index("l") + 1
        levels = batch.shape[level_dim]
        if levels != 1:
            raise ValueError(
                f"{type(self).__name__} is single-scale, but this batch carries {levels} scale "
                f"levels on axis 'l' (shape {tuple(batch.shape)}). Configure the dataset for one "
                "level per sample: give `resolutions` a single entry, or use "
                "`resolution_sampling` with `n_scales = 1`, which still varies the resolution but "
                "draws it independently per sample."
            )

        images = batch.squeeze(level_dim)
        if images.dim() == SPATIAL_RANK + 1:  # axis order declared no channel; add a singleton
            images = images.unsqueeze(1)
        return images

    def extra_forward_methods(self) -> tuple[str, ...]:
        """Both are entry points in their own right: SSL training drives `forward_features`
        directly, and linear-probe evaluation drives `get_intermediate_layers`, so FSDP2 has to
        all-gather around them as well as around `forward`."""
        return ("forward_features", "get_intermediate_layers")

    def flops(self, input_shape: tuple[int, ...]) -> int:
        """Rough forward FLOPs for one input of `input_shape`: patch embedding, attention, MLPs.

        The grid is derived from `input_shape` rather than validated against the configured
        `img_size` the way `ViT3D.flops` does, because unlike `ViT3D` this architecture really does
        run at many resolutions: it carries no position-embedding table to interpolate, DINOv3 SSL
        pushes global and local crops of different sizes through `forward_features_list` inside a
        single step, and `patch_features` reads the grid off its input for that reason. Pinning the
        estimate to `img_size` reports one number for all of them.

        Floor division matches the patch convolution, which is strided by `patch_size` and simply
        drops a ragged edge; nothing upstream rejects an input that is not a whole number of
        patches, so nothing here should either.
        """
        if len(input_shape) < SPATIAL_RANK:
            raise ValueError(
                f"input_shape {tuple(input_shape)} does not describe an input this model can run: "
                f"it needs at least {SPATIAL_RANK} axes to have a patch grid, e.g. (C, H, W)"
            )
        # The spatial extent is free, per the paragraph above, but the channel count is not: the
        # patch convolution is built for `in_chans` and would raise on anything else, so a shape
        # naming a different one describes a forward pass this model cannot run. It is optional
        # only in the sense that a caller may name the image alone and assert nothing about
        # channels; when it is named it is checked, because `patch_volume` below is linear in it
        # and the answer would otherwise come back scaled by the ratio of the two counts, which is
        # indistinguishable from a correct one.
        if len(input_shape) > SPATIAL_RANK and input_shape[-SPATIAL_RANK - 1] != self.in_chans:
            raise ValueError(
                f"input_shape {tuple(input_shape)} does not describe an input this model can run: "
                f"the axis before its last {SPATIAL_RANK} is the channel axis and must be the "
                f"configured in_chans {self.in_chans}"
            )

        grid = tuple(s // self.patch_size for s in input_shape[-SPATIAL_RANK:])
        num_patches = int(math.prod(grid))
        n = num_patches + 1 + self.n_storage_tokens
        d = self.embed_dim
        depth = len(self.blocks)

        # The patch convolution sees only the patch tokens; CLS and the storage tokens are learned
        # parameters spliced into the sequence afterwards, so `num_patches` is right here and `n`
        # is right everywhere below.
        patch_volume = self.patch_size**SPATIAL_RANK * self.in_chans
        patch_proj = 2 * num_patches * patch_volume * d

        # The FFN width is read off the built block rather than recomputed as `int(d * ffn_ratio)`,
        # because that expression is only the *nominal* hidden dim: `SwiGLUFFN` runs three
        # projections at 2/3 of it rounded up to its `align_to`, and the two coincide only when
        # 2/3 of the nominal width is already a multiple of that alignment. It is not for narrow
        # models -- `embed_dim=32` with `swiglu64` rounds 85 up to 128 -- and there is no way to
        # notice from here, so the built module is the only honest source.
        block = self.blocks[0]
        assert isinstance(block, SelfAttentionBlock)  # nn.ModuleList erases its element type
        mlp = block.mlp
        if isinstance(mlp, SwiGLUFFN):
            ffn = 2 * (3 * n * d * mlp.w1.out_features)
        else:
            ffn = 2 * (2 * n * d * mlp.fc1.out_features)

        per_block = 2 * (4 * n * d * d) + 2 * (2 * n * n * d) + ffn
        return int(patch_proj + depth * per_block)
