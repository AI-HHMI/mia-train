"""A hierarchical 3D convolutional encoder whose spatial mixing is dense 3^3 convolutions.

ConvNeXt's layout (Liu et al., 2022): a patchify stem, stages that halve the resolution and widen
the channels, pre-norm residual blocks with layer scale. The block's spatial operation is
changed, because in 3D ConvNeXt's depthwise convolution is the wrong operation for a GPU. On a
B300, PyTorch's 3D depthwise kernel runs at ~2% of the CUDA cores' peak, while a dense 3^3
convolution reaches 38-71% of the tensor cores' peak from 64 channels up
(experiments/large_inputs/README.md, phase 3). Two blocks:

  basic   ResNet's basic block (He et al., 2016): LN -> 3^3 -> GELU -> 3^3
  fused   EfficientNetV2's Fused-MBConv (Tan & Le, 2021): LN -> 3^3 to `expansion` x width ->
          GELU -> 1^3 back

Both are written as pre-norm residuals, x + layer_scale * f(LN(x)), as ConvNeXt's and the ViTs'
blocks are. LayerNorm rather than the originals' BatchNorm, because it normalizes each position
on its own. Nothing then depends on the batch, which is one crop per GPU at large crops. Nothing
depends on the crop size either, and predictions run at other crop sizes than training.

Everything runs channels-last, (B, D, H, W, C). cuDNN's dense 3D convolutions are fastest in that
layout, the norms and 1^3 projections become row-wise ops on it, and the last stage's output is
already the (B, N, C) token sequence a dense head takes. The stem and the downsampling between
stages are ConvNeXt's strided convolutions with kernel == stride. They are computed as
space-to-depth plus a matmul, which has no 2^31-element limit.

The encoder hands the head its last stage, at stride `patch_size` = 4 x 2^(stages - 1), through
`patch_features`. That is the interface a ViT's patch tokens use, so the dense heads work unchanged.

Parameters are named as the ViTs name theirs, because `engine.optimizer` (layerwise learning-rate
decay) and backbone freezing read depth off the names. The stem is `patch_embed`, and every block
is `blocks.<i>` in one flat list across the stages. The downsampling into a stage belongs to that
stage's first block, where its depth is that block's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, single_scale_volumes
from .registry import ModelRegistry

SPATIAL_RANK = 3
#: cuDNN indexes a convolution's tensors with 32 bits. Past 2^31 elements it falls back to kernels
#: measured ~35x slower per FLOP (gary_comparison's UNETR probe), so `Conv3x3` stays under it.
INDEX_LIMIT = 2**31
BLOCKS = ("basic", "fused")


class Conv3x3(nn.Conv3d):
    """A dense 3^3 convolution on a channels-last (B, D, H, W, C) tensor.

    An input or output of 2^31 elements or more runs in slabs along D. Each slab is read with one
    halo plane on either side and cut back to its own planes afterwards. That is exact, because an
    output plane reads only its own input plane and the two beside it.
    """

    def __init__(self, cin: int, cout: int, device: torch.device | None = None) -> None:
        super().__init__(cin, cout, 3, padding=1, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depth = x.shape[1]
        plane = x.shape[0] * x.shape[2] * x.shape[3] * max(self.in_channels, self.out_channels)
        planes = (INDEX_LIMIT - 1) // plane - 2  # a slab's own planes, leaving room for two halos
        if planes >= depth:
            return self._conv(x)
        if planes < 1:
            raise ValueError(
                f"one {tuple(x.shape[2:4])} plane at {max(self.in_channels, self.out_channels)} "
                "channels is already near 2^31 elements, so no slab along D brings this "
                "convolution under cuDNN's 32-bit indexing"
            )
        slabs = []
        for lo in range(0, depth, planes):
            hi = min(lo + planes, depth)
            start, stop = max(lo - 1, 0), min(hi + 1, depth)
            slabs.append(self._conv(x[:, start:stop])[:, lo - start : hi - start])
        return torch.cat(slabs, dim=1)

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        # (B, D, H, W, C) -> the channels-last (B, C, D, H, W) view cuDNN takes, and back.
        return super().forward(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


class PatchMerge(nn.Module):
    """(B, D, H, W, Cin) -> (B, D/f, H/f, W/f, Cout): a convolution with kernel == stride == f.

    ConvNeXt's stem (f = 4) and downsampling (f = 2), as space-to-depth followed by a linear map.
    The function is the same, since each output position reads exactly its own f^3 block, but
    cuBLAS has no 2^31-element limit and the layout stays channels-last.
    """

    def __init__(
        self, cin: int, cout: int, factor: int, device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(factor**SPATIAL_RANK * cin, cout, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w, c = x.shape
        f = self.factor
        blocks = x.reshape(b, d // f, f, h // f, f, w // f, f, c).permute(0, 1, 3, 5, 2, 4, 6, 7)
        return self.proj(blocks.reshape(b, d // f, h // f, w // f, f**SPATIAL_RANK * c))


class BasicBlock(nn.Module):
    """ResNet's basic block as a pre-norm residual: x + gamma * conv(GELU(conv(LN(x))))."""

    def __init__(
        self,
        dim: int,
        layerscale_init: float,
        downsample: nn.Module | None,
        device: torch.device | None,
    ) -> None:
        super().__init__()
        self.downsample = downsample  # into this block's stage, when it is the stage's first
        self.norm = nn.LayerNorm(dim, eps=1e-6, device=device)
        self.conv1 = Conv3x3(dim, dim, device=device)
        self.conv2 = Conv3x3(dim, dim, device=device)
        self.gamma = nn.Parameter(torch.full((dim,), layerscale_init, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        return x + self.gamma * self.conv2(F.gelu(self.conv1(self.norm(x))))


class FusedBlock(nn.Module):
    """EfficientNetV2's Fused-MBConv as a pre-norm residual: x + gamma * proj(GELU(conv(LN(x)))).

    The 3^3 convolution widens to `expansion` x dim and the 1^3 projection brings it back. This is
    MobileNetV2's inverted bottleneck with its 1^3 expansion and depthwise 3^3 fused into one
    dense convolution, which is what EfficientNetV2 did where depthwise convolutions were slow.
    """

    def __init__(
        self,
        dim: int,
        expansion: int,
        layerscale_init: float,
        downsample: nn.Module | None,
        device: torch.device | None,
    ) -> None:
        super().__init__()
        self.downsample = downsample  # into this block's stage, when it is the stage's first
        self.norm = nn.LayerNorm(dim, eps=1e-6, device=device)
        self.conv = Conv3x3(dim, expansion * dim, device=device)
        self.proj = nn.Linear(expansion * dim, dim, device=device)
        self.gamma = nn.Parameter(torch.full((dim,), layerscale_init, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        return x + self.gamma * self.proj(F.gelu(self.conv(self.norm(x))))


def _init_weights(module: nn.Module) -> None:
    # DINOv3's ConvNeXt initialisation.
    if isinstance(module, (nn.Conv3d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@ModelRegistry.register("convnet3d")
class ConvNet3D(BaseModel):
    """Stem to stride 4, then one stage per entry of `widths`, each at twice the last one's stride.

    Args:
        img_size: the crop side it is configured for. Any multiple of `patch_size` runs, since
            nothing here is sized by the crop; this only sets `num_patches`.
        in_chans: input channels.
        block: "basic" or "fused" (see the module docstring).
        widths: channels per stage. Stage i runs at stride 4 x 2^i.
        depths: blocks per stage.
        expansion: how far a "fused" block's 3^3 convolution widens. A "basic" block has none.
        layerscale_init: the initial layer scale of every block.
        device: where to build the parameters.
    """

    def __init__(
        self,
        img_size: int = 256,
        in_chans: int = 1,
        block: str = "basic",
        widths: Sequence[int] = (64, 256, 768),
        depths: Sequence[int] = (1, 4, 12),
        expansion: int = 4,
        layerscale_init: float = 1e-6,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if block not in BLOCKS:
            raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")
        if not widths or len(widths) != len(depths):
            raise ValueError(
                f"widths {tuple(widths)} and depths {tuple(depths)} must name the same, nonzero "
                "number of stages"
            )
        if min(depths) < 1:
            raise ValueError(
                f"depths {tuple(depths)}: every stage needs a block, because a stage's "
                "downsampling is its first block's"
            )
        self.block = block
        self.widths, self.depths = tuple(widths), tuple(depths)
        self.expansion = expansion
        self.in_chans = in_chans
        self.patch_size = 4 * 2 ** (len(self.widths) - 1)
        self.num_features = self.embed_dim = self.widths[-1]
        if img_size % self.patch_size:
            raise ValueError(
                f"img_size {img_size} must be a multiple of the output stride {self.patch_size} "
                f"that {len(self.widths)} stages give"
            )
        self.img_size = img_size
        self.num_patches = (img_size // self.patch_size) ** SPATIAL_RANK
        # What `pyramid_features` returns, for a head built before anything runs.
        self.pyramid_dims = self.widths
        self.pyramid_strides = tuple(4 * 2**index for index in range(len(self.widths)))
        #: Exclusive end, in `blocks`, of each stage.
        self.stage_ends = tuple(sum(self.depths[: index + 1]) for index in range(len(self.depths)))

        def make(width: int, downsample: nn.Module | None) -> nn.Module:
            if block == "basic":
                return BasicBlock(width, layerscale_init, downsample, device)
            return FusedBlock(width, expansion, layerscale_init, downsample, device)

        self.patch_embed = nn.Sequential(
            PatchMerge(in_chans, self.widths[0], 4, device),
            nn.LayerNorm(self.widths[0], eps=1e-6, device=device),
        )
        blocks: list[nn.Module] = []
        for index, (width, depth) in enumerate(zip(self.widths, self.depths, strict=True)):
            for position in range(depth):
                downsample = None
                if index and position == 0:
                    cin = self.widths[index - 1]
                    downsample = nn.Sequential(
                        nn.LayerNorm(cin, eps=1e-6, device=device),
                        PatchMerge(cin, width, 2, device),
                    )
                blocks.append(make(width, downsample))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6, device=device)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """(B, C, D, H, W) -> every stage's output, finest first, channels-last (B, *grid_i, C_i).

        The last passes the final norm and is what `patch_features` hands a head. The others are the
        stream as it leaves each stage, the skips `pyramid_features` hands a U-Net decoder.
        """
        spatial = tuple(x.shape[-SPATIAL_RANK:])
        if any(extent % self.patch_size for extent in spatial):
            raise ValueError(
                f"crop {spatial} must be a multiple of the output stride {self.patch_size} on "
                "every axis"
            )
        x = self.patch_embed(x.permute(0, 2, 3, 4, 1))
        stages = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index + 1 in self.stage_ends:
                stages.append(x)
        stages[-1] = self.norm(stages[-1])
        return stages

    def patch_features(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """(B, C, D, H, W) -> the last stage as (B, N, embed_dim) grid-order tokens, and the grid.

        Through `self(x)` rather than `forward`, so that FSDP2's forward hooks gather the
        parameters outside the blocks (the stem and the final norm), as on any forward call.
        """
        features = self(x)[-1]
        batch, *grid, channels = features.shape
        return features.reshape(batch, -1, channels), tuple(grid)

    def pyramid_features(self, x: torch.Tensor) -> tuple[list[torch.Tensor], tuple[int, ...]]:
        """(B, C, D, H, W) -> each stage's output as (B, C_i, *grid_i), finest first, and its grid.

        Channels-last views of the stage outputs, with no copy, which is the layout cuDNN's dense
        convolutions are fastest in. Through `self(x)` for FSDP2's hooks, as `patch_features` is.
        """
        stages = [stage.permute(0, 4, 1, 2, 3) for stage in self(x)]
        return stages, tuple(stages[-1].shape[2:])

    def prepare_input(self, batch: torch.Tensor, axes: str) -> torch.Tensor:
        """(B, *axes) -> (B, C, D, H, W); see `models.base.single_scale_volumes`."""
        return single_scale_volumes(self, batch, axes)

    def checkpointable_modules(self) -> tuple[nn.Module, ...]:
        """Every block: repeated, sized by the volume, and cheap to rerun against what it stores."""
        return tuple(self.blocks)

    def fsdp_units(self) -> tuple[nn.Module, ...]:
        """Every block, so a sharded run gathers one block's weights at a time."""
        return tuple(self.blocks)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        """Forward FLOPs for one input: the stem, the downsampling and every block's matmuls.

        Norms, activations, biases and the layer scale are left out, as the ViTs leave theirs out.
        Any multiple of `patch_size` runs, so the extent is read from `input_shape`, and one that
        `forward` would refuse is refused here too.
        """
        if len(input_shape) < SPATIAL_RANK:
            raise ValueError(
                f"input_shape {tuple(input_shape)} needs at least {SPATIAL_RANK} spatial axes"
            )
        if len(input_shape) > SPATIAL_RANK and input_shape[-SPATIAL_RANK - 1] != self.in_chans:
            raise ValueError(
                f"input_shape {tuple(input_shape)} has a channel axis other than the configured "
                f"in_chans {self.in_chans}"
            )
        spatial = input_shape[-SPATIAL_RANK:]
        if any(extent % self.patch_size for extent in spatial):
            raise ValueError(
                f"input_shape {tuple(input_shape)} must be a multiple of the output stride "
                f"{self.patch_size} on every spatial axis"
            )

        def positions(stride: int) -> int:
            return math.prod(extent // stride for extent in spatial)

        total = 2 * 4**SPATIAL_RANK * self.in_chans * self.widths[0] * positions(4)
        for index, (width, depth) in enumerate(zip(self.widths, self.depths, strict=True)):
            stride = 4 * 2**index
            if index:
                cin = self.widths[index - 1]
                total += 2 * 2**SPATIAL_RANK * cin * width * positions(stride)
            if self.block == "basic":  # two width -> width 3^3 convolutions
                per_position = 2 * 2 * 27 * width**2
            else:  # width -> expansion x width at 3^3, then a 1^3 back
                per_position = 2 * 27 * width * self.expansion * width
                per_position += 2 * self.expansion * width * width
            total += depth * per_position * positions(stride)
        return total
