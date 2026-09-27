"""Getting from a ViT's patch grid back to one prediction per voxel.

Any dense task on a transformer has to undo the patch embedding: tokens sit on a grid `patch_size`
times coarser than the input on every axis, and the head has to produce a value per voxel. The
first two heads here are alternative answers from the final layer's tokens alone, and they share a
signature -- `forward(x, size)` -- so an algorithm can offer the choice as configuration without
branching at the call site. The third, `UNETRHead`, also reads intermediate layers and the raw
image, so it takes those instead (see its docstring).

`VoxelHead` interpolates and then convolves at full resolution. Every sub-token detail is therefore
the responsibility of the convolutions that follow, which act on an already-smooth field and see
only their own kernel's width of it; at patch 16 that leaves little way for structure finer than
the token spacing to appear. It is also the expensive place to compute: cost at voxel resolution
scales with positions, so a `Conv3d(64->64, k=3)` over a 256-cube is ~1.9 TMAC, comparable to a
whole ViT-L encoder over the same crop.

`SubPixelHead` instead gives each token a learned readout of its own block, so detail comes out of
the weights and the wide arithmetic stays on the patch grid -- 4096 positions rather than 16.8M for
that same crop. It is both sharper in principle and several times cheaper, at the cost of ~17M
parameters and one structural risk (see below).

Both fold the resolution change *inside* the module rather than leaving it to the caller. That is
deliberate and is what makes activation checkpointing worth anything here: a checkpointed region
stores its own inputs, so an upsampling performed outside would leave its full-resolution result
held for the whole backward pass. Inside, the stored boundary is the patch grid, thousands of times
smaller.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import ChannelLayerNorm

# Rank -> the layers and the interpolation mode that operate on it. Trilinear/bilinear rather than
# nearest because a head's output is a continuous score, not a class index.
#
# Typed as `Any` because the dispatch is the point: `nn.Conv2d` and `nn.Conv3d` differ in the
# arity of `kernel_size`, so a union of the two constructors rejects the per-rank tuple that is
# correct for whichever one was selected.
CONV: dict[int, Any] = {2: nn.Conv2d, 3: nn.Conv3d}
CONV_TRANSPOSE: dict[int, Any] = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
INTERPOLATION = {2: "bilinear", 3: "trilinear"}


class VoxelHead(nn.Sequential):
    """Upsample to `size` by interpolation, then apply the layers at that resolution.

    A plain `nn.Sequential` with the interpolation folded in, so the resolution change and the
    layers that run at that resolution are one module.

    `size` is an argument rather than a constructor value because the head is meant to serve
    whatever crop it is given -- `nn.Upsample(size=...)` would fix the output shape at build time
    and quietly mis-scale a run that validates at a different crop from the one it trains on.
    `mode` is the opposite: it follows from the data's rank, which a head cannot change between
    calls, so it is settled once at construction.
    """

    def __init__(self, *layers: nn.Module, mode: str = "trilinear") -> None:
        super().__init__(*layers)
        self.mode = mode
        # (in, out, device, dtype) -> the 1-D resampling matrix. A plain dict rather than a buffer:
        # these are derived from two integers, not learned, and putting them in `state_dict` would
        # make a checkpoint depend on which crop sizes a run happened to see.
        self._resample: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    def _axis_matrix(
        self, extent: int, target: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """`(target, extent)` matrix applying 1-D linear interpolation along one axis.

        Built by pushing an identity through `F.interpolate` itself rather than from the
        interpolation formula. That is the point: `align_corners`, the half-pixel offset and the
        boundary clamping are conventions, and a hand-derived matrix that disagreed with torch's
        on any of them would produce a head that trains perfectly well and localises boundaries
        half a voxel off. Reading the convention out of the function being replaced cannot drift
        from it.
        """
        key = (extent, target, device, dtype)
        cached = self._resample.get(key)
        if cached is None:
            basis = torch.eye(extent, device=device, dtype=torch.float32).reshape(extent, 1, extent)
            columns = F.interpolate(basis, size=target, mode="linear", align_corners=False)
            cached = columns.squeeze(1).T.contiguous().to(dtype)
            self._resample[key] = cached
        return cached

    def forward(self, x: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:  # type: ignore[override]
        # Upsampling as a matmul per axis, not `F.interpolate`. Multi-linear interpolation is
        # separable -- each axis is resampled independently -- so this computes exactly the same
        # function, and `tests/unit/test_dense_heads.py` pins that against `F.interpolate` to
        # 1e-14 in double precision.
        #
        # The reason to write it out is the backward pass. `upsample_trilinear3d_backward` reduces
        # the full-resolution gradient onto the patch grid with `atomicAdd`, which at a 16^3 grid
        # over a 256-cube means ~262k contributions contending for each input cell. Measured on an
        # H100 at exactly that shape it costs 91.5 ms per forward+backward, against 10.2 ms for
        # the matmul form -- 9x -- because a matmul's backward is a matmul, with no contention.
        # It also removes the reason the operation had to stay in fp32: the atomic version loses
        # the gradient in bf16 (relative error 0.85, i.e. noise), while the matmul accumulates on
        # tensor cores in fp32 whatever the input dtype.
        #
        # fp32 is kept anyway, and not only for continuity with what autocast used to do here: at
        # this shape it is also the *faster* option, since autocast's casts on a multi-gigabyte
        # tensor cost more than bf16 arithmetic saves (10.2 ms against 12.5 ms). So the numerics
        # are unchanged from the interpolation this replaces, exactly, and nothing is traded.
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            for axis in reversed(range(len(size))):
                matrix = self._axis_matrix(x.shape[-1], size[axis], x.device, x.dtype)
                # Resample the trailing axis, then rotate it to the front of the spatial block so
                # the next pass sees a fresh one. After `rank` passes the axes are back in order,
                # and the volume has grown one axis at a time rather than all at once -- which
                # also keeps the intermediates smaller than a single fused upsample would.
                x = (x @ matrix.T).movedim(-1, 2)
        for layer in self:
            x = layer(x)
        return x


class SubPixelHead(nn.Module):
    """`(B, in_dim, *grid)` on the patch grid -> `(B, out_channels, *grid * patch)` at voxel scale.

    The expansion is a transposed convolution with `kernel_size == stride`, which is exactly a
    per-token linear map `R^hidden -> R^(readout * prod(patch))` whose output is written into that
    token's block -- the same "sub-pixel"/pixel-shuffle construction `nn.PixelShuffle` provides in
    2D, and which torch has no 3D equivalent of. Writing it as a transposed convolution rather than
    a `Linear` followed by a hand-written fold is deliberate: the fold is a permutation, and a
    permutation that transposes two spatial axes trains perfectly well and produces a segmentation
    that is silently wrong -- the exact failure this repo's affinity task has hit before, which is
    why it checks its axis order at all.

    `readout` is the width at full resolution and is the one number to watch for cost: every tensor
    after the expansion is `readout` channels over the whole volume, so it multiplies both the
    refinement convolutions' arithmetic and what activation checkpointing has to recompute.
    `hidden` is nearly free by comparison -- it only ever exists on the patch grid.

    **Blocks are decoded independently, so the seam between them is the thing to watch.** The
    tokens themselves are not independent -- attention gives each one global context -- but their
    decodings are separately parameterised, so nothing forces the field to be continuous across a
    block face. The refinement convolutions exist for that: `refine_depth` 3-wide convolutions give
    a receptive field of `2 * refine_depth + 1` across the seam. If that proves insufficient the
    principled fix is an overlapping kernel (`kernel_size = 2 * patch_size`, the support trilinear
    interpolation itself uses), at 8x the expansion parameters.

    `zero_init_output` zeroes the last convolution, so at construction the head emits its bias
    everywhere -- the trivial constant predictor. That protects an encoder that already solves the
    task from a randomly-initialised head's meaningless gradients, and it is the right choice when
    **warm-starting a trained encoder**.

    **Turn it off when the encoder also has to learn.** With that weight at zero, the gradient
    reaching everything before it is zero too, so the encoder trains on nothing until `out` grows
    -- which under a long warmup takes on the order of a thousand steps. Measured: a cold ViT-B
    from DINOv3 weights sat at a loss of exactly `ln 2` at step 100 and still had zero boundary
    accuracy at step 1000, against a comparable trilinear-head run already at 0.494 by step 100
    with 7x the gradient norm. On a warm encoder the same initialisation costs nothing, because the
    features are already right the moment the head opens.
    """

    def __init__(
        self,
        in_dim: int,
        patch_size: tuple[int, ...],
        out_channels: int,
        hidden: int = 256,
        readout: int = 16,
        refine_depth: int = 2,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        rank = len(patch_size)
        if rank not in CONV:
            raise ValueError(f"patch_size must have 2 or 3 entries, got {patch_size}")
        if refine_depth < 1:
            raise ValueError(
                f"refine_depth must be at least 1, got {refine_depth}: without a convolution "
                "after the expansion nothing mixes across block seams"
            )
        self.patch_size = tuple(patch_size)
        conv, conv_transpose = CONV[rank], CONV_TRANSPOSE[rank]

        self.project = conv(in_dim, hidden, kernel_size=1)
        self.expand = conv_transpose(
            hidden, readout, kernel_size=self.patch_size, stride=self.patch_size
        )
        if self.expand.kernel_size != self.expand.stride:
            # `_expand_tokens` evaluates this layer as a per-token matmul, which is only the same
            # function while stride equals kernel: that is what makes each token's output block
            # disjoint from its neighbours', so there is no overlap to add. The line above
            # satisfies it today, and this guards the edit that would not -- the class docstring
            # proposes exactly one, an overlapping `kernel_size = 2 * patch_size` to widen the
            # support across block seams. Made under that change, the matmul would quietly drop
            # the overlap and return a different function, and the equivalence test would not
            # notice because it builds the head through this same constructor.
            raise ValueError(
                f"SubPixelHead expands with kernel_size={self.expand.kernel_size} and "
                f"stride={self.expand.stride}. They must be equal: the expansion is computed as a "
                "per-token matmul (see _expand_tokens), which is only equivalent to the "
                "transposed convolution when each token's block is disjoint. An overlapping "
                "kernel needs the convolution back, and the matmul removed."
            )
        refine: list[nn.Module] = []
        for _ in range(refine_depth):
            refine += [conv(readout, readout, kernel_size=3, padding=1), nn.GELU()]
        self.refine = nn.Sequential(*refine)
        self.out = conv(readout, out_channels, kernel_size=1)

        if zero_init_output:
            nn.init.zeros_(self.out.weight)
        nn.init.zeros_(cast(torch.Tensor, self.out.bias))

    def set_output_bias(self, bias: float) -> None:
        """Start from a constant prediction of `bias` (a logit) rather than zero."""
        nn.init.constant_(cast(torch.Tensor, self.out.bias), bias)

    def _expand_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """`self.expand`'s function, computed as a matmul instead of a transposed convolution.

        Identical arithmetic, on `self.expand`'s own parameters -- the module still owns the weight
        and the bias, is still initialised by torch, and still appears under the same names in a
        checkpoint. Only the kernel that evaluates it changes.

        It has to change because cuDNN evaluates the convolution form catastrophically badly at
        this shape. Profiled on an H100 at a 256-cube with patch 16, the expansion cost 152 ms of
        a 344 ms step across two kernels (`strided_dgrad_indexed` and
        `sm80_xmma_dgrad_implicit_gemm_indexed`) -- note the `sm80`, an Ampere kernel on a Hopper
        card, cuDNN having no tuned Hopper kernel for a 3D transposed convolution at
        kernel == stride == 16 and falling back a generation. The arithmetic is 4096 tokens x 256
        channels x 16 readout x 4096 kernel elements = 68.7 GMAC, or 412 GFLOP with backward,
        which at this card's 989 TFLOP/s is 0.42 ms. Measured against that, the convolution runs
        at roughly 0.3% of peak.

        The matmul form is available only because `kernel_size == stride`, which makes the
        expansion separable per token: each input position writes one disjoint output block and
        nothing overlaps, so the whole layer is `(tokens, hidden) @ (hidden, readout * patch)`
        followed by a fold. That is a dense GEMM, which the same hardware runs near peak.

        The fold is the part the class docstring warns about, and the warning is right: a
        permutation that swaps two spatial axes trains perfectly well and segments silently wrong.
        It is not defended by care here but by `tests/unit/test_dense_heads.py`, which asserts this
        method agrees with `self.expand` -- the convolution it replaces -- elementwise. A wrong
        permutation fails that test rather than a downstream benchmark six weeks later.
        """
        batch, hidden, *grid = x.shape
        rank = len(grid)
        readout = self.expand.out_channels

        # (B, hidden, *grid) -> (B, tokens, hidden), one row per input position.
        tokens = x.flatten(2).transpose(1, 2)
        # ConvTranspose weight is (in_channels, out_channels, *kernel), so flattening everything
        # after the first axis gives (hidden, readout * prod(patch)) with the trailing axes still
        # in (out_channel, *kernel) order -- exactly what the reshape below unpacks.
        expanded = tokens @ self.expand.weight.reshape(hidden, -1)

        # (B, *grid, readout, *patch) -> (B, readout, grid_0, patch_0, grid_1, patch_1, ...), so
        # that each axis pairs its token index with its within-block offset before they are merged.
        expanded = expanded.reshape(batch, *grid, readout, *self.patch_size)
        order = [0, rank + 1]
        for axis in range(rank):
            order += [1 + axis, rank + 2 + axis]
        expanded = expanded.permute(*order).reshape(
            batch, readout, *[g * p for g, p in zip(grid, self.patch_size, strict=True)]
        )

        bias = self.expand.bias
        if bias is not None:
            expanded = expanded + bias.reshape(readout, *(1,) * rank)
        return expanded

    def forward(self, x: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
        # `size` is checked rather than interpolated to. With kernel == stride the output is
        # exactly `grid * patch_size`, and an encoder reaches its grid by floor division, so a crop
        # that is not a whole number of patches would leave a rim of voxels this head never covers.
        # Silently returning a smaller volume would surface as a shape error somewhere downstream,
        # or -- worse -- as a broadcast against a mis-sized target.
        expected = tuple(g * p for g, p in zip(x.shape[2:], self.patch_size, strict=True))
        if tuple(size) != expected:
            raise ValueError(
                f"a patch grid of {tuple(x.shape[2:])} at patch size {self.patch_size} covers "
                f"{expected}, but the crop is {tuple(size)}. A sub-pixel head can only produce a "
                "whole number of patches; use a crop divisible by the patch size."
            )
        x = self.project(x)
        x = self._expand_tokens(x)
        x = self.refine(x)
        return self.out(x)


class _ResidualBlock(nn.Module):
    """Two 3-wide convolutions, each followed by a channel LayerNorm, around a residual connection.

    MONAI's `UnetResBlock` -- the block UNETR's reference implementation builds its decoder from --
    with this repository's normalisation and activation. `ChannelLayerNorm` rather than BatchNorm
    because a dense run trains at one or a few crops per rank, where batch statistics are noise; and
    rather than InstanceNorm because those statistics are taken over the window, so a voxel's
    features would depend on which prediction window it happens to sit in, and overlapping windows
    would disagree about it. A 1x1 projection carries the residual when the width changes.
    """

    def __init__(self, rank: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv = CONV[rank]
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = ChannelLayerNorm(out_channels)
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = ChannelLayerNorm(out_channels)
        self.project: nn.Module = (
            conv(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(y + self.project(x))


class UNETRHead(nn.Module):
    """Tokens from several encoder depths plus the raw image -> `(B, out_channels, *image grid)`.

    UNETR's decoder (Hatamizadeh et al., WACV 2022). The patch grid is brought back to voxel
    resolution in `levels = log2(patch)` stages of x2 transposed convolution. At each stage the
    upsampled stream is concatenated with a skip and fused by a residual block: skip `k` (at stride
    `2^k`) is the encoder's `k`-th requested layer, brought there by its own chain of transposed
    convolutions, and the last, at full resolution, is a residual block run on the raw image. The
    deepest requested layer is the bottleneck the stream starts from.

    **Only the raw-image skip adds resolution.** The encoder skips come from intermediate depths but
    sit on the same patch grid as the final layer: they add less abstract features, not finer ones.
    Structure below the token scale can come from the image path alone, which is why
    `image_skip` is a switch -- with it off, the head is the paper's decoder minus its one
    full-resolution input.

    `widths[k]` is the channel count at stride `2^k`, finest first. The paper's 64/128/256/512 were
    sized for 96^3 crops; the default 16/32/64/128 is MONAI's and is what a 256^3 crop can afford.
    Measured on one B300 at a 256^3 crop behind a ViT-L/16 (compiled, one crop, forward + backward):
    this head costs 1.41-1.46x the sub-pixel head's step at 16.3 GiB, 64/128/256/512 costs 42 s a
    step, because its last fusion concatenates 2 x 64 channels over the crop -- exactly 2^31
    elements, cuDNN's 32-bit indexing limit. That limit is the real cap on the finest width:
    `2 * widths[0] * voxels` must stay under 2^31, so a 512^3 crop would need slabs
    (probes/unetr-feasibility on /nrs has the measurements). Upsampling by transposed convolution
    because that measured faster at kernel == stride == 2 than both a matmul-and-fold (the trick
    `SubPixelHead` needs at kernel 16) and a channels-last layout.

    `zero_init_output` has `SubPixelHead`'s meaning and its caveat: a zeroed output starts every
    prediction at the bias and sends the encoder no gradient until it grows, so turn it off when the
    encoder also has to learn.
    """

    def __init__(
        self,
        in_dim: int,
        patch_size: tuple[int, ...],
        out_channels: int,
        in_channels: int = 1,
        widths: Sequence[int] | None = None,
        image_skip: bool = True,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.levels = self.levels_for(patch_size)
        rank = len(patch_size)
        widths = tuple(widths) if widths is not None else tuple(16 * 2**k for k in range(self.levels))
        if len(widths) != self.levels or any(w < 1 for w in widths):
            raise ValueError(
                f"a patch of {tuple(patch_size)} takes {self.levels} x2 stages, so widths needs "
                f"{self.levels} positive entries (finest first), got {widths}"
            )
        self.patch_size = tuple(patch_size)
        self.widths = widths
        conv, up = CONV[rank], CONV_TRANSPOSE[rank]

        # Skip k = 1 .. levels-1: requested layer k-1, at stride 2^k with widths[k] channels.
        self.skips = nn.ModuleList()
        for k in range(1, self.levels):
            chain: list[nn.Module] = [up(in_dim, widths[k], kernel_size=2, stride=2)]
            for _ in range(self.levels - k - 1):
                chain += [up(widths[k], widths[k], kernel_size=2, stride=2),
                          _ResidualBlock(rank, widths[k], widths[k])]
            self.skips.append(nn.Sequential(*chain))

        # The stream: bottleneck -> stride 2^(levels-1) -> ... -> stride 1, fusing a skip at each.
        self.ups = nn.ModuleList()
        self.fuse = nn.ModuleList()
        channels = in_dim
        for k in reversed(range(self.levels)):
            self.ups.append(up(channels, widths[k], kernel_size=2, stride=2))
            joined = widths[k] if (k == 0 and not image_skip) else 2 * widths[k]
            self.fuse.append(_ResidualBlock(rank, joined, widths[k]))
            channels = widths[k]
        self.image: nn.Module | None = (
            _ResidualBlock(rank, in_channels, widths[0]) if image_skip else None
        )
        self.out = conv(widths[0], out_channels, kernel_size=1)

        if zero_init_output:
            nn.init.zeros_(self.out.weight)
        nn.init.zeros_(cast(torch.Tensor, self.out.bias))

    @staticmethod
    def levels_for(patch_size: tuple[int, ...]) -> int:
        """x2 stages from the patch grid to voxels: `log2(patch)`, for a cubic power-of-two patch.

        Anisotropic patches would need per-axis stage factors, and a patch that is not a power of two
        a non-x2 stage somewhere; neither is built, so both are refused rather than approximated.
        """
        if len(patch_size) not in CONV:
            raise ValueError(f"patch_size must have 2 or 3 entries, got {tuple(patch_size)}")
        side = patch_size[0]
        if any(p != side for p in patch_size) or side < 2 or side & (side - 1):
            raise ValueError(
                f"UNETRHead needs the same power-of-two patch on every axis (x2 stages up to "
                f"voxels), got {tuple(patch_size)}"
            )
        return side.bit_length() - 1

    def forward(self, features: Sequence[torch.Tensor], image: torch.Tensor) -> torch.Tensor:
        """`levels` token grids `(B, in_dim, *grid)`, shallowest first, and `(B, C, *grid*patch)`."""
        if len(features) != self.levels:
            raise ValueError(
                f"a patch of {self.patch_size} fuses {self.levels} encoder layers (the deepest last), "
                f"got {len(features)}"
            )
        grid = tuple(features[-1].shape[2:])
        expected = tuple(g * p for g, p in zip(grid, self.patch_size, strict=True))
        if tuple(image.shape[2:]) != expected:
            # The same contract as `SubPixelHead`: x2 stages reach exactly grid * patch, so an image
            # that is not a whole number of patches would leave a rim no stage produces.
            raise ValueError(
                f"a patch grid of {grid} at patch size {self.patch_size} covers {expected}, but the "
                f"image is {tuple(image.shape[2:])}; use a crop divisible by the patch size"
            )
        skips = [branch(f) for branch, f in zip(self.skips, features[:-1], strict=True)]
        x = features[-1]
        for index, k in enumerate(reversed(range(self.levels))):
            x = self.ups[index](x)
            if k >= 1:
                x = torch.cat([x, skips[k - 1]], dim=1)
            elif self.image is not None:
                x = torch.cat([x, self.image(image)], dim=1)
            x = self.fuse[index](x)
        return self.out(x)
