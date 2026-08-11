"""Getting from a ViT's patch grid back to one prediction per voxel.

Any dense task on a transformer has to undo the patch embedding: tokens sit on a grid `patch_size`
times coarser than the input on every axis, and the head has to produce a value per voxel. The two
heads here are alternative answers, and they share a signature -- `forward(x, size)` -- so an
algorithm can offer the choice as configuration without branching at the call site.

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

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:  # type: ignore[override]
        # Left to autocast, which runs this in fp32. That is the most expensive tensor in a dense
        # algorithm -- upsampling multiplies it by the cube of the patch size, so fp32 costs 32 GiB
        # rather than 16 at a 512-cube -- and forcing it to bf16 was measured and rejected:
        # accumulating eight neighbours in bf16 is 1.4x less accurate than accumulating in fp32 and
        # rounding once, with worst-case deviations of several percent of the feature scale. Memory
        # is bought with a bigger GPU, not with the one number the head is built to produce.
        x = F.interpolate(x, size=size, mode=self.mode, align_corners=False)
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

    The last convolution is zero-initialised, so at construction the head emits its bias
    everywhere. Fed a bias of `logit(positive rate)` that is the trivial constant predictor, which
    is a deliberate starting point when the encoder is warm: a randomly-initialised dense head
    would otherwise push meaningless gradients into weights that already solve the task. A
    consequence worth knowing: with that weight at zero the gradient reaching everything before it
    is also zero, so on the very first step only `out` learns. It unsticks itself immediately --
    `out.weight` does receive gradient, and once it is non-zero the rest of the head follows -- but
    under a long warmup the head stays quiet for longer than the step count alone suggests.
    """

    def __init__(
        self,
        in_dim: int,
        patch_size: tuple[int, ...],
        out_channels: int,
        hidden: int = 256,
        readout: int = 16,
        refine_depth: int = 2,
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
        refine: list[nn.Module] = []
        for _ in range(refine_depth):
            refine += [conv(readout, readout, kernel_size=3, padding=1), nn.GELU()]
        self.refine = nn.Sequential(*refine)
        self.out = conv(readout, out_channels, kernel_size=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(cast(torch.Tensor, self.out.bias))

    def set_output_bias(self, bias: float) -> None:
        """Start from a constant prediction of `bias` (a logit) rather than zero."""
        nn.init.constant_(cast(torch.Tensor, self.out.bias), bias)

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
        x = self.expand(x)
        x = self.refine(x)
        return self.out(x)
