"""Building DINO's multi-crop views, and the iBOT masks, on device.

DINO's objective only means something if the two views a sample produces genuinely differ: the
student sees one crop and must predict what the teacher saw in *another*. Feed it two identical
tensors and the loss is minimized by the constant function.

The reference builds its views in a torchvision transform on dataloader workers. That is the
wrong shape here for two reasons. This repo's dataset yields one volumetric crop per sample and
its loader takes no collate hook, so the views have nowhere to be assembled; and augmenting a
256-cubed volume eight times per sample on a CPU worker would starve the GPU. So views are derived
on device from the one crop the dataset provides, which also matches how the other strategies
here work -- masked autoencoding masks inside the algorithm, affinity segmentation builds its
targets there.

Geometry (crop, resize, flip, rotate) goes through a single `grid_sample` per view: one affine
matrix per sample expresses all of it at once, so ten views cost ten batched resamples rather
than a Python loop over samples. Rank 2 and rank 3 differ only in the size of that matrix, which
is what lets one implementation serve both the 2D and the 3D encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AugmentationConfig:
    """Everything stochastic about a view, in one place so a config file can state it.

    The intensity settings replace 2D DINO's colour pipeline. Hue, saturation and grayscale are
    meaningless on a single-channel volume, and solarize assumes an 8-bit display convention, so
    what survives is the part that actually applies: brightness, contrast, blur, and noise.
    """

    brightness: float = 0.4
    contrast: float = 0.4
    intensity_prob: float = 0.8
    blur_prob: float = 0.5
    blur_sigma: tuple[float, float] = (0.1, 2.0)
    noise_std: float = 0.05
    noise_prob: float = 0.5
    flip_prob: float = 0.5
    rotate_prob: float = 0.5


def random_resized_crop(
    volumes: torch.Tensor,
    out_size: tuple[int, ...],
    scale: tuple[float, float],
    *,
    flip_prob: float = 0.5,
    rotate_prob: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(B, C, *in_size) -> (B, C, *out_size), each sample cropped and resampled independently.

    `scale` bounds the fraction of the input's *volume* a crop covers, as in the 2D original, so
    the same numbers mean the same thing at either rank -- a side length scales as the rank-th
    root. Sampling is uniform in that fraction, then jittered per axis so crops are not all cubes.

    Flips are applied on every axis independently. Rotation is by a quarter turn in the first two
    axes only: for the volumes this is built for those are the in-plane axes, which share a voxel
    size, while the third is coarser -- rotating into it would resample across a different
    physical spacing and present the model with geometry that does not occur in the data.
    """
    batch, channels = volumes.shape[0], volumes.shape[1]
    rank = volumes.ndim - 2
    device = volumes.device
    draw = lambda *shape: torch.rand(*shape, device=device, generator=generator)  # noqa: E731

    # A crop covering fraction `f` of the volume has sides f**(1/rank) on average; the jitter
    # makes them unequal while keeping the product close to f.
    fraction = draw(batch) * (scale[1] - scale[0]) + scale[0]
    side = fraction.pow(1.0 / rank).unsqueeze(1).expand(batch, rank)
    jitter = torch.exp((draw(batch, rank) - 0.5) * 0.4)
    side = (side * jitter).clamp(0.05, 1.0)

    # grid_sample works in [-1, 1], so a crop is a scaling by `side` about a centre that must stay
    # far enough inside the volume for the crop to fit.
    centre = (draw(batch, rank) * 2 - 1) * (1 - side)

    if flip_prob > 0:
        flips = torch.where(draw(batch, rank) < flip_prob, -1.0, 1.0)
        side = side * flips

    # theta maps output coordinates to input ones: a diagonal scale plus a translation.
    theta = torch.zeros(batch, rank, rank + 1, device=device, dtype=torch.float32)
    theta[:, :, rank] = centre
    diagonal = torch.diag_embed(side)

    if rotate_prob > 0 and rank >= 2:
        # A quarter turn in the first two axes, as a permutation with a sign flip.
        turn = draw(batch) < rotate_prob
        rotation = torch.eye(rank, device=device).expand(batch, rank, rank).clone()
        rotation[turn, 0, 0] = 0.0
        rotation[turn, 1, 1] = 0.0
        rotation[turn, 0, 1] = -1.0
        rotation[turn, 1, 0] = 1.0
        diagonal = torch.bmm(rotation, diagonal)
    theta[:, :, :rank] = diagonal

    # affine_grid's size argument carries the *output* extent, which is how the resize happens.
    grid = F.affine_grid(
        theta, [batch, channels, *out_size], align_corners=False
    )
    mode = "bilinear"  # torch's name for linear interpolation at either rank
    return F.grid_sample(
        volumes.float(), grid, mode=mode, padding_mode="reflection", align_corners=False
    )


def _gaussian_blur(volumes: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur, one sigma per sample -> same shape.

    Separable because an isotropic Gaussian factorises: `rank` one-dimensional convolutions cost
    `rank * k` multiplies per voxel where the direct form costs `k ** rank`, which at rank 3 is
    the difference between usable and not.
    """
    rank = volumes.ndim - 2
    batch, channels = volumes.shape[0], volumes.shape[1]
    radius = 4
    offsets = torch.arange(-radius, radius + 1, device=volumes.device, dtype=torch.float32)

    # (B, k), normalised per sample
    kernel = torch.exp(-(offsets[None, :] ** 2) / (2 * sigma[:, None] ** 2))
    kernel = kernel / kernel.sum(dim=1, keepdim=True)

    conv = (F.conv1d, F.conv2d, F.conv3d)[rank - 1]
    out = volumes
    for axis in range(rank):
        shape = [batch, 1] + [1] * rank
        shape[2 + axis] = 2 * radius + 1
        # One group per (sample, channel) so each sample keeps its own sigma in a single call.
        weight = kernel.reshape(shape).repeat_interleave(channels, dim=0)
        merged = out.reshape(1, batch * channels, *out.shape[2:])
        # Reflect rather than zero at the border: zero padding pulls edge voxels toward the
        # background and darkens the rim of every view, which is a systematic artefact the model
        # would happily learn to key on. F.pad takes its axes last-first.
        pad = [0] * (2 * rank)
        pad[2 * (rank - 1 - axis)] = radius
        pad[2 * (rank - 1 - axis) + 1] = radius
        merged = F.pad(merged, pad, mode="reflect")
        out = conv(merged, weight, groups=batch * channels)
        out = out.reshape(batch, channels, *out.shape[2:])
    return out


def photometric(
    views: torch.Tensor,
    config: AugmentationConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Brightness, contrast, blur and noise, each applied to a random subset of the batch."""
    batch = views.shape[0]
    device = views.device
    rank = views.ndim - 2
    per_sample = (batch,) + (1,) * (rank + 1)
    draw = lambda *shape: torch.rand(*shape, device=device, generator=generator)  # noqa: E731

    apply_intensity = (draw(batch) < config.intensity_prob).view(per_sample)
    brightness = 1 + (draw(batch).view(per_sample) * 2 - 1) * config.brightness
    contrast = 1 + (draw(batch).view(per_sample) * 2 - 1) * config.contrast
    mean = views.mean(dim=tuple(range(1, views.ndim)), keepdim=True)
    adjusted = (views - mean) * contrast + mean * brightness
    views = torch.where(apply_intensity, adjusted, views)

    if config.blur_prob > 0:
        low, high = config.blur_sigma
        sigma = draw(batch) * (high - low) + low
        blurred = _gaussian_blur(views, sigma)
        views = torch.where((draw(batch) < config.blur_prob).view(per_sample), blurred, views)

    if config.noise_prob > 0 and config.noise_std > 0:
        noise = torch.randn(views.shape, device=device, generator=generator) * config.noise_std
        add_noise = (draw(batch) < config.noise_prob).view(per_sample)
        views = torch.where(add_noise, views + noise, views)

    return views


def block_mask(
    grid: tuple[int, ...],
    batch: int,
    ratio: tuple[float, float],
    *,
    sample_probability: float = 0.5,
    min_patches: int = 4,
    max_aspect: float = 1 / 0.3,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """iBOT masks over a patch grid -> (batch, prod(grid)) bool.

    Blocks rather than scattered patches, following BEiT and the reference. A scattered mask can
    be filled in by copying immediate neighbours, so it never forces the model to represent
    anything at range; a contiguous block removes that shortcut.

    Only `sample_probability` of the batch is masked at all -- the rest pass through whole, which
    keeps the unmasked patch statistics in the teacher's diet.
    """
    total = math.prod(grid)
    rank = len(grid)
    masks = torch.zeros(batch, *grid, dtype=torch.bool, device=device)
    n_masked = int(batch * sample_probability)
    if n_masked == 0:
        return masks.reshape(batch, total)

    # A spread of ratios rather than one, so the model sees both light and heavy masking.
    targets = torch.linspace(ratio[0], ratio[1], n_masked)
    for sample in range(n_masked):
        wanted = int(targets[sample].item() * total)
        filled = 0
        for _ in range(64):  # bounded: a block may fail to place among what is already masked
            if filled >= wanted:
                break
            remaining = wanted - filled
            span = max(min_patches + 1, remaining)
            area = float(
                torch.empty(1).uniform_(min_patches, span, generator=generator).item()
            )
            aspect = math.exp(
                (torch.rand(1, generator=generator).item() - 0.5) * 2 * math.log(max_aspect)
            )
            side = max(1, int(round((area * aspect) ** (1.0 / rank))))
            extents = [min(side, grid[axis]) for axis in range(rank)]
            corner = [
                int(torch.randint(0, grid[axis] - extents[axis] + 1, (1,), generator=generator))
                for axis in range(rank)
            ]
            window: tuple[int | slice, ...] = (sample, *(
                slice(corner[axis], corner[axis] + extents[axis]) for axis in range(rank)
            ))
            fresh = int((~masks[window]).sum().item())
            # Only place a block that fits the remaining budget, so the requested ratio is an
            # upper bound rather than a suggestion. Overshooting would mask more than the caller
            # asked for and, at the top of the range, leave the student almost nothing to see.
            if fresh == 0 or fresh > remaining:
                continue
            masks[window] = True
            filled += fresh
    return masks.reshape(batch, total)
