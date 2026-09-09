"""Axial rotary position embedding over continuous coordinates.

Rotary embeddings (RoPE) make an attention logit depend on the *difference* between two tokens'
positions rather than their absolute values. Written out here rather than taken from a library
because the interesting property for multi-resolution models is that positions need not be integer
sequence indices: feed physical coordinates and two patches describing the same place get the same
rotation, whatever resolution they came from.

"Axial" means each spatial axis owns a slice of the head dimension and rotates it by its own
coordinate, so the axes stay separable. The alternative -- concatenating every axis's angles and
rotating the head dimension as one block -- mixes axes at the halfway split and is not what
axis-wise RoPE means.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

NORMALIZE_MODES = ("separate", "max", "min")


def _unit_extent(extent: Sequence[int], mode: str) -> torch.Tensor:
    """The denominator each axis is divided by, as a (rank,) tensor.

    `"separate"` divides each axis by its own extent, so a non-cubic crop is stretched to a cube in
    coordinate space. `"max"` and `"min"` divide every axis by one shared extent, which preserves
    the crop's aspect ratio and leaves the short axes covering less than the full [-1, 1] range.
    DINOv3's own default is `"separate"`, and the choice must match whatever the backbone was
    trained with -- a decoder positioned under one convention and an encoder under the other
    disagree about where a voxel is, by an amount that grows with how anisotropic the crop is.
    """
    if mode not in NORMALIZE_MODES:
        raise ValueError(f"normalize mode must be one of {NORMALIZE_MODES}, got {mode!r}")
    sizes = torch.tensor(list(extent), dtype=torch.float32)
    if mode == "separate":
        return sizes
    return sizes.new_full(sizes.shape, float(sizes.max() if mode == "max" else sizes.min()))


def voxel_coords(
    voxels: torch.Tensor, extent: Sequence[int], mode: str = "separate"
) -> torch.Tensor:
    """Voxel indices -> coordinates in [-1, 1], with the half-voxel offset applied.

    `voxels` is `(..., rank)`, integral or fractional; the result has the same shape. Voxel `i` on
    an axis of extent `E` lands at `2 * (i + 0.5) / E - 1`, so index 0 sits just inside the near
    face and `E - 1` just inside the far one.

    This exists as a named function, beside `patch_grid_coords`, because the two have to agree
    exactly: a promptable segmenter's whole positional story is that a point prompt and the patch
    token containing it rotate identically. Two callers each writing out `2 * x / E - 1` would
    agree until one of them dropped the half-voxel, and the symptom -- prompts landing half a patch
    off, worse at coarse patch sizes -- is a quality regression with no shape to catch it.
    """
    denominator = _unit_extent(extent, mode).to(device=voxels.device, dtype=torch.float32)
    return 2.0 * (voxels.float() + 0.5) / denominator - 1.0


def patch_grid_coords(
    grid: Sequence[int],
    patch_size: Sequence[int],
    extent: Sequence[int],
    mode: str = "separate",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Patch-token centres as `(prod(grid), rank)` coordinates in [-1, 1], row-major.

    Row-major because that is the order every patch embedding in this repo emits tokens in, and
    what `BaseModel.patch_features` promises alongside the grid.

    The centre of patch `j` is voxel `(j + 0.5) * patch`, which `voxel_coords` would offset by
    another half voxel -- so the half is subtracted back out here rather than calling it. `extent`
    is the crop's true voxel extent, not `grid * patch`: an encoder reaches its grid by floor
    division, so a crop that is not a whole number of patches has a rim, and normalising by
    `grid * patch` would place every token slightly outside the frame the prompts use.
    """
    axes = [
        (torch.arange(count, dtype=torch.float32, device=device) + 0.5) * float(step) - 0.5
        for count, step in zip(grid, patch_size, strict=True)
    ]
    centres = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, -2)
    return voxel_coords(centres, extent, mode)


def split_rope_dims(head_dim: int, spatial_rank: int) -> tuple[int, ...]:
    """Divide a head's channels among spatial axes, each an even count.

    Even because rotation acts on pairs of channels. The remainder goes to the first axis instead
    of being dropped, so the whole head dimension is used when it divides unevenly.
    """
    if head_dim < 2 * spatial_rank:
        raise ValueError(
            f"head_dim {head_dim} cannot cover {spatial_rank} spatial axes: each axis needs at "
            f"least one channel pair, so head_dim must be at least {2 * spatial_rank}. Raise "
            "embed_dim or lower num_heads."
        )
    per_axis = 2 * (head_dim // spatial_rank // 2)
    dims = [per_axis] * spatial_rank
    dims[0] += 2 * ((head_dim - sum(dims)) // 2)
    return tuple(dims)


@dataclass(frozen=True)
class RotaryTables:
    """Precomputed cos/sin per axis, ready to rotate a (B, N, heads, head_dim) tensor.

    A callable rather than a method on the embedding module so attention can apply rotation
    without knowing anything about how the angles were produced -- the attention layer stays a
    generic kernel with a swappable position encoding.
    """

    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    axis_dims: tuple[int, ...]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate (B, N, heads, head_dim) in place of its position-free self."""
        rotated = []
        start = 0
        for (cos, sin), width in zip(self.tables, self.axis_dims, strict=True):
            chunk = x[..., start : start + width]
            first, second = chunk.chunk(2, dim=-1)
            # The standard 2D rotation, applied to (first, second) as one complex pair per
            # frequency. cos/sin arrive as (B, N, 1, width/2) and broadcast over heads.
            rotated.append(
                torch.cat(
                    (first * cos - second * sin, second * cos + first * sin), dim=-1
                )
            )
            start += width

        if start < x.shape[-1]:
            # Channels past the axis allocations carry no position, which is the usual RoPE
            # arrangement when head_dim does not divide evenly.
            rotated.append(x[..., start:])
        # Angles are computed in fp32 for precision; casting back keeps a bf16 forward in bf16,
        # which also matters because a silently promoted dtype would disable the FA4 kernel.
        return torch.cat(rotated, dim=-1).to(x.dtype)


class AxialRotaryEmbedding(nn.Module):
    """Turns per-token coordinates into rotation tables.

    Frequencies are learnable and initialised to a geometric progression: every layer owns its own
    copy, so a layer can widen or narrow the range of distances its attention is sensitive to
    instead of inheriting one fixed schedule. That is MuViT's design.

    **Two ways to set the initial schedule, and which one is right depends on what a coordinate
    means.** `base` is the usual transformer form, `theta_k = coordinate / base^(2k/d)`, and it
    assumes coordinates are integer sequence indices -- the natural reading when a coordinate is a
    patch index. `min_period`/`max_period` instead says directly which wavelengths the rotation
    should span, `theta_k = 2*pi * coordinate / period_k` with the periods geometric between the
    two. That is the parametrisation DINOv3 uses, and it is the one that makes sense when
    coordinates are *normalised* to a fixed range such as [-1, 1]: there, `base` would put every
    angle in the first fraction of a turn and the whole schedule would collapse onto one
    slowly-varying frequency.

    Exactly one of the two must be given, since they are alternative descriptions of the same
    tensor and a caller supplying both has an expectation that cannot be met.
    """

    def __init__(
        self,
        head_dim: int,
        spatial_rank: int,
        base: float | None = 10000.0,
        min_period: float | None = None,
        max_period: float | None = None,
    ) -> None:
        super().__init__()
        both_periods = min_period is not None and max_period is not None
        if (base is None) == (not both_periods):
            raise ValueError(
                "set exactly one of `base` or `min_period`+`max_period`; got "
                f"base={base}, min_period={min_period}, max_period={max_period}"
            )
        if base is not None and base <= 1.0:
            raise ValueError(f"rotary base must be greater than 1, got {base}")
        if both_periods:
            assert min_period is not None and max_period is not None  # narrowed for the checker
            if not 0 < min_period < max_period:
                raise ValueError(
                    f"need 0 < min_period < max_period, got {min_period} and {max_period}"
                )

        self.axis_dims = split_rope_dims(head_dim, spatial_rank)
        self.spatial_rank = spatial_rank
        # Stored as the reciprocal of the period (times 2*pi) so the forward pass is a multiply.
        # One Parameter per axis, since axes may have different widths.
        self.inv_freqs = nn.ParameterList(
            nn.Parameter(self._initial_inv_freq(width, base, min_period, max_period))
            for width in self.axis_dims
        )

    @staticmethod
    def _initial_inv_freq(
        width: int, base: float | None, min_period: float | None, max_period: float | None
    ) -> torch.Tensor:
        """`width // 2` reciprocal wavelengths, geometrically spaced, for one axis."""
        if base is not None:
            return 1.0 / (base ** (torch.arange(0, width, 2).float() / width))
        assert min_period is not None and max_period is not None
        # Geometric from min_period at index 0 to max_period at the last index, matching
        # `layers.dinov3.rope`, so a decoder positioned this way rotates under the same law as a
        # DINOv3 backbone rather than merely a similar-looking one.
        exponents = torch.linspace(0.0, 1.0, width // 2)
        periods = min_period * (max_period / min_period) ** exponents
        return 2.0 * math.pi / periods

    @property
    def rotary_dim(self) -> int:
        """How many of a head's channels carry position."""
        return sum(self.axis_dims)

    def forward(self, coords: torch.Tensor) -> RotaryTables:
        """(B, N, spatial_rank) coordinates -> tables for rotating queries and keys."""
        if coords.shape[-1] != self.spatial_rank:
            raise ValueError(
                f"expected coordinates with {self.spatial_rank} components on the last axis, got "
                f"{tuple(coords.shape)}"
            )

        tables = []
        for axis, inv_freq in enumerate(self.inv_freqs):
            # fp32 regardless of autocast: the angle is a product of a possibly large coordinate
            # and a small frequency, and half precision loses the distinctions that make nearby
            # positions different.
            angles = coords[..., axis].float().unsqueeze(-1) * inv_freq.float()
            angles = angles.unsqueeze(-2)  # (B, N, 1, width/2), broadcasting over heads
            tables.append((angles.cos(), angles.sin()))
        return RotaryTables(tuple(tables), self.axis_dims)
