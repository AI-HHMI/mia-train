"""DINOv3 axial rotary position embeddings, in 2D and 3D.

Ported from the DINOv3 reference implementation. These produce the (sin, cos) tables that
`layers.dinov3.attention.SelfAttention` applies to queries and keys; no coordinate mixing across
axes and no learnable weights (except the superposition variant's single depth gate).

Distinct from `layers.common.rope.AxialRotaryEmbedding`, which the from-scratch ViT3D/MuViT3D use.
That one takes explicit per-token coordinates and learns its frequencies; these derive coordinates
from the patch grid itself, normalise them to [-1, +1], and add DINOv3's train-time coordinate
augmentations (shift/jitter/rescale), which is what makes DINOv3 robust to changes in resolution
and aspect ratio.

Frequencies are parametrised either by `base` or by `min_period` + `max_period`, never both.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RopePositionEmbedding(nn.Module):
    """2D axial RoPE over a H x W patch grid."""

    # Declared so type checking sees a Tensor rather than nn.Module's generic attribute union.
    periods: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because DINOv3 initialises the teacher with
        # teacher.load_state_dict(student.state_dict()).
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods", torch.empty(D_head // 4, device=device, dtype=dtype), persistent=True
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd: dict[str, Any] = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in
        # [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in
        # [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            # __init__ rejects the case where neither parametrisation is given, so reaching this
            # branch means both periods are set.
            assert self.min_period is not None and self.max_period is not None
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.D_head // 4, device=device, dtype=dtype
            )  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods


class RopePositionEmbedding3D(nn.Module):
    """3D axial RoPE over a D x H x W patch grid, for volumetric data such as EM.

    Each of the three axes owns a third of the rotary channels. When the head dimension is not
    divisible by 6 the leftover channels carry no position and are passed through unchanged.
    """

    periods: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """
        Args:
            embed_dim: The embedding dimension.
            num_heads: The number of attention heads.
            base: The base for the geometric progression of periods.
            min_period: The minimum period for the sinusoidal embeddings.
            max_period: The maximum period for the sinusoidal embeddings.
            normalize_coords: How to normalize coordinates -- "max" by max(D, H, W), "min" by
                min(D, H, W), or "separate" per axis.
            shift_coords: If not None, shift coordinates by a random value in [-shift, shift]
                during training.
            jitter_coords: If not None, jitter coordinates by a log-uniform value during training.
            rescale_coords: If not None, rescale coordinates by a log-uniform value during training.
            dtype: The data type for tensors.
            device: The device to place tensors on.
        """
        super().__init__()
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        if D_head % 2 != 0:
            # `rope_apply` rotates the head by pairing channel i with channel i + D_head/2, which
            # an odd head dimension cannot do. The 2D embeddings rule this out via their
            # `embed_dim % (4 * num_heads) == 0` assertion; this one accepts more shapes, so it
            # has to say so itself.
            raise ValueError(
                f"embed_dim {embed_dim} / num_heads {num_heads} gives an odd head dimension "
                f"{D_head}; rotary embedding rotates channel pairs, so it must be even."
            )
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        # For 3D axial RoPE the rotary channels must split three ways, one axis each. If D_head
        # isn't divisible by 6 we rotate the largest multiple of 6 and leave the rest untouched.
        self.D_rope = (D_head // 6) * 6

        # The number of periods is D_rope // 6: three axes, each needing a sin/cos pair.
        self.register_buffer(
            "periods", torch.empty(self.D_rope // 6, device=device, dtype=dtype), persistent=True
        )
        self._init_weights()

    def forward(self, *, D: int, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the (sin, cos) tables -> 2 * [D*H*W, embed_dim // num_heads]."""
        device = self.periods.device
        dtype = self.dtype
        dd: dict[str, Any] = {"device": device, "dtype": dtype}

        # Prepare coords in range [0, 1] before shifting to [-1, +1]
        if self.normalize_coords == "max":
            max_DHW = max(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / max_DHW
            coords_h = torch.arange(0.5, H, **dd) / max_DHW
            coords_w = torch.arange(0.5, W, **dd) / max_DHW
        elif self.normalize_coords == "min":
            min_DHW = min(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / min_DHW
            coords_h = torch.arange(0.5, H, **dd) / min_DHW
            coords_w = torch.arange(0.5, W, **dd) / min_DHW
        elif self.normalize_coords == "separate":
            coords_d = torch.arange(0.5, D, **dd) / D
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        # Create a 3D grid of coordinates
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"), dim=-1
        )  # [D, H, W, 3]
        coords = coords.flatten(0, 2)  # [DHW, 3]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # --- Coordinate augmentations (during training) ---
        if self.training and self.shift_coords is not None:
            shift_dhw = torch.empty(3, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_dhw[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_dhw = torch.empty(3, **dd).uniform_(-jitter_max, jitter_max).exp()
            coords *= jitter_dhw[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale = torch.empty(1, **dd).uniform_(-rescale_max, rescale_max).exp()
            coords *= rescale

        # Prepare angles and sin/cos.
        # coords is [DHW, 3], periods is [D_rope//6] -> angles [DHW, 3, D_rope//6]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        # Combine the per-axis position info -> [DHW, 3 * D_rope//6] = [DHW, D_rope//2]
        angles = angles.flatten(1, 2)

        # Pad the *half*-angle block up to D_head//2 before tiling, when D_head is not divisible
        # by 6. Both halves of the tile must line up with the halves that `rope_apply`'s
        # rotate-half operates on, and a zero angle gives cos=1, sin=0 -- so the leftover channels
        # are genuinely passed through. (Padding sin/cos with zeros *after* tiling, as the
        # upstream fork does, zeroes those channels instead of preserving them, and shifts the
        # second copy of the angles out of alignment with the rotate-half split, which destroys
        # RoPE's defining property that a logit depends only on the displacement between two
        # positions. `tests/unit/test_dinov3_rope.py` pins the property this ordering restores.)
        if self.D_rope < self.D_head:
            angles = F.pad(angles, (0, (self.D_head - self.D_rope) // 2))  # [DHW, D_head//2]

        # Tile for the sin/cos pairs -> [DHW, D_head]
        angles = angles.tile(2)
        cos = torch.cos(angles)  # [DHW, D_head]
        sin = torch.sin(angles)  # [DHW, D_head]

        return (sin, cos)

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            # The denominator is D_rope // 3, which is 2 * (D_rope // 6)
            periods = self.base ** (
                2 * torch.arange(self.D_rope // 6, device=device, dtype=dtype) / (self.D_rope // 3)
            )
        else:
            assert self.min_period is not None and self.max_period is not None
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_rope // 6, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods


class RopePositionEmbedding3DSuperposition(nn.Module):
    """3D RoPE that adds a gated depth angle on top of the 2D spatial layout.

    Keeps the exact channel split, buffer shape and frequency schedule of the 2D embedding, so a
    pretrained 2D DINOv3 checkpoint's `periods` transfer unchanged. Depth is injected by adding
    its angle to the existing H/W angles, scaled by `depth_scale`, which initialises at zero --
    so the module starts out numerically identical to the 2D case and learns how much depth to
    mix in.
    """

    periods: torch.Tensor
    depth_periods: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        depth_base: float | None = 10000.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        # Matches the 2D implementation's dimension check exactly.
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.depth_base = depth_base if depth_base is not None else base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        # Retain the exact 2D buffer footprint (D_head // 4)
        self.register_buffer(
            "periods", torch.empty(D_head // 4, device=device, dtype=dtype), persistent=True
        )
        self.register_buffer(
            "depth_periods", torch.empty(D_head // 4, device=device, dtype=dtype), persistent=True
        )

        # Zero-initialised learnable gate, so the 3D injection starts off disabled.
        self.depth_scale = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))

        self._init_weights()

    def forward(self, *, D: int, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the (sin, cos) tables -> 2 * [D*H*W, embed_dim // num_heads]."""
        device = self.periods.device
        dtype = self.dtype
        dd: dict[str, Any] = {"device": device, "dtype": dtype}

        # Prepare coords in range [0, 1] before shifting to [-1, +1]
        if self.normalize_coords == "max":
            max_DHW = max(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / max_DHW
            coords_h = torch.arange(0.5, H, **dd) / max_DHW
            coords_w = torch.arange(0.5, W, **dd) / max_DHW
        elif self.normalize_coords == "min":
            min_DHW = min(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / min_DHW
            coords_h = torch.arange(0.5, H, **dd) / min_DHW
            coords_w = torch.arange(0.5, W, **dd) / min_DHW
        elif self.normalize_coords == "separate":
            coords_d = torch.arange(0.5, D, **dd) / D
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        # Create 3D meshgrids for each axis independently
        grid_d, grid_h, grid_w = torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")

        # Separate spatial and depth paths to preserve the original 2D math mapping
        coords_spatial = torch.stack([grid_h, grid_w], dim=-1).flatten(0, 2)  # [DHW, 2]
        coords_spatial = 2.0 * coords_spatial - 1.0

        coords_depth = grid_d.flatten(0, 2)  # [DHW]
        coords_depth = 2.0 * coords_depth - 1.0

        # --- Coordinate augmentations (during training) ---
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords_spatial += shift_hw[None, :]

            # No `[None]` here: `coords_depth` is 1-D [DHW], so a [1, 1] operand would broadcast
            # the in-place add to [1, DHW] and fail. `shift_d` is already [1]. (Upstream has the
            # `[None]` and raises on the first training step with this rope type.)
            shift_d = torch.empty(1, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords_depth += shift_d

        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)

            jitter_hw = torch.empty(2, **dd).uniform_(-jitter_max, jitter_max).exp()
            coords_spatial *= jitter_hw[None, :]

            jitter_d = torch.empty(1, **dd).uniform_(-jitter_max, jitter_max).exp()
            coords_depth *= jitter_d

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale = torch.empty(1, **dd).uniform_(-rescale_max, rescale_max).exp()
            coords_spatial *= rescale
            coords_depth *= rescale

        # Baseline 2D spatial angles, matching the original layout: [DHW, 2, D_head//4]
        angles_spatial = 2 * math.pi * coords_spatial[:, :, None] / self.periods[None, None, :]
        angles_spatial = angles_spatial.flatten(1, 2)  # [DHW, D_head//2]

        # Depth angles: [DHW, D_head//4]
        angles_depth = 2 * math.pi * coords_depth[:, None] / self.depth_periods[None, :]

        # Duplicate the depth angles so they overlay the H and W channels evenly: [DHW, D_head//2]
        angles_depth_expanded = angles_depth.repeat(1, 2)

        # Superposition: add gated depth information to the spatial angles
        angles = angles_spatial + self.depth_scale * angles_depth_expanded  # [DHW, D_head//2]

        # Tile identically to the 2D version to match the head dimension: [DHW, D_head]
        angles = angles.tile(2)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return (sin, cos)

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            # `depth_base` falls back to `base` in __init__, so it is set whenever `base` is.
            assert self.depth_base is not None
            # Replicate the exact 2D frequency distribution calculation
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )
            depth_periods = self.depth_base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )
        else:
            assert self.min_period is not None and self.max_period is not None
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
            depth_periods = periods.clone()

        self.periods.data = periods
        self.depth_periods.data = depth_periods
