"""Convolutional patch embedding for DINOv3, in 2D and 3D.

Ported from the DINOv3 reference implementation. A strided convolution whose kernel equals its
stride is exactly a per-patch linear projection, so this is the standard ViT patchifier with the
custom fan-in-scaled `reset_parameters` DINOv3 uses in place of PyTorch's conv default.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn


def make_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


def make_3tuple(x: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(x, tuple):
        assert len(x) == 3, "Input tuple must have length 3"
        return x

    assert isinstance(x, int), "Input must be an integer or a tuple of 3 integers"
    return (x, x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, C, H, W) -> (B, N, D), or (B, H, W, D) unflattened.

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
        flatten_embedding: Return (B, N, D) rather than (B, H, W, D).
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[..., nn.Module] | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The grid is read back off the convolution output rather than assumed from img_size, so
        # a resolution other than the configured one still works -- which is what lets DINOv3
        # train on multiple crop sizes with one patch embedding.
        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self) -> None:
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class PatchEmbed3D(nn.Module):
    """3D volume to patch embedding: (B, C, D, H, W) -> (B, N, E), or (B, D, H, W, E) unflattened.

    Args:
        img_size: Volume size (D, H, W).
        patch_size: Patch token size (P_d, P_h, P_w).
        in_chans: Number of input volume channels.
        embed_dim: Number of linear projection output channels (embedding dim).
        norm_layer: Normalization layer.
        flatten_embedding: Return (B, N, E) rather than (B, D, H, W, E).
    """

    def __init__(
        self,
        img_size: int | tuple[int, int, int] = (512, 512, 512),
        patch_size: int | tuple[int, int, int] = (16, 16, 16),
        in_chans: int = 1,  # EM data is often single-channel
        embed_dim: int = 768,
        norm_layer: Callable[..., nn.Module] | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_DHW = make_3tuple(img_size)
        patch_DHW = make_3tuple(patch_size)
        patch_grid_size = (
            image_DHW[0] // patch_DHW[0],
            image_DHW[1] // patch_DHW[1],
            image_DHW[2] // patch_DHW[2],
        )

        self.img_size = image_DHW
        self.patch_size = patch_DHW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1] * patch_grid_size[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_DHW, stride=patch_DHW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) -> (B, N, E), N the number of patches and E the embedding dim."""
        _, _, D, H, W = x.shape

        # Unlike the 2D case, a partial patch is rejected rather than silently cropped by the
        # convolution: volumetric crops are assembled by the caller, so a size that does not tile
        # is a configuration error rather than a resolution the model is expected to handle.
        p_d, p_h, p_w = self.patch_size
        assert D % p_d == 0, f"Input volume depth {D} is not a multiple of patch depth {p_d}"
        assert H % p_h == 0, f"Input volume height {H} is not a multiple of patch height {p_h}"
        assert W % p_w == 0, f"Input volume width {W} is not a multiple of patch width {p_w}"

        x = self.proj(x)  # B, E, D_grid, H_grid, W_grid
        D, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)  # B, (D_grid*H_grid*W_grid), E
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, D, H, W, self.embed_dim)  # B D H W E
        return x

    def flops(self) -> float:
        Do, Ho, Wo = self.patches_resolution
        flops = (
            Do
            * Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        )
        if self.norm is not None:
            flops += Do * Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self) -> None:
        k = 1 / (self.in_chans * (self.patch_size[0] ** 3))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
