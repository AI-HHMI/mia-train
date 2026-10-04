"""How the dense strategies get from an encoder's patch tokens to scores at every voxel.

`affinity_seg` and `semantic_seg` differ in what their scores mean and how they are supervised,
not in how the scores are produced: both pick one of the heads in `layers.common.dense_heads` from
their configuration, feed it what it reads from the encoder, and decode a large crop slab by slab.
`DenseDecoding` is that shared part. It is a mixin on the strategy rather than a module of its own,
so the head's parameters keep the names (`decoder.*`, `decoder_out.*`) every checkpoint carries.

The heads themselves, including how each decodes just a span of voxels, are layers. What lives here
is what a layer may not know about: the encoder -- which features a head needs from it, and whether
it can provide them -- and the strategy's configuration.

The heads (`DECODERS`):

  * `"interpolate"`: a 1x1 projection to `hidden_dim` on the patch grid, trilinear upsampling,
    then a 3x3 and a 1x1 convolution at voxel resolution (`VoxelHead`).
  * `"linear"`: a 1x1 convolution from the token features straight to the output channels on the
    patch grid, then trilinear upsampling, so nothing is learned at voxel resolution. The standard
    dense linear probe (DINO's, and the `linear_probe` decoder of muvit2-experiments).
  * `"subpixel"`: each token decodes its own patch block (`SubPixelHead`).
  * `"unetr"`: UNETR's decoder on tokens from several encoder depths plus the raw image
    (`UNETRHead`).
  * `"unet"`: the same decoder for a hierarchical encoder's own feature maps (`UNetHead`).

**Slabs.** `decode_chunks` splits everything downstream of the patch grid into that many slabs along
the first spatial axis (`_chunk_spans`), and each slab is one call of the head with its `span`, so
only one slab's voxel-resolution tensors exist at a time; the strategies run each slab inside its
own checkpoint. Every head decodes a span exactly, so the slabs together are the undivided decode.
The one condition is the interpolating heads': the crop must be a whole number of patches along the
slab axis, which `VoxelHead` checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from layers.common.dense_heads import (
    CONV,
    INTERPOLATION,
    SubPixelHead,
    UNetHead,
    UNETRHead,
    VoxelHead,
    default_skip_layers,
)
from models.base import BaseModel

DECODERS = ("interpolate", "linear", "subpixel", "unetr", "unet")


class DenseDecoding:
    """Mixin for a strategy that decodes an encoder's patch tokens to voxel-resolution channels.

    The strategy must be an `nn.Module`, set `self.encoder` and `self.decode_chunks`, and call
    `_build_head` from its `__init__`.
    """

    encoder: BaseModel
    decode_chunks: int
    decoder: nn.Module
    decoder_out: nn.Module
    #: Which entry of `DECODERS` the head is.
    decoder_kind: str
    #: 0-based encoder blocks the UNETR head reads, deepest last; None for the other heads.
    skip_layers: tuple[int, ...] | None

    def _build_head(
        self,
        model: BaseModel,
        decoder: str,
        out_channels: int,
        rank: int,
        *,
        hidden_dim: int,
        readout_dim: int,
        refine_depth: int,
        zero_init_output: bool,
        widths: Sequence[int] | None,
        skip_layers: Sequence[int] | None,
        image_skip: bool,
    ) -> None:
        """Set `decoder`, on the patch grid, and `decoder_out`, to voxel resolution."""
        if decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")
        self.decoder_kind = decoder
        self.skip_layers = None
        conv = CONV[rank]
        # `embed_dim` is an int attribute, but reading it off an nn.Module widens its static type,
        # so it is narrowed once here rather than at each use.
        embed_dim: int = model.embed_dim  # type: ignore[assignment]
        if decoder in ("subpixel", "unetr"):
            # Scalar on the DINOv3 models, a tuple on `ViT3D`; normalised as `simmim` does it.
            patch = cast(Any, model).patch_size
            patch_size: tuple[int, ...] = (
                (patch,) * rank if isinstance(patch, int) else tuple(patch)
            )
        if decoder == "subpixel":
            # `_decode` is unchanged by the choice: `SubPixelHead` takes its own projection, so the
            # patch-grid stage is a no-op and both heads share the `(x, size)` call.
            self.decoder = nn.Identity()
            self.decoder_out = SubPixelHead(
                embed_dim, patch_size, out_channels,
                hidden=hidden_dim, readout=readout_dim,
                refine_depth=refine_depth,
                zero_init_output=zero_init_output,
            )
        elif decoder == "unetr":
            if type(model).layer_patch_features is BaseModel.layer_patch_features:
                raise ValueError(
                    f"decoder = 'unetr' reads intermediate encoder layers, which "
                    f"{type(model).__name__} does not provide (it does not implement "
                    "layer_patch_features); use decoder = 'subpixel' or 'interpolate'"
                )
            levels = UNETRHead.levels_for(patch_size)
            depth = len(cast(Any, model).blocks)
            layers = (
                tuple(int(layer) for layer in skip_layers)
                if skip_layers is not None else default_skip_layers(depth, levels)
            )
            if (
                len(layers) != levels
                or any(b <= a for a, b in zip(layers, layers[1:], strict=False))
                or layers[0] < 0 or layers[-1] >= depth
            ):
                raise ValueError(
                    f"decoder_skip_layers must be {levels} strictly increasing 0-based block "
                    f"indices below the encoder's depth {depth} (one per x2 stage of patch "
                    f"{patch_size}, deepest last), got {layers}"
                )
            self.skip_layers = layers
            # Every token grid goes straight to the head, which projects each itself.
            self.decoder = nn.Identity()
            self.decoder_out = UNETRHead(
                embed_dim, patch_size, out_channels,
                in_channels=cast(Any, model).in_chans,
                widths=widths, image_skip=image_skip,
                zero_init_output=zero_init_output,
            )
        elif decoder == "unet":
            if type(model).pyramid_features is BaseModel.pyramid_features:
                raise ValueError(
                    f"decoder = 'unet' reads the encoder's feature map at every stride, which "
                    f"{type(model).__name__} does not provide (it does not implement "
                    "pyramid_features); a ViT has one grid, which is what decoder = 'unetr' is for"
                )
            # The maps go straight to the head, which fuses each at its own stride.
            self.decoder = nn.Identity()
            self.decoder_out = UNetHead(
                cast(Any, model).pyramid_dims, cast(Any, model).pyramid_strides, out_channels,
                in_channels=cast(Any, model).in_chans,
                widths=widths, image_skip=image_skip,
                zero_init_output=zero_init_output,
            )
        elif decoder == "linear":
            # Scores on the patch grid, then interpolated: the output can be no sharper than an
            # interpolation of the token grid, which is what makes this a probe of the features.
            self.decoder = conv(embed_dim, out_channels, kernel_size=1)
            self.decoder_out = VoxelHead(mode=INTERPOLATION[rank])
        else:
            self.decoder = nn.Sequential(
                conv(embed_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
            )
            self.decoder_out = VoxelHead(
                conv(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                conv(hidden_dim, out_channels, kernel_size=1),
                mode=INTERPOLATION[rank],
            )

    def _encode(
        self, volumes: torch.Tensor
    ) -> tuple[torch.Tensor | list[torch.Tensor], tuple[int, ...]]:
        """What the head reads: the final layer's patch tokens, or the UNETR head's skip layers.

        One `(B, N, C)` tensor, or for `decoder = "unetr"` one per entry of `skip_layers`, deepest
        last, all on the same grid. For `decoder = "unet"`, the encoder's map at each stride,
        finest first, with the deepest map's grid.
        """
        if self.decoder_kind == "unet":
            return self.encoder.pyramid_features(volumes)
        if self.skip_layers is not None:
            return self.encoder.layer_patch_features(volumes, self.skip_layers)
        return self.encoder.patch_features(volumes)

    def _decode(
        self,
        tokens: torch.Tensor | list[torch.Tensor],
        grid: tuple[int, ...],
        size: torch.Size | tuple[int, ...],
        volumes: torch.Tensor | None = None,
        span: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """`_encode`'s tokens on `grid` -> (B, out_channels, *size) scores, or only voxels `span`.

        `span = (lo, hi)` is handed to the head, which decodes exactly those voxels of the first
        spatial axis (`layers.common.dense_heads`). `volumes` is the encoder's own input, which only
        the UNETR and U-Net heads read (their full-resolution skip); the other heads decode from the
        tokens alone.
        """
        if self.decoder_kind == "unet":
            assert volumes is not None, "the U-Net head reads the raw image as well as the maps"
            return self.decoder_out(tokens, volumes, span=span)
        if self.skip_layers is not None:
            assert volumes is not None, "the UNETR head reads the raw image as well as the tokens"
            return self.decoder_out([self._fold(t, grid) for t in tokens], volumes, span=span)
        assert isinstance(tokens, torch.Tensor)
        return self.decoder_out(self.decoder(self._fold(tokens, grid)), tuple(size), span=span)

    @staticmethod
    def _fold(tokens: torch.Tensor, grid: tuple[int, ...]) -> torch.Tensor:
        """(B, N, C) patch tokens -> (B, C, *grid), refusing tokens that do not fill `grid`."""
        batch, num_tokens, channels = tokens.shape
        expected = 1
        for extent in grid:
            expected *= extent
        if num_tokens != expected:
            raise ValueError(
                f"encoder returned {num_tokens} tokens but its grid {grid} holds "
                f"{expected}; a dense head cannot fold a token sequence back into a volume "
                "it does not fill"
            )

        # (B, N, C) -> (B, C, *grid). Tokens are in row-major grid order, which is what the
        # encoders' patch embeddings produce and what `patch_features` promises.
        return tokens.transpose(1, 2).reshape(batch, channels, *grid)

    def _chunk_spans(self, extent: int) -> list[tuple[int, int]]:
        """`decode_chunks` contiguous spans of the patch grid's first axis, near-equal in size.

        In tokens; a slab's voxels are its span times the patch size. The first axis and not
        another, because that is the axis every head decodes a span of. Remainders go to the
        earliest chunks.
        """
        chunks = min(self.decode_chunks, extent)
        base, extra = divmod(extent, chunks)
        spans, start = [], 0
        for index in range(chunks):
            stop = start + base + (1 if index < extra else 0)
            spans.append((start, stop))
            start = stop
        return spans

    def _decode_volume(
        self,
        tokens: torch.Tensor | list[torch.Tensor],
        grid: tuple[int, ...],
        size: torch.Size | tuple[int, ...],
        volumes: torch.Tensor,
    ) -> torch.Tensor:
        """`_decode` over the whole volume, in `decode_chunks` slabs when there are several.

        The same scores either way; the slabs only bound how much of the head's voxel-resolution
        half exists at once, which keeps a large crop's prediction under its training's limits.
        """
        if self.decode_chunks == 1:
            return self._decode(tokens, grid, size, volumes)
        patch = size[0] // grid[0]
        return torch.cat(
            [
                self._decode(tokens, grid, size, volumes, span=(lo * patch, hi * patch))
                for lo, hi in self._chunk_spans(grid[0])
            ],
            dim=2,
        )
