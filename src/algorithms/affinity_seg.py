"""Supervised instance segmentation by affinity prediction.

The training half of the Neuron Instance Segmentation Benchmark task, following the BANIS
baseline: rather than predicting instance *identities* -- which are arbitrary and unbounded in
number -- the network predicts, for each voxel and a few fixed offsets, whether the voxel there
belongs to the same object. Turning those affinities back into instances is a post-processing step
that happens outside this repo, in the benchmark's own tooling, because it needs a connected
components pass over a whole 12-gigavoxel cube and the dependencies that implies.

What lives here is everything that is genuinely training: the target construction, the masked
loss, and a decoder that lifts an encoder's patch tokens back to voxel resolution. The decoder is
part of the algorithm, not the model, for the same reason masked autoencoding keeps its own -- it
exists to serve one objective and is not what you keep afterwards.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.base import BaseDataset
from models.base import BaseModel

from .affinity.targets import (
    LONG_RANGE,
    affinities_from_labels,
    affinity_offsets,
    relabel_connected,
)
from .base import BaseAlgorithm
from .registry import AlgorithmRegistry

SPATIAL_RANK = 3


class VoxelHead(nn.Sequential):
    """The voxel-resolution end of a dense head: upsample to `size`, then apply the layers.

    A plain `nn.Sequential` with the interpolation folded in, so the resolution change and the
    layers that run at that resolution are one module. That matters for activation checkpointing,
    which stores whatever crosses a module's boundary: with the upsampling inside, the boundary is
    the patch grid and nothing full-resolution is retained.

    `size` is an argument rather than a constructor value because the head is meant to serve
    whatever crop it is given -- `nn.Upsample(size=...)` would fix the output shape at build time
    and quietly mis-scale a run that validates at a different crop from the one it trains on.
    """

    def forward(self, x: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:  # type: ignore[override]
        # Left to autocast, which runs this in fp32. That is the most expensive tensor in the
        # algorithm -- upsampling multiplies it by the cube of the patch size, so fp32 costs 32
        # GiB rather than 16 at a 512-cube -- and forcing it to bf16 was measured and rejected:
        # accumulating eight neighbours in bf16 is 1.4x less accurate than accumulating in fp32
        # and rounding once, with worst-case deviations of several percent of the feature scale.
        # Memory is bought with a bigger GPU, not with the one number the head is built to
        # produce.
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for layer in self:
            x = layer(x)
        return x


@AlgorithmRegistry.register("affinity_seg")
class AffinitySegmentation(BaseAlgorithm):
    """Predict short- and long-range affinities from an encoder's patch features.

    `long_range` sets the second offset block. The benchmark uses 10 voxels, far enough that
    getting it right requires more than a local boundary cue, which is what makes it useful
    alongside the nearest-neighbour offsets.

    `ignore_index` marks voxels whose true instance is unknown; they are excluded from the loss
    rather than treated as background. NISB itself has none -- every voxel is either background
    (0) or an instance -- but the reference pipeline reserves -1 for it and datasets with partial
    annotation need it.
    """

    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset | None = None,
        input_axes: str | None = None,
        input_key: str = "img",
        label_key: str = "label",
        long_range: int = LONG_RANGE,
        decoder_hidden_dim: int = 64,
        ignore_index: int = -1,
        split_disconnected: bool = True,
    ) -> None:
        super().__init__(model, dataset)
        if long_range < 1:
            raise ValueError(f"long_range must be at least 1 voxel, got {long_range}")

        self.input_axes = self._resolve_input_axes(input_axes, dataset)
        self.input_key = input_key
        self.label_key = label_key
        self.ignore_index = ignore_index
        self.split_disconnected = split_disconnected
        self.offsets = affinity_offsets(SPATIAL_RANK, long_range)
        self.encoder = model

        # Patch tokens -> voxel-resolution affinity logits. Upsampling is by interpolation to
        # whatever spatial size the input had, rather than a transposed convolution with the
        # patch size baked in, so the same head serves encoders whose patch sizes differ (and
        # `ViT3D`'s tuple patch size as readily as the DINOv3 models' scalar one).
        # `embed_dim` is an int attribute, but reading it off an nn.Module widens its static
        # type, so it is narrowed once here rather than at each use.
        embed_dim: int = model.embed_dim  # type: ignore[assignment]
        self.decoder = nn.Sequential(
            nn.Conv3d(embed_dim, decoder_hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.decoder_out = VoxelHead(
            nn.Conv3d(decoder_hidden_dim, decoder_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(decoder_hidden_dim, len(self.offsets), kernel_size=1),
        )

    def checkpointable_modules(self) -> tuple[nn.Module, ...]:
        """The full-resolution half of the head, which is where this algorithm's memory goes.

        `decoder` runs on the patch grid and costs nothing; everything inside `decoder_out` runs
        at voxel resolution, so each of its tensors is `decoder_hidden_dim` times the size of the
        crop -- 16 GiB apiece at a 512-cube. Recomputing them is cheap beside holding them.

        The upsampling is deliberately part of `decoder_out` rather than done by the caller: a
        checkpointed region stores its own inputs, so an interpolation performed outside would
        leave its full-resolution result held for the whole backward pass and give back only half
        of what checkpointing is worth here.
        """
        return (self.decoder_out,)

    @staticmethod
    def _resolve_input_axes(input_axes: str | None, dataset: BaseDataset | None) -> str:
        """Settle on the sample axis order, preferring the dataset's own answer.

        Mirrors `MAE._resolve_input_axes`: an explicit setting that contradicts the dataset is
        rejected rather than silently winning.
        """
        from_dataset = dataset.sample_axes if dataset is not None else None
        if input_axes is not None and from_dataset is not None and input_axes != from_dataset:
            raise ValueError(
                f"input_axes={input_axes!r} contradicts the dataset's sample_axes="
                f"{from_dataset!r}; remove input_axes and let the dataset declare the layout"
            )
        axes = input_axes or from_dataset
        if axes is None:
            raise ValueError(
                "no axis order available: pass input_axes, or use a dataset that declares "
                "sample_axes"
            )

        # The affinity channels are defined against the benchmark's x,y,z index order -- the same
        # order its skeletons are indexed in. A dataset delivering z,y,x would produce targets
        # transposed relative to the published channel convention, train perfectly well, and score
        # as nonsense. Nothing downstream can detect it, so it is checked here.
        spatial = [axis for axis in axes if axis not in "lc"]
        if spatial != ["x", "y", "z"]:
            raise ValueError(
                f"axis order {axes!r} gives spatial order {''.join(spatial)!r}, but affinity "
                "targets are defined in x,y,z order to match the benchmark's channel convention "
                "and skeleton indexing. Set the dataset's output_axes so the spatial axes read "
                'x,y,z (e.g. "lcxyz").'
            )
        return axes

    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """(B, *label axes) -> (B, X, Y, Z) int64.

        Labels arrive without the channel axis the image carries (miao returns `(L, X, Y, Z)` for
        a 3D label group beside a `cxyz` image), so the level axis is located against the axis
        string with `c` removed rather than reusing the model's `prepare_input`.
        """
        label_axes = self.input_axes.replace("c", "")
        expected_dims = len(label_axes) + 1
        if labels.dim() != expected_dims:
            raise ValueError(
                f"labels imply a {expected_dims}-D batch from axis order {label_axes!r} (the "
                f"sample axes {self.input_axes!r} without a channel), got {tuple(labels.shape)}. "
                "A dataset whose label group carries its own channel axis is not supported."
            )

        level_dim = label_axes.index("l") + 1
        levels = labels.shape[level_dim]
        if levels != 1:
            raise ValueError(
                f"affinity targets are single-scale, but this batch carries {levels} scale levels "
                f"on axis 'l' (shape {tuple(labels.shape)}). Configure the dataset for one level."
            )
        return labels.squeeze(level_dim).long()

    def _decode(
        self, tokens: torch.Tensor, grid: tuple[int, ...], size: torch.Size
    ) -> torch.Tensor:
        """(B, N, C) patch tokens on `grid` -> (B, n_offsets, *size) affinity logits."""
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
        x = tokens.transpose(1, 2).reshape(batch, channels, *grid)
        x = self.decoder(x)
        return self.decoder_out(x, tuple(size))

    def _targets(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, X, Y, Z) instance ids -> (affinity target, loss mask), both float/bool."""
        if self.split_disconnected:
            # Per sample: components must not be shared across a batch, and the ids of one crop
            # say nothing about another's.
            labels = torch.stack([relabel_connected(sample) for sample in labels])
        target, mask = affinities_from_labels(labels, self.offsets, self.ignore_index)
        return target.float(), mask

    def _step(self, batch: Any) -> dict[str, torch.Tensor]:
        if self.label_key not in batch:
            raise KeyError(
                f"batch has no {self.label_key!r} key, so there is nothing to supervise against. "
                "Set `label_key` on the dataset's volumes so miao reads the instance "
                f"segmentation alongside the image; got keys {sorted(batch)}"
            )

        volumes = self.encoder.prepare_input(batch[self.input_key], self.input_axes)
        labels = self._prepare_labels(batch[self.label_key])
        if labels.shape[0] != volumes.shape[0] or labels.shape[1:] != volumes.shape[2:]:
            raise ValueError(
                f"label crop {tuple(labels.shape)} does not match the image crop "
                f"{tuple(volumes.shape)} it must be co-registered with (expected "
                f"{(volumes.shape[0], *volumes.shape[2:])})"
            )

        tokens, grid = self.encoder.patch_features(volumes)
        logits = self._decode(tokens, grid, volumes.shape[2:])
        target, mask = self._targets(labels)

        # Masked mean rather than a masked tensor: the border slab each offset shifts in from has
        # no neighbour, and scoring it would train the network on invented targets.
        per_voxel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        denominator = mask.sum().clamp_min(1.0)
        loss = (per_voxel * mask).sum() / denominator

        with torch.no_grad():
            correct = ((logits > 0) == (target > 0.5)) & mask
            accuracy = correct.sum() / denominator
            positive_rate = (target * mask).sum() / denominator
        return {
            "loss": loss,
            "affinity_accuracy": accuracy,
            "target_positive_rate": positive_rate,
            "masked_fraction": mask.float().mean(),
        }

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)
