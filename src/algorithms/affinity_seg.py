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

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.base import BaseDataset
from layers.common.dense_heads import SubPixelHead, VoxelHead
from models.base import BaseModel

from .affinity.targets import (
    LONG_RANGE,
    SplitDisconnectedLabels,
    affinities_from_labels,
    affinity_offsets,
    cc3d_available,
    relabel_connected,
)
from .base import BaseAlgorithm
from .registry import AlgorithmRegistry

SPATIAL_RANK = 3
DECODERS = ("interpolate", "subpixel")


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
        decoder: str = "interpolate",
        decoder_hidden_dim: int = 64,
        decoder_readout_dim: int = 16,
        decoder_refine_depth: int = 2,
        decoder_zero_init_output: bool = True,
        ignore_index: int = -1,
        split_disconnected: bool = True,
    ) -> None:
        super().__init__(model, dataset)
        if long_range < 1:
            raise ValueError(f"long_range must be at least 1 voxel, got {long_range}")
        if decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")

        self.input_axes = self._resolve_input_axes(input_axes, dataset)
        self.input_key = input_key
        self.label_key = label_key
        self.ignore_index = ignore_index
        self.split_disconnected = split_disconnected
        self.offsets = affinity_offsets(SPATIAL_RANK, long_range)
        self.decoder_kind = decoder
        self.encoder = model
        # Set by `sample_transform` when the engine takes the connected-components pass off this
        # algorithm's hands and into the dataloader's workers. Until then `_targets` does it
        # itself, so an algorithm driven without the engine -- a test, a notebook -- still
        # produces split targets rather than silently unsplit ones.
        self._split_delegated = False

        # Patch tokens -> voxel-resolution affinity logits, by one of two routes.
        #
        # `"interpolate"` upsamples to whatever spatial size the input had and then convolves at
        # that resolution, so the same head serves encoders whose patch sizes differ (and
        # `ViT3D`'s tuple patch size as readily as the DINOv3 models' scalar one) and crops that
        # are not a whole number of patches. It is the default because every checkpoint this repo
        # has trained on this task carries it.
        #
        # `"subpixel"` gives each token a learned readout of its own patch block instead. It is
        # both sharper in principle -- detail comes from weights rather than from interpolating a
        # coarse field -- and considerably cheaper, because the wide arithmetic stays on the patch
        # grid: measured against this same head at 256^3, roughly a sixth of the multiply-adds and
        # a quarter of the activation memory. It needs the encoder's patch size, and crops
        # divisible by it.
        #
        # `embed_dim` is an int attribute, but reading it off an nn.Module widens its static
        # type, so it is narrowed once here rather than at each use.
        embed_dim: int = model.embed_dim  # type: ignore[assignment]
        if decoder == "subpixel":
            # Scalar on the DINOv3 models, a tuple on `ViT3D`; normalised as `simmim` does it.
            patch = cast(Any, model).patch_size
            patch_size: tuple[int, ...] = (
                (patch,) * SPATIAL_RANK if isinstance(patch, int) else tuple(patch)
            )
            # `_decode` is unchanged by the choice: `SubPixelHead` takes its own projection, so the
            # patch-grid stage is a no-op and both heads share the `(x, size)` call.
            self.decoder: nn.Module = nn.Identity()
            self.decoder_out: nn.Module = SubPixelHead(
                embed_dim, patch_size, len(self.offsets),
                hidden=decoder_hidden_dim, readout=decoder_readout_dim,
                refine_depth=decoder_refine_depth,
                zero_init_output=decoder_zero_init_output,
            )
        else:
            self.decoder = nn.Sequential(
                nn.Conv3d(embed_dim, decoder_hidden_dim, kernel_size=1),
                nn.GELU(),
            )
            self.decoder_out = VoxelHead(
                nn.Conv3d(decoder_hidden_dim, decoder_hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(decoder_hidden_dim, len(self.offsets), kernel_size=1),
                mode="trilinear",
            )

    def sample_transform(self) -> SplitDisconnectedLabels | None:
        """Hand the connected-components pass to the dataloader's workers, when it can be.

        `None` -- keeping the work on the training device -- in the two cases where it cannot:
        when `split_disconnected` is off and there is no work to do at all, and when the optional
        `affinity` extra is not installed. The second is a fallback rather than an error because
        the device path is *correct*, only slow: making a missing optional dependency break every
        existing affinity config would be a worse trade than running them at the speed they
        already run at. It says so rather than degrading in silence, since the symptom otherwise
        is a run that is 40% slower than an identical one elsewhere for no visible reason.
        """
        if not self.split_disconnected:
            return None
        if not cc3d_available():
            print(
                "[affinity] splitting disconnected components on the training device: cc3d is "
                "not installed. This costs ~107 ms per step at 256^3 and reports as zero FLOPs, "
                "so `mfu` will understate the run. Install it with: pip install -e '.[affinity]'",
                flush=True,
            )
            return None
        self._split_delegated = True
        return SplitDisconnectedLabels(self.label_key)

    def checkpointable_modules(self) -> tuple[nn.Module, ...]:
        """The full-resolution half of the head, which is where this algorithm's memory goes.

        `decoder` runs on the patch grid and costs nothing; everything inside `decoder_out` runs
        at voxel resolution, so each of its tensors is a decoder width times the size of the crop
        -- 16 GiB apiece at a 512-cube. Recomputing them is cheap beside holding them.

        The resolution change is deliberately part of `decoder_out` rather than done by the
        caller, under either decoder: a checkpointed region stores its own inputs, so an
        upsampling performed outside would leave its full-resolution result held for the whole
        backward pass and give back only half of what checkpointing is worth here. Both heads
        therefore present a boundary at the patch grid, where a tensor is thousands of times
        smaller than at voxel resolution.
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
        """(B, X, Y, Z) instance ids -> (affinity target, loss mask), both float/bool.

        Annotated because this is the part of the step a FLOP counter cannot see. `mfu` scores
        every operation here at zero -- they are comparisons, gathers and scatters, not
        multiply-adds -- while `relabel_connected` alone runs a data-dependent number of
        device-to-host synchronizations per sample. A trace is the only thing that shows it.
        """
        with torch.profiler.record_function("affinity_targets"):
            if self.split_disconnected and not self._split_delegated:
                # Per sample: components must not be shared across a batch, and the ids of one crop
                # say nothing about another's.
                with torch.profiler.record_function("relabel_connected"):
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

        with torch.profiler.record_function("encoder"):
            tokens, grid = self.encoder.patch_features(volumes)
        with torch.profiler.record_function("decoder"):
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
            # Accuracy restricted to the voxel/offset pairs that a boundary separates. Pooled
            # accuracy is a poor guide on this task -- the target is ~83% positive, so predicting
            # "same object" everywhere already scores 0.83 and says nothing -- and it is precisely
            # the negatives that decide whether objects come apart, since a missed cut merges two
            # objects and a spurious one fragments one. Reported separately so that a head getting
            # sharper is visible as a number rather than only in a figure.
            cut = mask & (target <= 0.5)
            cut_total = cut.sum().clamp_min(1.0)
            cut_accuracy = (correct & cut).sum() / cut_total
        return {
            "loss": loss,
            "affinity_accuracy": accuracy,
            "boundary_accuracy": cut_accuracy,
            "target_positive_rate": positive_rate,
            "masked_fraction": mask.float().mean(),
        }

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)
