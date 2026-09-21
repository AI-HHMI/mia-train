"""Supervised instance segmentation by affinity prediction.

The training half of the Neuron Instance Segmentation Benchmark task, following the BANIS
baseline: rather than predicting instance *identities* -- which are arbitrary and unbounded in
number -- the network predicts, for each voxel and a few fixed offsets, whether the voxel there
belongs to the same object. Turning those affinities back into instances is a post-processing step
that happens outside this repo, in the benchmark's own tooling, because it needs a connected
components pass over a whole 12-gigavoxel cube and the dependencies that implies.

What lives here is everything that is genuinely training: the target construction, the masked
loss, and a decoder that lifts an encoder's patch tokens back to voxel resolution. Optionally the
head also predicts local shape descriptors (Sheridan et al., 2023) as an auxiliary target beside
the affinities -- the paper's MTLSD network -- built in `affinity.lsd`; see `lsd_sigma` below. The
decoder is part of the algorithm, not the model, for the same reason masked autoencoding keeps its
own -- it exists to serve one objective and is not what you keep afterwards.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from data.base import BaseDataset
from layers.common.dense_heads import SubPixelHead, VoxelHead
from models.base import BaseModel

from .affinity.lsd import (
    BACKGROUND_MODES,
    channel_groups,
    kernel_radius,
    lsd_channels,
    lsd_from_labels,
)
from .affinity.targets import (
    LONG_RANGE,
    SplitDisconnectedLabels,
    affinities_from_labels,
    affinity_offsets,
    relabel_connected,
)
from .base import BaseAlgorithm
from .registry import AlgorithmRegistry

SPATIAL_RANK = 3
DECODERS = ("interpolate", "subpixel")

#: Label dtypes `_prepare_labels` passes through untouched. Signed, because `ignore_index` is
#: negative; integer, because a float label cannot be trusted to have kept its ids distinct.
SIGNED_INTEGER = (torch.int8, torch.int16, torch.int32, torch.int64)


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

    `lsd_sigma` turns on the paper's MTLSD form: the head also predicts the ten local shape
    descriptors of Sheridan et al. (2023) as an auxiliary target, sharing everything but the last
    1x1 convolution with the affinities -- exactly where the reference network splits its outputs.
    The descriptors are built on the device from the same labels the affinities are (see
    `affinity.lsd` for why not in a worker), trained with masked MSE on a sigmoid output as in the
    reference, and added to the affinity loss with weight `lsd_weight`. `lsd_sigma` is the Gaussian
    window's standard deviation in the data's physical units (nanometres for the lmd stores; a
    scalar or one value per axis), converted per sample from the batch's voxel size, so one setting
    means the same window on 9x9x20 nm serial-section data as on 8 nm isotropic FIB-SEM.
    `lsd_downsample` computes the statistics on a strided grid and nearest-upsamples, the
    reference's `downsample`; its cost grows with objects times voxels, and 2 (the default, and
    the paper's setting) is the difference between 40-75 ms and 300-630 ms per 256^3 sample of
    50-110 objects on a B300 (`mia-train-experiments/lsd_aux_v1/probes/lsd_target_cost`).
    `lsd_background` decides whether background voxels are left out of the descriptor loss
    (`"ignore"`, the reference's default) or supervised towards the all-zero descriptor (`"zero"`).
    Prediction is untouched: `logits()` and the artifact carry the affinity channels only.

    `decode_chunks` splits everything downstream of the patch grid into that many slabs along the
    first spatial axis, each decoded and scored inside its own checkpoint, so only one slab's
    voxel-resolution activations are live at a time. 1 (the default) is the undivided path,
    unchanged.

    It is worth having because this half of the algorithm is proportional to *voxels* -- the crop
    cubed -- while the encoder is proportional to tokens, so past a certain crop the head is the
    whole cost. Measured on a 7B DINOv3 at a 512-cube on B300s
    (`experiments/b300_capability_run`): 72% of the step's time was `convolution_backward` in this
    head, and the memory frontier was set by its voxel-resolution tensors rather than by the 6.7B
    parameters in front of them.

    Two things make the slabs exact rather than approximate, and both are pinned by
    `tests/unit/test_affinity_chunked.py` against the undivided path:

      * each slab is decoded with a halo wide enough for `SubPixelHead.refine`'s convolutions and
        cropped afterwards, so seam voxels see the context they would have seen, and
      * each slab's targets are built from labels reaching `long_range` past its end, because the
        affinity offsets are positive and a slab's last voxels are compared against the next
        slab's first.

    Metrics accumulate as ratios of sums, never as means of means, so uneven slabs weight
    correctly. The halo costs roughly 1.25-1.5x the head's arithmetic depending on slab width, and
    tends to buy more than it costs: a slab's tensors are small enough to stay under cuDNN's 2^31
    element limit, where the same convolutions stop falling back to its int64 direct kernels.

    Only the sub-pixel head can be chunked. The interpolating one resizes the whole patch grid in a
    single `F.interpolate`, whose scale factor comes from the sizes it is handed -- a slab plus halo
    would sample at a different rate, and the seams would be wrong with no shape to catch it.
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
        decode_chunks: int = 1,
        lsd_sigma: float | Sequence[float] | None = None,
        lsd_weight: float = 1.0,
        lsd_downsample: int = 2,
        lsd_background: str = "ignore",
        pixel_size_key: str = "pixel_size",
    ) -> None:
        super().__init__(model, dataset)
        if long_range < 1:
            raise ValueError(f"long_range must be at least 1 voxel, got {long_range}")
        if decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")
        if decode_chunks < 1:
            raise ValueError(f"decode_chunks must be at least 1, got {decode_chunks}")
        if decode_chunks > 1 and decoder != "subpixel":
            # The interpolating head resizes the *whole* patch grid in one `F.interpolate`, and a
            # chunk of that resize is not a resize of the chunk: the scale factor is derived from
            # the sizes it is given, so a slab plus halo would sample at a different rate and the
            # seams would be wrong in a way no shape check catches. The sub-pixel head decodes each
            # token into its own disjoint block, which is what makes slabs exact.
            raise ValueError(
                f"decode_chunks > 1 needs decoder = 'subpixel', got {decoder!r}"
            )

        self.lsd_sigma: tuple[float, ...] | None = None
        if lsd_sigma is not None:
            sigma = (
                (float(lsd_sigma),) * SPATIAL_RANK
                if isinstance(lsd_sigma, int | float)
                else tuple(float(s) for s in lsd_sigma)
            )
            if len(sigma) != SPATIAL_RANK or any(s <= 0 for s in sigma):
                raise ValueError(
                    f"lsd_sigma must be a positive scalar or {SPATIAL_RANK} positive values in the "
                    f"data's physical units, got {lsd_sigma!r}"
                )
            if lsd_weight < 0:
                raise ValueError(f"lsd_weight must be non-negative, got {lsd_weight}")
            if lsd_downsample < 1:
                raise ValueError(f"lsd_downsample must be at least 1, got {lsd_downsample}")
            if lsd_background not in BACKGROUND_MODES:
                raise ValueError(
                    f"lsd_background must be one of {BACKGROUND_MODES}, got {lsd_background!r}"
                )
            self.lsd_sigma = sigma
        self.lsd_weight = lsd_weight
        self.lsd_downsample = lsd_downsample
        self.lsd_background = lsd_background
        self.pixel_size_key = pixel_size_key
        #: Descriptor channels the head emits after the affinity ones; 0 when the target is off.
        self.lsd_channels = lsd_channels(SPATIAL_RANK) if self.lsd_sigma is not None else 0

        self.input_axes = self._resolve_input_axes(input_axes, dataset)
        self.input_key = input_key
        self.label_key = label_key
        self.ignore_index = ignore_index
        self.split_disconnected = split_disconnected
        self.offsets = affinity_offsets(SPATIAL_RANK, long_range)
        self.decoder_kind = decoder
        self.decode_chunks = decode_chunks
        self.long_range = long_range
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
        # Affinities first, descriptors after. One wider output convolution rather than two heads:
        # the two are the same function, and the reference network is built the same way.
        head_channels = len(self.offsets) + self.lsd_channels
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
                embed_dim, patch_size, head_channels,
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
                nn.Conv3d(decoder_hidden_dim, head_channels, kernel_size=1),
                mode="trilinear",
            )

    def sample_transform(self) -> SplitDisconnectedLabels | None:
        """Hand the connected-components pass to the dataloader's workers, when there is one to do.

        `None` only when `split_disconnected` is off, since then there is no work at all. cc3d is a
        core dependency, so the pass always has somewhere to go; `_targets` still keeps its
        on-device path for callers that never attach a transform -- a test, or `predict.py` -- and
        that path is a placement difference rather than a fallback.
        """
        if not self.split_disconnected:
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
        """(B, *label axes) -> (B, X, Y, Z), in a signed integer type.

        Labels arrive without the channel axis the image carries (miao returns `(L, X, Y, Z)` for
        a 3D label group beside a `cxyz` image), so the level axis is located against the axis
        string with `c` removed rather than reusing the model's `prepare_input`.

        **The dtype is preserved when it is already a signed integer, and widened to int64
        otherwise.** This used to widen unconditionally, and at a large crop that made it the
        biggest tensor in the step: `.long()` on an int32 label volume cannot be a view, so it
        allocates a second copy at twice the size while the caller's batch still holds the first.
        At a 1600-cube the pair is 45.8 GiB of a ~265 GiB step, 30.5 GiB of which is the copy --
        an order of magnitude more than any parallelism setting moved
        (`experiments/b300_capability_run`).

        Nothing downstream reads the width. `affinities_from_labels` and `relabel_connected` only
        ever evaluate `labels > 0`, `labels != ignore_index` and `a == b`: membership and sign,
        which mean the same thing in any signed integer type.

        What the widening *was* worth is kept. A label that arrives as a float cannot be trusted to
        have preserved its own ids -- this repo has already lost 64-bit segment ids to a trip
        through float32, distinct neurons merging into one while the dtype still read as integral
        afterwards -- and an unsigned label cannot represent `ignore_index`. Both are still
        converted, because for those the conversion carries information. A signed integer is left
        alone, because for it the conversion carries none.
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
        if levels == 1:
            return self._as_signed_integer(labels.squeeze(level_dim))

        # Multi-scale batch: supervise the FINEST level, index 0. miao returns one label array per
        # scale, and a multi-scale encoder's dense head predicts into the finest grid (see
        # `MuViT3D.patch_features`), so level 0 is the one co-registered with the logits. The
        # coarser label levels describe a larger physical extent at the same voxel count and are
        # not targets -- scoring against one would supervise the model on a different task at a
        # different scale, and the shapes would not complain.
        #
        # miao orders levels fine-to-coarse and MuViT3D enforces that ordering on its `levels`, so
        # index 0 is the finest by construction rather than by convention.
        return self._as_signed_integer(labels.select(level_dim, 0))

    @staticmethod
    def _as_signed_integer(labels: torch.Tensor) -> torch.Tensor:
        """`labels` unchanged if it is already a signed integer, else a int64 copy of it."""
        return labels if labels.dtype in SIGNED_INTEGER else labels.long()

    #: What a prediction over this algorithm's output means, for the artifact a predictor writes.
    #: Declared here rather than inferred by the predictor: `(6, X, Y, Z)` of floats could equally
    #: be six class scores, and thresholding those as affinities yields a segmentation rather than
    #: an error.
    prediction_kind = "affinity"

    @property
    def prediction_channels(self) -> int:
        """Channels a prediction carries: one per offset, short-range block then long-range.

        Never the descriptors. They are an auxiliary training target, and an affinity artifact is
        six channels by contract -- the scorer refuses any other count.
        """
        return len(self.offsets)

    @staticmethod
    def squash(logits: torch.Tensor) -> torch.Tensor:
        """Logits -> the stored representation, `sigmoid(0.2 * logit)`.

        BANIS' `scale_sigmoid`. Training uses plain `binary_cross_entropy_with_logits`, so the
        logits are directly comparable between the two codebases; the 0.2 is only how BANIS stores
        and thresholds them, and reproducing it is what makes a threshold mean the same thing in
        both pipelines. A predictor records the convention beside the data.
        """
        return torch.sigmoid(0.2 * logits)

    #: How `squash` should be described in an artifact, so a consumer need not guess.
    squash_convention = "sigmoid(0.2 * logit)"

    def logits(self, volumes: torch.Tensor) -> torch.Tensor:
        """(B, C, *spatial) input -> (B, n_offsets, *spatial) affinity logits.

        Public for the same reason `semantic_seg.logits` is: prediction over a whole volume drives
        the model directly and needs scores for a window rather than a loss. Previously a caller had
        to reach for `encoder.patch_features` and the private `_decode` and reproduce their pairing,
        which is a copy of `_step`'s middle that could drift from it.
        """
        tokens, grid = self.encoder.patch_features(volumes)
        # The last SPATIAL_RANK axes are the spatial ones under both encoder layouts --
        # (B, C, D, H, W) from a single-scale encoder and (B, L, C, D, H, W) from a multi-scale
        # one -- so index from the end, exactly as `_step` does. The affinity block only: the
        # descriptor channels, when the head has them, are not part of a prediction.
        return self._decode(tokens, grid, volumes.shape[-SPATIAL_RANK:])[:, : len(self.offsets)]

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

    def _split_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """The connected-components split, unless the dataloader's workers already did it."""
        if self.split_disconnected and not self._split_delegated:
            # Per sample: components must not be shared across a batch, and the ids of one crop
            # say nothing about another's.
            with torch.profiler.record_function("relabel_connected"):
                labels = torch.stack([relabel_connected(sample) for sample in labels])
        return labels

    def _affinity_targets(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split (B, X, Y, Z) instance ids -> (affinity target, loss mask), both float/bool.

        Annotated because this is the part of the step a FLOP counter cannot see. `mfu` scores
        every operation here at zero -- they are comparisons, gathers and scatters, not
        multiply-adds -- while `relabel_connected` alone runs a data-dependent number of
        device-to-host synchronizations per sample. A trace is the only thing that shows it.
        """
        with torch.profiler.record_function("affinity_targets"):
            target, mask = affinities_from_labels(labels, self.offsets, self.ignore_index)
            return target.float(), mask

    def _targets(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, X, Y, Z) instance ids as they arrive -> (affinity target, loss mask).

        Splits first when that is this algorithm's job. The step itself calls the two halves
        separately, so that one split serves both the affinities and the descriptors.
        """
        return self._affinity_targets(self._split_labels(labels))

    def _lsd_targets(
        self, labels: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split labels -> (descriptor target, loss mask), on the device. See `affinity.lsd`."""
        with torch.profiler.record_function("lsd_targets"):
            return lsd_from_labels(
                labels, sigma, self.lsd_downsample, self.ignore_index, self.lsd_background
            )

    def _sigma_voxels(self, batch: Any) -> torch.Tensor | None:
        """`lsd_sigma` in voxels of this batch's finest level, per sample: `(B, 3)`, or None.

        Physical units divided by the voxel size miao reports for level 0, the level the labels
        are supervised at. Per sample, because a batch drawn from volumes of different voxel
        sizes is a different window in voxels for each of them.
        """
        if self.lsd_sigma is None:
            return None
        if self.pixel_size_key not in batch:
            raise KeyError(
                f"batch has no {self.pixel_size_key!r} key, so lsd_sigma={self.lsd_sigma} in "
                "physical units cannot be converted to voxels. miao's datasets report it; got "
                f"keys {sorted(batch)}"
            )
        pixel_size = batch[self.pixel_size_key]
        if pixel_size.ndim != 3 or pixel_size.shape[-1] != SPATIAL_RANK:
            raise ValueError(
                f"{self.pixel_size_key!r} must be (B, levels, {SPATIAL_RANK}) in the labels' "
                f"spatial axis order, got {tuple(pixel_size.shape)}"
            )
        sigma = torch.tensor(self.lsd_sigma, dtype=torch.float64, device=pixel_size.device)
        return sigma / pixel_size[:, 0].to(torch.float64)

    def _lsd_sums(
        self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """One region's descriptor loss and diagnostics as unnormalised sums.

        `(loss, masked voxels, voxels, *squared error per channel group)`: sums, so that slabs
        compose exactly as the affinity terms do. Sigmoid then squared error, the reference's
        loss, in float32 whatever precision the head ran in -- the target is float32 and the
        error is small numbers squared.
        """
        squared = (torch.sigmoid(logits.float()) - target) ** 2 * mask
        with torch.no_grad():
            blocks = channel_groups(SPATIAL_RANK).values()
            groups = tuple(squared[:, block].sum() for block in blocks)
            voxels = mask.sum().float()
            elements = torch.tensor(float(mask.numel()), device=mask.device)
        return (squared.sum(), voxels, elements, *groups)

    def _lsd_metrics(self, sums: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """`_lsd_sums`, possibly accumulated over slabs -> the logged descriptor metrics."""
        loss_sum, voxels, elements, *groups = sums
        denominator = voxels.clamp_min(1.0)
        metrics = {"loss_lsd": loss_sum / (denominator * self.lsd_channels)}
        # Per group, so a curve shows which statistic the head is learning: the offsets and the
        # size are the easy ones, the Pearson coefficients the hard ones.
        for (name, block), total in zip(channel_groups(SPATIAL_RANK).items(), groups, strict=True):
            metrics[f"lsd_mse_{name}"] = total / (denominator * (block.stop - block.start))
        metrics["lsd_masked_fraction"] = voxels / elements
        return metrics

    def _with_lsd(
        self, affinity: dict[str, torch.Tensor], lsd: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Both objectives in one metrics dict: the total under `loss`, each part under its name."""
        loss_affinity = affinity["loss"]
        return {
            "loss": loss_affinity + self.lsd_weight * lsd["loss_lsd"],
            "loss_affinity": loss_affinity,
            **{name: value for name, value in affinity.items() if name != "loss"},
            **lsd,
        }

    def _refine_reach(self) -> int:
        """Voxels of context the head's full-resolution convolutions read on each side.

        `SubPixelHead.refine` is `refine_depth` convolutions of width 3, so each one reaches one
        voxel; `project`, the expansion and `out` are all per-token or 1x1 and reach none. A slab
        decoded with this much halo, then cropped, is elementwise identical to the same slab of an
        undivided decode -- which is what `tests/unit/test_affinity_chunked.py` pins.
        """
        return sum(
            1
            for module in self.decoder_out.modules()
            if isinstance(module, nn.Conv3d) and max(module.kernel_size) > 1
        )

    def _chunk_spans(self, extent: int) -> list[tuple[int, int]]:
        """`decode_chunks` contiguous spans of the patch grid's first axis, near-equal in size.

        The first axis and not another, because patch tokens arrive in row-major grid order: a
        contiguous *range of tokens* is exactly a slab along that axis, so a chunk is a slice
        rather than a gather. Remainders go to the earliest chunks.
        """
        chunks = min(self.decode_chunks, extent)
        base, extra = divmod(extent, chunks)
        spans, start = [], 0
        for index in range(chunks):
            stop = start + base + (1 if index < extra else 0)
            spans.append((start, stop))
            start = stop
        return spans

    def _chunk_terms(
        self,
        tokens: torch.Tensor,
        grid: tuple[int, ...],
        labels: torch.Tensor,
        span: tuple[int, int],
        halo: int,
        patch: int,
        sigma: torch.Tensor | None,
        lsd_halo: int,
    ) -> tuple[torch.Tensor, ...]:
        """One slab's contribution to every metric, as unnormalized sums.

        Sums rather than means because that is what composes: each reported metric is a ratio of
        two of these, so a run split into slabs reports exactly what an undivided one does
        regardless of how the slabs are sized.

        `labels` arrive already split. Their slab reaches `lsd_halo` voxels *before* the slab and
        `max(long_range, lsd_halo)` past it: the affinity offsets are positive and compare a slab's
        last voxels against the next slab's first, while a descriptor's window is symmetric and
        reads `kernel_radius` cells on either side. Both targets are built on the extended slab
        and cropped back, which is what keeps them equal to the undivided volume's
        (`tests/unit/test_affinity_chunked.py`). Being inside the checkpointed region, they are
        rebuilt in the backward pass with the slab's activations; that is the price of not holding
        a voxel-resolution target per slab across the whole backward, and it is a price only the
        descriptors make noticeable.
        """
        lo, hi = span
        # Halo tokens on each side feed the refine convolutions the context they would have had
        # in an undivided decode; the volume's own faces have none, which is also what an
        # undivided decode sees there.
        token_lo, token_hi = max(lo - halo, 0), min(hi + halo, grid[0])
        plane = grid[1] * grid[2]
        slab = tokens[:, token_lo * plane : token_hi * plane, :]
        slab_grid = (token_hi - token_lo, grid[1], grid[2])

        logits = self._decode(slab, slab_grid, torch.Size(s * patch for s in slab_grid))
        # Back to the slab's own voxels, dropping the halo the convolutions have now consumed.
        keep_lo, keep_hi = (lo - token_lo) * patch, (hi - token_lo) * patch
        logits = logits[:, :, keep_lo:keep_hi]
        width = keep_hi - keep_lo
        n_affinity = len(self.offsets)

        # Past the volume's end there is nothing to reach for, and `affinities_from_labels` masks
        # those out exactly as it does for an undivided volume.
        label_lo = max(lo * patch - lsd_halo, 0)
        offset = lo * patch - label_lo
        label_hi = min(hi * patch + max(self.long_range, lsd_halo), labels.shape[1])
        target, mask = self._affinity_targets(labels[:, label_lo:label_hi])
        target, mask = target[:, :, offset : offset + width], mask[:, :, offset : offset + width]

        affinity_logits = logits[:, :n_affinity]
        per_voxel = F.binary_cross_entropy_with_logits(affinity_logits, target, reduction="none")
        loss_sum = (per_voxel * mask).sum()
        with torch.no_grad():
            correct = ((affinity_logits > 0) == (target > 0.5)) & mask
            cut = mask & (target <= 0.5)
            terms = (
                mask.sum(),
                correct.sum(),
                (target * mask).sum(),
                cut.sum(),
                (correct & cut).sum(),
                torch.tensor(float(mask.numel()), device=mask.device),
            )
        affinity_terms = (loss_sum, *(term.float() for term in terms))
        if sigma is None:
            return affinity_terms

        lsd_hi = min(hi * patch + lsd_halo, labels.shape[1])
        lsd_target, lsd_mask = self._lsd_targets(labels[:, label_lo:lsd_hi], sigma)
        lsd_target = lsd_target[:, :, offset : offset + width]
        lsd_mask = lsd_mask[:, :, offset : offset + width]
        return (*affinity_terms, *self._lsd_sums(logits[:, n_affinity:], lsd_target, lsd_mask))

    def _step(self, batch: Any) -> dict[str, torch.Tensor]:
        if self.label_key not in batch:
            raise KeyError(
                f"batch has no {self.label_key!r} key, so there is nothing to supervise against. "
                "Set `label_key` on the dataset's volumes so miao reads the instance "
                f"segmentation alongside the image; got keys {sorted(batch)}"
            )

        volumes = self.encoder.prepare_input(batch[self.input_key], self.input_axes)
        labels = self._prepare_labels(batch[self.label_key])
        # The last three axes are the spatial ones under both encoder layouts -- (B, C, D, H, W)
        # from a single-scale encoder and (B, L, C, D, H, W) from a multi-scale one -- so index
        # from the end. `shape[2:]` is the same thing for the 5-D case and silently picks up the
        # channel axis for the 6-D one.
        spatial = volumes.shape[-3:]
        if labels.shape[0] != volumes.shape[0] or labels.shape[1:] != spatial:
            raise ValueError(
                f"label crop {tuple(labels.shape)} does not match the image crop "
                f"{tuple(volumes.shape)} it must be co-registered with (expected "
                f"{(volumes.shape[0], *spatial)})"
            )
        sigma = self._sigma_voxels(batch)

        with torch.profiler.record_function("encoder"):
            tokens, grid = self.encoder.patch_features(volumes)

        # Once, for the whole crop, before any slab: connectivity is a property of the volume, and
        # an object that leaves a slab and comes back is one object, not two.
        labels = self._split_labels(labels)
        n_affinity = len(self.offsets)

        if self.decode_chunks == 1:
            with torch.profiler.record_function("decoder"):
                logits = self._decode(tokens, grid, spatial)
            # NOTE `logits()` is the same pair of calls without the profiler regions; kept separate
            # so the training step's annotations stay where the profiler expects them.
            target, mask = self._affinity_targets(labels)
            affinity_logits = logits[:, :n_affinity]

            # Masked mean rather than a masked tensor: the border slab each offset shifts in from
            # has no neighbour, and scoring it would train the network on invented targets.
            per_voxel = F.binary_cross_entropy_with_logits(
                affinity_logits, target, reduction="none"
            )
            denominator = mask.sum().clamp_min(1.0)
            loss = (per_voxel * mask).sum() / denominator

            with torch.no_grad():
                correct = ((affinity_logits > 0) == (target > 0.5)) & mask
                accuracy = correct.sum() / denominator
                positive_rate = (target * mask).sum() / denominator
                # Accuracy restricted to the voxel/offset pairs that a boundary separates. Pooled
                # accuracy is a poor guide on this task -- the target is ~83% positive, so
                # predicting "same object" everywhere already scores 0.83 and says nothing -- and
                # it is precisely the negatives that decide whether objects come apart, since a
                # missed cut merges two objects and a spurious one fragments one. Reported
                # separately so that a head getting sharper is visible as a number rather than only
                # in a figure.
                cut = mask & (target <= 0.5)
                cut_total = cut.sum().clamp_min(1.0)
                cut_accuracy = (correct & cut).sum() / cut_total
            metrics = {
                "loss": loss,
                "affinity_accuracy": accuracy,
                "boundary_accuracy": cut_accuracy,
                "target_positive_rate": positive_rate,
                "masked_fraction": mask.float().mean(),
            }
            if sigma is None:
                return metrics
            lsd_target, lsd_mask = self._lsd_targets(labels, sigma)
            sums = self._lsd_sums(logits[:, n_affinity:], lsd_target, lsd_mask)
            return self._with_lsd(metrics, self._lsd_metrics(sums))

        # Chunked: decode and score one slab of the volume at a time, each inside its own
        # checkpoint, so only one slab's full-resolution activations are ever live. What that buys
        # is the whole reason this exists -- everything downstream of the patch grid is
        # proportional to *voxels*, which is the crop cubed, and on a 7B encoder at a 512-cube it
        # was 72% of the step's time and most of its memory. It is also why the chunks are cheaper
        # than the sum of their parts: a slab's tensors drop back under cuDNN's 2^31 element limit,
        # where its convolutions stop falling back to the int64 direct kernels.
        patch = spatial[0] // grid[0]
        halo = -(-self._refine_reach() // patch)  # ceil, in whole tokens
        lsd_halo = 0
        if sigma is not None:
            if patch % self.lsd_downsample:
                raise ValueError(
                    f"lsd_downsample={self.lsd_downsample} must divide the patch size {patch} "
                    "when decode_chunks > 1: a slab starts on a patch boundary, and the "
                    "descriptors' strided grid has to coincide there with the undivided volume's"
                )
            # The window's radius on the strided grid, widened back to fine voxels, so a slab's
            # strided lattice is the undivided volume's own and every kept cell sees its whole
            # window. The batch's widest window, since the samples may differ in voxel size.
            lsd_halo = self.lsd_downsample * kernel_radius(
                float(sigma[:, 0].max()) / self.lsd_downsample
            )
        totals: list[torch.Tensor] | None = None
        with torch.profiler.record_function("decoder"):
            for span in self._chunk_spans(grid[0]):
                terms = checkpoint(
                    self._chunk_terms,
                    tokens,
                    grid,
                    labels,
                    span,
                    halo,
                    patch,
                    sigma,
                    lsd_halo,
                    use_reentrant=False,
                )
                totals = list(terms) if totals is None else [
                    running + term for running, term in zip(totals, terms, strict=True)
                ]

        assert totals is not None  # `_chunk_spans` never returns an empty list
        loss_sum, mask_sum, correct_sum, positive_sum, cut_sum, cut_correct, elements, *lsd = totals
        denominator = mask_sum.clamp_min(1.0)
        metrics = {
            "loss": loss_sum / denominator,
            "affinity_accuracy": correct_sum / denominator,
            "boundary_accuracy": cut_correct / cut_sum.clamp_min(1.0),
            "target_positive_rate": positive_sum / denominator,
            "masked_fraction": mask_sum / elements,
        }
        if not lsd:
            return metrics
        return self._with_lsd(metrics, self._lsd_metrics(tuple(lsd)))

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)
