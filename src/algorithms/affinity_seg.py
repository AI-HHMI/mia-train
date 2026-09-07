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
from torch.utils.checkpoint import checkpoint

from data.base import BaseDataset
from layers.common.dense_heads import SubPixelHead, VoxelHead
from models.base import BaseModel

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
        """Channels the head emits: one per offset, short-range block then long-range."""
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
        # one -- so index from the end, exactly as `_step` does.
        return self._decode(tokens, grid, volumes.shape[-SPATIAL_RANK:])

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
    ) -> tuple[torch.Tensor, ...]:
        """One slab's contribution to every metric, as unnormalized sums.

        Sums rather than means because that is what composes: each reported metric is a ratio of
        two of these, so a run split into slabs reports exactly what an undivided one does
        regardless of how the slabs are sized.
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

        # Labels reach `long_range` further than the slab, because the affinity offsets are
        # positive: the last voxels of a slab are compared against the first of the next. Past the
        # volume's end there is nothing to reach for, and `affinities_from_labels` masks those out
        # exactly as it does for an undivided volume.
        label_hi = min(hi * patch + self.long_range, labels.shape[1])
        target, mask = self._targets(labels[:, lo * patch : label_hi])
        width = keep_hi - keep_lo
        target, mask = target[:, :, :width], mask[:, :, :width]

        per_voxel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss_sum = (per_voxel * mask).sum()
        with torch.no_grad():
            correct = ((logits > 0) == (target > 0.5)) & mask
            cut = mask & (target <= 0.5)
            terms = (
                mask.sum(),
                correct.sum(),
                (target * mask).sum(),
                cut.sum(),
                (correct & cut).sum(),
                torch.tensor(float(mask.numel()), device=mask.device),
            )
        return (loss_sum, *(term.float() for term in terms))

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

        with torch.profiler.record_function("encoder"):
            tokens, grid = self.encoder.patch_features(volumes)

        if self.decode_chunks == 1:
            with torch.profiler.record_function("decoder"):
                logits = self._decode(tokens, grid, spatial)
            # NOTE `logits()` is the same pair of calls without the profiler regions; kept separate
            # so the training step's annotations stay where the profiler expects them.
            target, mask = self._targets(labels)

            # Masked mean rather than a masked tensor: the border slab each offset shifts in from
            # has no neighbour, and scoring it would train the network on invented targets.
            per_voxel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            denominator = mask.sum().clamp_min(1.0)
            loss = (per_voxel * mask).sum() / denominator

            with torch.no_grad():
                correct = ((logits > 0) == (target > 0.5)) & mask
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
            return {
                "loss": loss,
                "affinity_accuracy": accuracy,
                "boundary_accuracy": cut_accuracy,
                "target_positive_rate": positive_rate,
                "masked_fraction": mask.float().mean(),
            }

        # Chunked: decode and score one slab of the volume at a time, each inside its own
        # checkpoint, so only one slab's full-resolution activations are ever live. What that buys
        # is the whole reason this exists -- everything downstream of the patch grid is
        # proportional to *voxels*, which is the crop cubed, and on a 7B encoder at a 512-cube it
        # was 72% of the step's time and most of its memory. It is also why the chunks are cheaper
        # than the sum of their parts: a slab's tensors drop back under cuDNN's 2^31 element limit,
        # where its convolutions stop falling back to the int64 direct kernels.
        patch = spatial[0] // grid[0]
        halo = -(-self._refine_reach() // patch)  # ceil, in whole tokens
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
                    use_reentrant=False,
                )
                totals = list(terms) if totals is None else [
                    running + term for running, term in zip(totals, terms, strict=True)
                ]

        assert totals is not None  # `_chunk_spans` never returns an empty list
        loss_sum, mask_sum, correct_sum, positive_sum, cut_sum, cut_correct, elements = totals
        denominator = mask_sum.clamp_min(1.0)
        return {
            "loss": loss_sum / denominator,
            "affinity_accuracy": correct_sum / denominator,
            "boundary_accuracy": cut_correct / cut_sum.clamp_min(1.0),
            "target_positive_rate": positive_sum / denominator,
            "masked_fraction": mask_sum / elements,
        }

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)
