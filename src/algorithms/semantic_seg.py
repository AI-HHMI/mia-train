"""Supervised semantic segmentation: one class per voxel.

The other dense task here, `affinity_seg`, predicts *relationships* between neighbouring voxels
because instance identities are arbitrary and unbounded. Semantic classes are neither -- there is
a fixed vocabulary and "mitochondrion" means the same thing in every crop -- so this predicts the
class directly and needs no post-processing to be read as a segmentation.

One algorithm serves 2D and 3D. Rank enters in exactly two places, both derived from the encoder's
own patch grid rather than configured: the convolution used by the head, and the interpolation
mode that lifts it back to input resolution. Everything else -- the loss, the metrics, the axis
handling -- is written against `len(grid)`.

The head is the one `affinity_seg` uses (`algorithms.dense.decoding`): the same choice of decoders,
and the same slab decoding for crops whose voxel-resolution tensors would not fit otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from data.base import BaseDataset
from layers.common.dense_heads import CONV
from models.base import BaseModel

from .base import BaseAlgorithm
from .dense.decoding import DenseDecoding
from .registry import AlgorithmRegistry


@AlgorithmRegistry.register("semantic_seg")
class SemanticSegmentation(DenseDecoding, BaseAlgorithm):
    """Per-voxel class prediction from an encoder's patch tokens.

    `num_classes` is the size of the label vocabulary including background at index 0. CellMap
    ids are sparse -- a crop uses a dozen of the ~60 -- so this is the id space, not the number of
    classes present in any one crop.

    `ignore_index` excludes voxels from the loss. It defaults to -1, which never occurs in a uint8
    label volume, so by default every voxel is supervised and background is a class like any
    other. That is the right reading for CellMap, where 0 means "annotated, and not one of these
    organelles" rather than "unannotated". Where a label volume does mark unannotated voxels, set
    it to their value: they are then left out of the loss and of every metric.

    `class_weights` reweights the loss per class. Dense EM segmentation is severely imbalanced --
    background dominates and small organelles are rare -- so a run that optimises plain accuracy
    can score well while never predicting a rare class at all. Left unset, no reweighting is
    applied; the per-class IoU in the metrics is what exposes the problem.

    `decoder` picks how patch tokens become class scores at voxel resolution, from the heads in
    `layers.common.dense_heads`, built by `algorithms.dense.decoding`:

      * `"interpolate"` (the default, and this algorithm's only head before 2026-10-04): a 1x1
        projection to `decoder_hidden_dim` on the patch grid, trilinear upsampling, then a 3x3 and
        a 1x1 convolution at voxel resolution.
      * `"linear"`: one 1x1 convolution from token features to class scores on the patch grid,
        then trilinear upsampling. The dense linear probe: with the encoder frozen it measures the
        encoder's features and nothing else.
      * `"subpixel"`: each token decodes its own patch block. `decoder_hidden_dim` is its width on
        the patch grid, `decoder_readout_dim` its width at voxel resolution, and
        `decoder_refine_depth` the 3x3 convolutions that mix across block seams.
      * `"unetr"`: UNETR's decoder, on tokens from `decoder_skip_layers` (by default the ends of the
        encoder's quarters) and the raw image. `decoder_widths` are its channels at strides 1, 2,
        4, ..., and `decoder_image_skip` switches the image path. Needs an encoder that implements
        `layer_patch_features`.
      * `"unet"`: the same decoder on a hierarchical encoder's own maps (`pyramid_features`).

    `decoder_zero_init_output` zeroes the sub-pixel, UNETR and U-Net heads' last convolution, with
    `affinity_seg`'s meaning and default. Turn it off when the encoder is being trained too: a
    zeroed output sends the encoder no gradient until it grows (`SubPixelHead`).

    `decode_chunks` decodes and scores the volume in that many slabs along the first spatial axis,
    each inside its own checkpoint, so only one slab's voxel-resolution tensors exist at a time. At
    a 512^3 crop a single 16-channel tensor at voxel resolution already reaches cuDNN's 2^31-element
    limit. Every head can be chunked; the interpolating ones need a crop that is a whole number of
    patches along that axis (`VoxelHead`). The loss and the metrics are accumulated as
    sums over slabs and divided once, so `decode_chunks` changes memory and speed, not what is
    computed (`tests/unit/test_semantic_seg.py`). `logits()` decodes in the same slabs, so
    prediction stays under the same limits as training.

    `checkpoint_decoder` recomputes the head in the backward pass when the volume is decoded whole;
    with `decode_chunks > 1` every slab already is.
    """

    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset | None = None,
        input_axes: str | None = None,
        input_key: str = "img",
        label_key: str = "label",
        num_classes: int = 64,
        decoder_hidden_dim: int = 128,
        ignore_index: int = -1,
        class_weights: tuple[float, ...] | list[float] | None = None,
        checkpoint_decoder: bool = False,
        decoder: str = "interpolate",
        decoder_readout_dim: int = 16,
        decoder_refine_depth: int = 2,
        decoder_zero_init_output: bool = True,
        decoder_widths: Sequence[int] | None = None,
        decoder_skip_layers: Sequence[int] | None = None,
        decoder_image_skip: bool = True,
        decode_chunks: int = 1,
    ) -> None:
        super().__init__(model, dataset)
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if decode_chunks < 1:
            raise ValueError(f"decode_chunks must be at least 1, got {decode_chunks}")

        self.input_axes = self._resolve_input_axes(input_axes, dataset)
        self.input_key = input_key
        self.label_key = label_key
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.checkpoint_decoder = checkpoint_decoder
        self.decode_chunks = decode_chunks
        self.encoder = model

        self.spatial_rank = len([axis for axis in self.input_axes if axis not in "lc"])
        if self.spatial_rank not in CONV:
            raise ValueError(
                f"axis order {self.input_axes!r} implies {self.spatial_rank} spatial axes; this "
                "algorithm supports 2 or 3"
            )
        self._build_head(
            model, decoder, num_classes, self.spatial_rank,
            hidden_dim=decoder_hidden_dim, readout_dim=decoder_readout_dim,
            refine_depth=decoder_refine_depth, zero_init_output=decoder_zero_init_output,
            widths=decoder_widths, skip_layers=decoder_skip_layers, image_skip=decoder_image_skip,
        )

        weights = None if class_weights is None else torch.tensor(list(class_weights)).float()
        if weights is not None and weights.numel() != num_classes:
            raise ValueError(
                f"class_weights has {weights.numel()} entries but num_classes={num_classes}"
            )
        # A buffer, not a plain attribute: it has to follow the module to the GPU, and it belongs
        # in the checkpoint so a resumed run keeps the weighting it was trained with.
        self.register_buffer("class_weights", weights, persistent=weights is not None)

    def checkpointable_modules(self) -> tuple[nn.Module, ...]:
        """The full-resolution half of the head, where this algorithm's memory goes.

        Same reasoning as `affinity_seg`: `decoder` runs on the patch grid and is negligible,
        while everything inside `decoder_out` runs at input resolution and scales with
        `num_classes`. The upsampling is deliberately inside that module, since a checkpointed
        region stores its own inputs and an interpolation done outside would leave its
        full-resolution result held for the whole backward pass.
        """
        return (self.decoder_out,)

    @staticmethod
    def _resolve_input_axes(input_axes: str | None, dataset: BaseDataset | None) -> str:
        """Settle on the sample axis order, preferring the dataset's own answer."""
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
        return axes

    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """(B, *label axes) -> (B, *spatial) int64.

        Labels carry the level axis but not the channel axis -- there is one class per voxel, not
        one per channel -- so the level axis is located against the axis string with 'c' removed.
        """
        label_axes = self.input_axes.replace("c", "")
        expected_dims = len(label_axes) + 1
        if labels.dim() != expected_dims:
            raise ValueError(
                f"labels imply a {expected_dims}-D batch from axis order {label_axes!r}, got "
                f"{tuple(labels.shape)}"
            )
        level_dim = label_axes.index("l") + 1
        levels = labels.shape[level_dim]
        if levels != 1:
            raise ValueError(
                f"semantic targets are single-scale, but this batch carries {levels} levels on "
                f"axis 'l' (shape {tuple(labels.shape)})"
            )
        return labels.squeeze(level_dim).long()

    #: See `affinity_seg.prediction_kind`. Per-voxel class scores, so a predictor writes them as
    #: `class_scores` and an argmax -- not a threshold -- turns them into a labelling.
    prediction_kind = "class_scores"

    @property
    def prediction_channels(self) -> int:
        return self.num_classes

    @staticmethod
    def squash(logits: torch.Tensor) -> torch.Tensor:
        """Class scores are stored as softmax probabilities, so tiles blend commensurably.

        Blending raw logits across overlapping tiles would average quantities whose scale is
        arbitrary per tile; probabilities are on a common scale and sum to one.
        """
        return torch.softmax(logits, dim=1)

    squash_convention = "softmax over classes"

    def logits(self, volumes: torch.Tensor) -> torch.Tensor:
        """(B, C, *spatial) input -> (B, num_classes, *spatial) class scores.

        Public because evaluation drives the model through it directly -- tiled and orthoplane
        inference need scores for a window, not a loss. With `decode_chunks > 1` the head runs in
        the training step's slabs and the slabs are concatenated: the same scores, from tensors
        that stay under the same limits.
        """
        tokens, grid = self._encode(volumes)
        return self._decode_volume(tokens, grid, volumes.shape[-self.spatial_rank:], volumes)

    def _terms(self, scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Scores and labels -> the unnormalised sums every reported number is a ratio of.

        Sums rather than means because that is what composes: a volume decoded in slabs reports
        exactly what it would decoded whole, however the slabs are sized. In order: the summed
        cross-entropy and its normaliser (the supervised voxels, or their total class weight when
        `class_weights` is set -- `F.cross_entropy`'s own 'mean'), the correctly classified and the
        supervised voxels, and per class the intersection, the predicted and the labelled count.
        """
        # `register_buffer` widens the attribute's static type to Tensor | Module; it is a
        # tensor or None by construction here.
        weights: torch.Tensor | None = self.class_weights  # type: ignore[assignment]
        valid = labels != self.ignore_index
        targets = labels[valid]
        loss_sum = F.cross_entropy(
            scores, labels, weight=weights, ignore_index=self.ignore_index, reduction="sum"
        )
        normaliser = (
            valid.sum().to(loss_sum.dtype) if weights is None else weights[targets].sum()
        )
        with torch.no_grad():
            predicted = scores.argmax(dim=1)[valid]
            hit = predicted == targets
            intersection = torch.bincount(targets[hit], minlength=self.num_classes)
            predicted_count = torch.bincount(predicted, minlength=self.num_classes)
            labelled_count = torch.bincount(targets, minlength=self.num_classes)
        return (
            loss_sum, normaliser, hit.sum(), valid.sum(),
            intersection, predicted_count, labelled_count,
        )

    def _slab_terms(
        self,
        tokens: torch.Tensor | list[torch.Tensor],
        grid: tuple[int, ...],
        labels: torch.Tensor,
        span: tuple[int, int],
        volumes: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """One slab's `_terms`: scores of voxels `span` of the first axis, against their labels."""
        scores = self._decode(tokens, grid, volumes.shape[2:], volumes, span=span)
        return self._terms(scores, labels[:, span[0] : span[1]])

    def _step(self, batch: Any) -> dict[str, torch.Tensor]:
        if self.label_key not in batch:
            raise KeyError(
                f"batch has no {self.label_key!r} key, so there is nothing to supervise against; "
                f"got keys {sorted(batch)}"
            )
        volumes = self.encoder.prepare_input(batch[self.input_key], self.input_axes)
        labels = self._prepare_labels(batch[self.label_key])
        if labels.shape[0] != volumes.shape[0] or labels.shape[1:] != volumes.shape[2:]:
            raise ValueError(
                f"label crop {tuple(labels.shape)} does not match the image crop "
                f"{tuple(volumes.shape)} it must be co-registered with"
            )

        tokens, grid = self._encode(volumes)
        size = volumes.shape[2:]
        if self.decode_chunks == 1:
            if self.checkpoint_decoder:
                scores = checkpoint(self._decode, tokens, grid, size, volumes, use_reentrant=False)
            else:
                scores = self._decode(tokens, grid, size, volumes)
            totals = self._terms(scores, labels)
        else:
            # Decode and score one slab at a time, each inside its own checkpoint, so only one
            # slab's voxel-resolution activations are ever live -- which is what `decode_chunks`
            # is for, the head being proportional to voxels while the encoder is to tokens.
            patch = size[0] // grid[0]
            sums: tuple[torch.Tensor, ...] | None = None
            for lo, hi in self._chunk_spans(grid[0]):
                terms = checkpoint(
                    self._slab_terms, tokens, grid, labels, (lo * patch, hi * patch), volumes,
                    use_reentrant=False,
                )
                sums = terms if sums is None else tuple(
                    running + term for running, term in zip(sums, terms, strict=True)
                )
            assert sums is not None  # `_chunk_spans` never returns an empty list
            totals = sums
        return self._metrics(totals)

    @staticmethod
    def _metrics(totals: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """`_terms`' sums over the whole volume -> the loss and the logged metrics."""
        loss_sum, normaliser, correct, supervised, intersection, predicted, labelled = totals
        # A batch in which every voxel is ignored has nothing to learn from. Dividing its summed
        # cross-entropy -- zero, over no voxels -- by 1 rather than by its zero normaliser keeps the
        # loss at exactly 0 with a defined gradient, where a mean would be 0/0 and turn every
        # parameter it reaches into NaN.
        loss = loss_sum / torch.where(normaliser > 0, normaliser, torch.ones_like(normaliser))
        with torch.no_grad():
            # NaN when nothing is supervised: there is no voxel to have got right.
            accuracy = correct / supervised
            # Mean IoU over the classes *present in this batch*, which is the number that moves
            # when a rare organelle starts being predicted. Accuracy will not: background alone
            # can carry it past 0.9 while every organelle is missed. NaN when no class is present,
            # rather than a 0 that would drag the logged average down as if the model had failed.
            present = labelled > 0
            union = predicted + labelled - intersection
            mean_iou = (intersection / union.clamp_min(1))[present].mean()
        return {
            "loss": loss,
            "pixel_accuracy": accuracy,
            "mean_iou": mean_iou,
            "classes_present": present.sum().float(),
        }

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)
