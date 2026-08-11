"""Supervised semantic segmentation: one class per voxel.

The other dense task here, `affinity_seg`, predicts *relationships* between neighbouring voxels
because instance identities are arbitrary and unbounded. Semantic classes are neither -- there is
a fixed vocabulary and "mitochondrion" means the same thing in every crop -- so this predicts the
class directly and needs no post-processing to be read as a segmentation.

One algorithm serves 2D and 3D. Rank enters in exactly two places, both derived from the encoder's
own patch grid rather than configured: the convolution used by the head, and the interpolation
mode that lifts it back to input resolution. Everything else -- the loss, the metrics, the axis
handling -- is written against `len(grid)`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.base import BaseDataset
from layers.common.dense_heads import CONV, INTERPOLATION, VoxelHead
from models.base import BaseModel

from .base import BaseAlgorithm
from .registry import AlgorithmRegistry


@AlgorithmRegistry.register("semantic_seg")
class SemanticSegmentation(BaseAlgorithm):
    """Per-voxel class prediction from an encoder's patch tokens.

    `num_classes` is the size of the label vocabulary including background at index 0. CellMap
    ids are sparse -- a crop uses a dozen of the ~60 -- so this is the id space, not the number of
    classes present in any one crop.

    `ignore_index` excludes voxels from the loss. It defaults to -1, which never occurs in a uint8
    label volume, so by default every voxel is supervised and background is a class like any
    other. That is the right reading for CellMap, where 0 means "annotated, and not one of these
    organelles" rather than "unannotated".

    `class_weights` reweights the loss per class. Dense EM segmentation is severely imbalanced --
    background dominates and small organelles are rare -- so a run that optimises plain accuracy
    can score well while never predicting a rare class at all. Left unset, no reweighting is
    applied; the per-class IoU in the metrics is what exposes the problem.
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
    ) -> None:
        super().__init__(model, dataset)
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")

        self.input_axes = self._resolve_input_axes(input_axes, dataset)
        self.input_key = input_key
        self.label_key = label_key
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.checkpoint_decoder = checkpoint_decoder
        self.encoder = model

        self.spatial_rank = len([axis for axis in self.input_axes if axis not in "lc"])
        if self.spatial_rank not in CONV:
            raise ValueError(
                f"axis order {self.input_axes!r} implies {self.spatial_rank} spatial axes; this "
                "algorithm supports 2 or 3"
            )
        conv = CONV[self.spatial_rank]

        embed_dim: int = model.embed_dim  # type: ignore[assignment]
        self.decoder = nn.Sequential(conv(embed_dim, decoder_hidden_dim, kernel_size=1), nn.GELU())
        self.decoder_out = VoxelHead(
            conv(decoder_hidden_dim, decoder_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            conv(decoder_hidden_dim, num_classes, kernel_size=1),
            mode=INTERPOLATION[self.spatial_rank],
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

    def logits(self, volumes: torch.Tensor) -> torch.Tensor:
        """(B, C, *spatial) input -> (B, num_classes, *spatial) class scores.

        Public because evaluation drives the model through it directly -- tiled and orthoplane
        inference need scores for a window, not a loss.
        """
        tokens, grid = self.encoder.patch_features(volumes)
        batch, num_tokens, channels = tokens.shape
        expected = 1
        for extent in grid:
            expected *= extent
        if num_tokens != expected:
            raise ValueError(
                f"encoder returned {num_tokens} tokens but its grid {grid} holds {expected}; a "
                "dense head cannot fold a token sequence back into a volume it does not fill"
            )

        x = tokens.transpose(1, 2).reshape(batch, channels, *grid)
        x = self.decoder(x)
        size = tuple(volumes.shape[2:])
        if self.checkpoint_decoder:
            from utils.checkpointing import checkpointed

            return checkpointed(self.decoder_out, x, size)
        return self.decoder_out(x, size)

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

        scores = self.logits(volumes)
        # `register_buffer` widens the attribute's static type to Tensor | Module; it is a
        # tensor or None by construction here.
        weights: torch.Tensor | None = self.class_weights  # type: ignore[assignment]
        loss = F.cross_entropy(scores, labels, weight=weights, ignore_index=self.ignore_index)

        with torch.no_grad():
            predicted = scores.argmax(dim=1)
            valid = labels != self.ignore_index
            denominator = valid.sum().clamp_min(1)
            accuracy = ((predicted == labels) & valid).sum() / denominator
            # Mean IoU over the classes *present in this batch*, which is the number that moves
            # when a rare organelle starts being predicted. Accuracy will not: background alone
            # can carry it past 0.9 while every organelle is missed.
            intersections, unions = [], []
            for klass in torch.unique(labels[valid]):
                p, t = (predicted == klass) & valid, (labels == klass) & valid
                intersections.append((p & t).sum())
                unions.append((p | t).sum())
            # A crop where every voxel is ignored has no class to score. The loss is already nan
            # there (cross-entropy over an empty selection), so this reports nan too rather than
            # inventing a 0 that would drag the logged average down as if the model had failed.
            iou = (
                torch.stack([i / u.clamp_min(1) for i, u in zip(intersections, unions, strict=True)]
                            ).mean()
                if intersections
                else torch.tensor(float("nan"), device=loss.device)
            )

        return {
            "loss": loss,
            "pixel_accuracy": accuracy,
            "mean_iou": iou,
            "classes_present": torch.tensor(float(len(intersections)), device=loss.device),
        }

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)
