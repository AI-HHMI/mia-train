from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from prediction.types import VolumePredictor

if TYPE_CHECKING:  # annotation only: algorithms must not depend on data/ at runtime
    from data.base import BaseDataset


class BaseAlgorithm(nn.Module, abc.ABC):
    """Wraps a model with a training strategy (masking, loss, logged metrics).

    `dataset` is the training dataset this algorithm will consume, supplied by the engine at
    construction. Most strategies ignore it; those that must match the data's layout read it
    here rather than having the same setting restated in their own config section.
    """

    def __init__(self, model: nn.Module, dataset: BaseDataset | None = None) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset

    @abc.abstractmethod
    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        """Run one training iteration; the returned dict must include a "loss" key."""

    @abc.abstractmethod
    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        """Run one validation iteration; returns logged metrics for this batch."""

    def sample_transform(self) -> Any | None:
        """Per-sample preprocessing this algorithm needs, to run in the dataloader's workers.

        `None` for most strategies. A strategy answers with a callable when it derives something
        from a sample that is expensive, depends only on that sample, and would otherwise sit on
        the critical path between the batch arriving and the loss, e.g affinity prediction's
        connected-components pass over the label crop.

        The engine attaches the result to the training *and* validation datasets, which is what
        separates this from `[augment]`: augmentation deliberately never touches validation,
        because it changes what a validation number means. This is not augmentation -- it is part
        of constructing the target, and train and validation must construct targets the same way
        or their losses are not comparable.

        Returning a callable is a promise that the algorithm no longer does this work itself, so a
        strategy must decide once, at construction, rather than per call.
        """
        return None

    def volume_predictor(self) -> VolumePredictor | None:
        """How to run this strategy over a whole volume, or `None` for the dense default.

        `None` -- the default, and the right answer for any strategy whose output is a fixed number
        of channels per voxel -- lets the entrypoint fall back to the dense default,
        `prediction.dense.DensePredictor`: tile, run `logits`, `squash`, blend overlapping tiles
        by weighted average. That path needs `logits`,
        `prediction_kind`, `prediction_channels`, `squash` and `squash_convention` on the strategy,
        which is what `affinity_seg` and `semantic_seg` provide.

        A strategy whose whole-volume output is not a dense field answers with its own predictor,
        anything satisfying `prediction.types.VolumePredictor`.
        Promptable segmentation is the case this exists for: its prediction is a *search* over
        prompts that yields a set of masks, reconciled across tiles by overlap rather than by
        averaging, and forcing that through `logits()` would mean a search procedure impersonating
        a field. Returning a predictor here keeps the procedure in `src/algorithms/`, where the
        strategy lives, rather than in a second entrypoint that would have to be duplicated for
        every strategy with a non-dense output.
        """
        return None

    def checkpointable_modules(self) -> tuple[nn.Module, ...]:
        """Submodules of the *strategy* worth recomputing in backward, beside the model's own.

        A strategy that owns a decoder answers with it: a dense head running at input resolution
        can hold more than the encoder that feeds it. The model's blocks are collected separately
        from `BaseModel.checkpointable_modules`, so this covers only what the algorithm adds.
        """
        return ()

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Alias for `training_step`, so the engine can drive training through `__call__`.

        This is what makes plain data-parallel replication correct. `replicate` (DDP) hangs its
        gradient all-reduce off `forward` hooks, so a strategy driven through `training_step`
        directly would skip the sync and train on unsynchronized gradients with no error. Going
        through `forward` puts the whole step — including a model the strategy drives via methods
        of its own, like masked autoencoding calling `embed`/`encode` — inside the hooked region.
        """
        return self.training_step(batch)
