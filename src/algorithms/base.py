from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

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

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Alias for `training_step`, so the engine can drive training through `__call__`.

        This is what makes plain data-parallel replication correct. `replicate` (DDP) hangs its
        gradient all-reduce off `forward` hooks, so a strategy driven through `training_step`
        directly would skip the sync and train on unsynchronized gradients with no error. Going
        through `forward` puts the whole step — including a model the strategy drives via methods
        of its own, like masked autoencoding calling `embed`/`encode` — inside the hooked region.
        """
        return self.training_step(batch)
