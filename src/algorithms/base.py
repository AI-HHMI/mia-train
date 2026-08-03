from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn


class BaseAlgorithm(nn.Module, abc.ABC):
    """Wraps a model with a training strategy (masking, loss, logged metrics)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @abc.abstractmethod
    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        """Run one training iteration; the returned dict must include a "loss" key."""

    @abc.abstractmethod
    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        """Run one validation iteration; returns logged metrics for this batch."""
