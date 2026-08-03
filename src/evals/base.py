from __future__ import annotations

import abc

import torch.nn as nn
import torch.utils.data as data


class BaseEvalTask(nn.Module, abc.ABC):
    """Downstream zero-shot or fine-tuning evaluation loop run against a trained model."""

    @abc.abstractmethod
    def evaluate(self, model: nn.Module, dataloader: data.DataLoader) -> dict[str, float]:
        """Run this task's evaluation loop against `model`; returns metric name -> value."""
