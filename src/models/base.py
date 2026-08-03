from __future__ import annotations

import abc
from typing import Any

import torch.nn as nn
from torch.distributed.tensor.parallel import ParallelStyle


class BaseModel(nn.Module, abc.ABC):
    """Pure architecture definition; exposes parameter counts and FLOP calculators."""

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ...

    @abc.abstractmethod
    def flops(self, input_shape: tuple[int, ...]) -> int:
        """Estimated forward-pass FLOPs for a single input of the given shape (no batch dim)."""

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    def tensor_parallel_plan(self) -> dict[str, ParallelStyle] | None:
        """Optional module-path -> ParallelStyle plan for TP; None means unsupported."""
        return None
