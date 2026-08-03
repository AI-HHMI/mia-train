from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.tensorboard import SummaryWriter


def reduce_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Average each metric across all ranks, so logged values describe the global batch.

    Tensor-parallel forwards can return sharded outputs, so DTensors are materialized before
    the reduction.
    """
    reduced: dict[str, float] = {}
    for name, value in metrics.items():
        tensor = value.detach()
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()
        tensor = tensor.float().mean()
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        reduced[name] = float(tensor.item())
    return reduced


class MetricLogger:
    """Writes scalars to stdout and TensorBoard from the primary rank only."""

    def __init__(self, log_dir: Path | None = None, is_primary: bool = True) -> None:
        self._is_primary = is_primary
        self._writer: SummaryWriter | None = None
        if is_primary and log_dir is not None:
            self._writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, step: int, metrics: dict[str, float], prefix: str = "train") -> None:
        if not self._is_primary:
            return
        formatted = "  ".join(f"{name}={value:.6g}" for name, value in sorted(metrics.items()))
        print(f"[{prefix}] step {step}  {formatted}", flush=True)
        if self._writer is not None:
            for name, value in metrics.items():
                self._writer.add_scalar(f"{prefix}/{name}", value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
