from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data

from algorithms.base import BaseAlgorithm
from data.base import BaseDataset
from distributed.parallel_dims import ParallelDims
from engine.config import TrainerConfig
from engine.trainer import Trainer
from models.base import BaseModel


class _TinyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0


class _ReconstructAlgorithm(BaseAlgorithm):
    """Reconstructs its own input, so loss must fall toward zero if the loop works."""

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": (self.model(batch) - batch).pow(2).mean()}

    def validation_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": (self.model(batch) - batch).pow(2).mean()}


class _FixedDataset(data.Dataset):
    def __init__(self, n: int = 32) -> None:
        generator = torch.Generator().manual_seed(0)
        self._items = [torch.randn(8, generator=generator) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._items[index]


class _SyntheticDataset(BaseDataset):
    def build_dataset(self) -> data.Dataset:
        return _FixedDataset()


def _build_trainer(output_dir: Path, config: TrainerConfig, world_size: int) -> Trainer:
    dims = ParallelDims(dp_shard=world_size)
    mesh = dims.build_mesh("cpu")
    return Trainer(
        algorithm=_ReconstructAlgorithm(_TinyModel()),
        train_dataset=_SyntheticDataset(),
        config=config,
        output_dir=output_dir,
        dims=dims,
        mesh=mesh,
        val_dataset=_SyntheticDataset(),
    )


def _fsdp_training_reduces_loss(rank: int, world_size: int, output_dir: str) -> tuple[float, float]:
    config = TrainerConfig(
        max_steps=30, batch_size=4, lr=1e-2, log_every=1000, precision="fp32", seed=0
    )
    trainer = _build_trainer(Path(output_dir), config, world_size)
    first = trainer.algorithm(next(iter(trainer.train_loader)))["loss"].item()
    trainer.train()
    last = trainer.algorithm(next(iter(trainer.train_loader)))["loss"].item()
    return first, last


def _checkpoint_resumes_at_saved_step(
    rank: int, world_size: int, output_dir: str
) -> tuple[int, int]:
    config = TrainerConfig(
        max_steps=10, batch_size=4, lr=1e-2, log_every=1000, checkpoint_every=10, seed=0
    )
    first_run = _build_trainer(Path(output_dir), config, world_size)
    steps_done = first_run.train()

    # A fresh Trainer over the same output_dir must resume, not retrain from zero.
    resumed = _build_trainer(Path(output_dir), config, world_size)
    resumed_from = resumed.checkpoints.load_latest()
    return steps_done, resumed_from


def _validate_returns_metrics(rank: int, world_size: int, output_dir: str) -> dict[str, float]:
    config = TrainerConfig(max_steps=2, batch_size=4, log_every=1000, seed=0)
    trainer = _build_trainer(Path(output_dir), config, world_size)
    return trainer.validate()


def _dp_rank_covers_every_shard(rank: int, world_size: int) -> int:
    dims = ParallelDims(dp_shard=world_size)
    mesh = dims.build_mesh("cpu")
    assert dims.dp_world_size == world_size
    return dims.dp_rank(mesh)


@pytest.mark.cpu_dist
def test_fsdp_training_reduces_loss(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_fsdp_training_reduces_loss, world_size=2, args=(tmp,))
    for first, last in results:
        assert last < first


@pytest.mark.cpu_dist
def test_checkpoint_resumes_at_saved_step(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_checkpoint_resumes_at_saved_step, world_size=2, args=(tmp,))
    assert all(result == (10, 10) for result in results)


@pytest.mark.cpu_dist
def test_validate_returns_metrics(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_validate_returns_metrics, world_size=2, args=(tmp,))
    for metrics in results:
        assert "loss" in metrics
        assert metrics["loss"] > 0.0


@pytest.mark.cpu_dist
def test_dp_rank_covers_every_shard(run_distributed):
    ranks = run_distributed(_dp_rank_covers_every_shard, world_size=2)
    assert sorted(ranks) == [0, 1]
