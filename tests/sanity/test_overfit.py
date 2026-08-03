from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data

from algorithms.base import BaseAlgorithm
from data.base import BaseDataset
from engine.config import TrainerConfig
from engine.trainer import Trainer
from models.base import BaseModel


class _TinyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0


class _ReconstructAlgorithm(BaseAlgorithm):
    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": (self.model(batch) - batch).pow(2).mean()}

    def validation_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": (self.model(batch) - batch).pow(2).mean()}


class _SingleBatchDataset(data.Dataset):
    def __init__(self, n: int = 8) -> None:
        generator = torch.Generator().manual_seed(0)
        self._items = [torch.randn(8, generator=generator) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._items[index]


class _SyntheticDataset(BaseDataset):
    def build_dataset(self) -> data.Dataset:
        return _SingleBatchDataset()


@pytest.mark.slow
def test_mini_model_overfits_synthetic_batch(tmp_path: Path):
    algorithm = _ReconstructAlgorithm(_TinyModel())
    dataset = _SyntheticDataset()
    config = TrainerConfig(
        max_steps=200, batch_size=8, lr=1e-2, log_every=50, precision="fp32", seed=0
    )
    trainer = Trainer(
        algorithm=algorithm, train_dataset=dataset, config=config, output_dir=tmp_path
    )

    batch = next(iter(trainer.train_loader))
    before = algorithm.training_step(batch)["loss"].item()
    trainer.train()
    after = algorithm.training_step(batch)["loss"].item()

    assert after < before * 0.1, f"expected overfitting, got {before:.4f} -> {after:.4f}"
