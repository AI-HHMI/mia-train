from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data

from algorithms.base import BaseAlgorithm
from algorithms.registry import AlgorithmRegistry
from data.base import BaseDataset
from data.registry import DataRegistry
from distributed.parallel_dims import ParallelDims
from engine.config import TrainerConfig
from engine.run import build_trainer, resolve_output_dir
from models.base import BaseModel
from models.registry import ModelRegistry
from utils.config import ComponentConfig, RunConfig


class _TinyModel(BaseModel):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width * 2), nn.ReLU(), nn.Linear(width * 2, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0


class _ReconstructAlgorithm(BaseAlgorithm):
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


def _register_components() -> None:
    ModelRegistry.register("tiny")(_TinyModel)
    AlgorithmRegistry.register("reconstruct")(_ReconstructAlgorithm)
    DataRegistry.register("synthetic")(_SyntheticDataset)


def _run_config(world_size: int, max_steps: int = 5) -> RunConfig:
    return RunConfig(
        experiment_name="itest",
        model=ComponentConfig(name="tiny", kwargs={"width": 8}),
        algorithm=ComponentConfig(name="reconstruct"),
        data=ComponentConfig(name="synthetic"),
        trainer=TrainerConfig(max_steps=max_steps, batch_size=4, lr=1e-2, log_every=1000, seed=0),
        parallelism=ParallelDims(dp_shard=world_size),
        val_data=ComponentConfig(name="synthetic"),
    )


def _output_dir_agrees_across_ranks(rank: int, world_size: int) -> str:
    # Stagger the ranks past a whole-second boundary before resolving. Without this the ranks
    # would almost always format the same timestamp anyway, so the test would pass even if
    # `resolve_output_dir` never broadcast and each rank simply trusted its own clock.
    time.sleep(1.5 * rank)
    return str(resolve_output_dir(Path("/tmp/mia_itest_outputs"), "agree"))


def _build_from_registries_and_train(
    rank: int, world_size: int, output_dir: str
) -> tuple[float, float]:
    _register_components()
    config = _run_config(world_size)
    mesh = config.parallelism.build_mesh("cpu")
    trainer = build_trainer(config, Path(output_dir), mesh=mesh)

    batch = next(iter(trainer.train_loader))
    before = trainer.algorithm.training_step(batch)["loss"].item()
    trainer.train()
    after = trainer.algorithm.training_step(batch)["loss"].item()
    return before, after


def _unregistered_model_name_raises(rank: int, world_size: int, output_dir: str) -> bool:
    _register_components()
    config = _run_config(world_size)
    broken = RunConfig(
        experiment_name=config.experiment_name,
        model=ComponentConfig(name="not-registered"),
        algorithm=config.algorithm,
        data=config.data,
        trainer=config.trainer,
        parallelism=config.parallelism,
    )
    try:
        build_trainer(broken, Path(output_dir), mesh=config.parallelism.build_mesh("cpu"))
    except KeyError:
        return True
    return False


@pytest.mark.cpu_dist
def test_output_dir_agrees_across_ranks(run_distributed):
    paths = run_distributed(_output_dir_agrees_across_ranks, world_size=2)
    assert len(set(paths)) == 1, f"ranks disagreed on the run directory: {paths}"


@pytest.mark.cpu_dist
def test_build_from_registries_and_train(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_build_from_registries_and_train, world_size=2, args=(tmp,))
    for before, after in results:
        assert after < before


@pytest.mark.cpu_dist
def test_unregistered_model_name_raises(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        assert all(run_distributed(_unregistered_model_name_raises, world_size=2, args=(tmp,)))
