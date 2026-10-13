from __future__ import annotations

import dataclasses
import json
import tempfile
import time
from pathlib import Path

import numpy as np
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
from engine.run import _prepare_run_dir, build_trainer, resolve_output_dir
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


class _RandomDraws(data.Dataset):
    """Ignores the index and draws from numpy's global state, as miao's `sampling: random` does.

    PyTorch reseeds that state in every DataLoader worker from the loader's base seed, so what a
    rank's workers draw depends on that seed alone, not on the indices its sampler hands out.
    """

    def __len__(self) -> int:
        return 32

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.random.standard_normal(8).astype(np.float32))


class _RandomDrawsDataset(BaseDataset):
    def build_dataset(self) -> data.Dataset:
        return _RandomDraws()


class _ConstantDraws(data.Dataset):
    """The same item whatever the index: what every rank received before 2026-10-12."""

    def __len__(self) -> int:
        return 32

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.ones(8)


class _ConstantDataset(BaseDataset):
    def build_dataset(self) -> data.Dataset:
        return _ConstantDraws()


class _RecordingAlgorithm(_ReconstructAlgorithm):
    """Records a fingerprint of every batch it trains on."""

    def __init__(self, model: nn.Module, dataset: BaseDataset | None = None) -> None:
        super().__init__(model, dataset)
        self.seen: list[float] = []

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen.append(float(batch.double().sum()))
        return super().training_step(batch)


class _RankDependentHead(BaseAlgorithm):
    """Each rank trains through a different head, as a promptable correction round that feeds its
    mask prompt on some ranks and not on others uses different parameters on each."""

    def __init__(self, model: nn.Module, dataset: BaseDataset | None = None) -> None:
        super().__init__(model, dataset)
        self.heads = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        head = self.heads[torch.distributed.get_rank() % 2]
        return {"loss": (head(self.model(batch)) - batch).pow(2).mean()}

    def validation_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.training_step(batch)


def _register_components() -> None:
    ModelRegistry.register("tiny")(_TinyModel)
    AlgorithmRegistry.register("reconstruct")(_ReconstructAlgorithm)
    AlgorithmRegistry.register("recording")(_RecordingAlgorithm)
    AlgorithmRegistry.register("rank_dependent_head")(_RankDependentHead)
    DataRegistry.register("synthetic")(_SyntheticDataset)
    DataRegistry.register("random_draws")(_RandomDrawsDataset)
    DataRegistry.register("constant")(_ConstantDataset)


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
    before = trainer.algorithm(batch)["loss"].item()
    trainer.train()
    after = trainer.algorithm(batch)["loss"].item()
    return before, after


def _first_batch_of_random_draws(rank: int, world_size: int, output_dir: str) -> np.ndarray:
    _register_components()
    config = _run_config(world_size)
    # One worker: PyTorch reseeds only worker processes, which is where every real run loads.
    config = dataclasses.replace(
        config,
        data=ComponentConfig(name="random_draws"),
        val_data=None,
        trainer=dataclasses.replace(config.trainer, num_workers=1),
    )
    trainer = build_trainer(config, Path(output_dir), mesh=config.parallelism.build_mesh("cpu"))
    return next(iter(trainer.train_loader)).numpy()


def _parameters_after_unrelated_draws(
    rank: int, world_size: int, output_dir: str, draws: int
) -> list[np.ndarray]:
    _register_components()
    # Whatever ran before `build_trainer` -- here a rank- and run-dependent number of draws --
    # leaves the global generator in its own state. The initialisation must not depend on it.
    torch.rand(draws + rank)
    config = _run_config(world_size)
    trainer = build_trainer(config, Path(output_dir), mesh=config.parallelism.build_mesh("cpu"))
    return [p.full_tensor().detach().numpy() for p in trainer.algorithm.parameters()]


def _first_batch_without_workers(rank: int, world_size: int, output_dir: str) -> np.ndarray:
    _register_components()
    config = _run_config(world_size)
    # No workers: the dataset then draws from numpy's global state in this very process, which
    # PyTorch reseeds only inside workers.
    config = dataclasses.replace(config, data=ComponentConfig(name="random_draws"), val_data=None)
    trainer = build_trainer(config, Path(output_dir), mesh=config.parallelism.build_mesh("cpu"))
    return next(iter(trainer.train_loader)).numpy()


def _batches_across_a_resume(
    rank: int, world_size: int, output_dir: str
) -> tuple[list[float], list[float]]:
    _register_components()
    config = _run_config(world_size, max_steps=2)
    config = dataclasses.replace(
        config,
        algorithm=ComponentConfig(name="recording"),
        data=ComponentConfig(name="random_draws"),
        val_data=None,
        # No FLOP probe, so `seen` holds exactly the batches trained on.
        trainer=dataclasses.replace(
            config.trainer, num_workers=1, checkpoint_every=2, measure_mfu=False
        ),
    )
    first = build_trainer(config, Path(output_dir), mesh=config.parallelism.build_mesh("cpu"))
    first.train()
    longer = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, max_steps=4))
    second = build_trainer(longer, Path(output_dir), mesh=longer.parallelism.build_mesh("cpu"))
    second.train()
    return first.algorithm.seen, second.algorithm.seen


def _train_with_rank_dependent_parameters(
    rank: int, world_size: int, output_dir: str
) -> list[bool]:
    _register_components()
    config = _run_config(world_size, max_steps=3)
    config = dataclasses.replace(
        config, algorithm=ComponentConfig(name="rank_dependent_head"), val_data=None
    )
    trainer = build_trainer(config, Path(output_dir), mesh=config.parallelism.build_mesh("cpu"))
    heads = trainer.algorithm.heads
    before = [p.full_tensor().detach().clone() for p in heads.parameters()]
    trainer.train()
    after = [p.full_tensor().detach() for p in heads.parameters()]
    # Every row, not just the tensor: each rank holds a slice of every head, and a reduce-scatter
    # that pairs one rank's head-0 gradient with another's head-1 gradient (equal sizes, so
    # nothing raises) updates each head only in the slice of the rank that used it.
    return [
        bool((b != a).reshape(b.shape[0], -1).any(dim=1).all())
        for b, a in zip(before, after, strict=True)
    ]


def _train_on_constant_data(rank: int, world_size: int, output_dir: str) -> str | None:
    _register_components()
    config = _run_config(world_size)
    config = dataclasses.replace(config, data=ComponentConfig(name="constant"), val_data=None)
    trainer = build_trainer(config, Path(output_dir), mesh=config.parallelism.build_mesh("cpu"))
    try:
        trainer.train()
    except RuntimeError as error:
        return str(error)
    return None


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
def test_ranks_draw_different_samples(run_distributed):
    """A dataset that samples at random rather than by index gives each rank its own draws, and
    the same ones again on a second run. Seeded alike, every rank's workers drew the same samples:
    one sample per step, copied to every rank, whatever the sampler's indices said."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        first = run_distributed(_first_batch_of_random_draws, world_size=2, args=(tmp,))
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        again = run_distributed(_first_batch_of_random_draws, world_size=2, args=(tmp,))
    assert not np.array_equal(first[0], first[1]), "both ranks drew the same batch"
    for rank in range(2):
        np.testing.assert_array_equal(first[rank], again[rank])


@pytest.mark.cpu_dist
def test_initialisation_follows_the_seed(run_distributed):
    """The model's random initialisation follows `trainer.seed`, not the generator state left by
    whatever ran before `build_trainer`: two runs that leave it in different states start from
    the same parameters."""
    results = []
    for draws in (1, 5):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            results.append(run_distributed(
                _parameters_after_unrelated_draws, world_size=2, args=(tmp, draws)
            ))
    for a, b in zip(results[0][0], results[1][0], strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.cpu_dist
def test_draws_without_workers_follow_the_seed(run_distributed):
    """With `num_workers = 0` a dataset drawing from numpy's global state still gives each rank
    its own draws, and the same ones on every run: numpy is seeded with torch, per rank."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        first = run_distributed(_first_batch_without_workers, world_size=2, args=(tmp,))
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        again = run_distributed(_first_batch_without_workers, world_size=2, args=(tmp,))
    assert not np.array_equal(first[0], first[1]), "both ranks drew the same batch"
    for rank in range(2):
        np.testing.assert_array_equal(first[rank], again[rank])


@pytest.mark.cpu_dist
def test_resume_does_not_replay_the_data(run_distributed):
    """A run resumed at step 2 trains steps 3-4 on new draws, not on the draws of steps 1-2:
    checkpoints hold no generator state, so the streams are reseeded from the restored step."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_batches_across_a_resume, world_size=2, args=(tmp,))
    for first, second in results:
        assert len(first) == 2 and len(second) == 2
        assert not set(first) & set(second), f"the resumed attempt replayed {first} as {second}"


@pytest.mark.cpu_dist
def test_ranks_may_use_different_parameters(run_distributed):
    """Ranks that train through different parameters in one step still reduce gradients over
    one layout, so every slice of both heads trains."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_train_with_rank_dependent_parameters, world_size=2, args=(tmp,))
    for changed in results:
        assert all(changed), f"part of a head did not train: {changed}"


@pytest.mark.cpu_dist
def test_identical_batches_on_every_rank_stop_the_run(run_distributed):
    """Every data-parallel rank holding the same first batch stops the run, instead of training
    one sample copied to every rank."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        messages = run_distributed(_train_on_constant_data, world_size=2, args=(tmp,))
    for message in messages:
        assert message is not None and "same first batch" in message, message


@pytest.mark.cpu_dist
def test_unregistered_model_name_raises(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        assert all(run_distributed(_unregistered_model_name_raises, world_size=2, args=(tmp,)))


def _resume_latest_agrees_across_ranks(rank: int, world_size: int, root: str) -> str:
    # Every rank must be handed the SAME directory to resume into. The caller seeds one, so the
    # scan itself is deterministic here and this pins the broadcast rather than a race: without
    # it, only rank 0 fills in the name and the others resume into output_root / "".
    # The rank-dependent-timestamp case is covered by test_output_dir_agrees_across_ranks.
    return str(resolve_output_dir(Path(root), "agree", "latest"))


def _prep_failure_reaches_every_rank(rank: int, world_size: int, root: str) -> bool:
    """An incompatible resume must fail on ALL ranks, not hang them in the barrier."""
    output_dir = Path(root) / "hasconfig_20260101_000000"
    try:
        _prepare_run_dir(
            output_dir, Path(root) / "config.toml", {"model": {"kwargs": {"d": 2}}}, rank
        )
    except RuntimeError as error:
        return "could not prepare" in str(error)
    return False


@pytest.mark.cpu_dist
def test_resume_latest_agrees_across_ranks(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        existing = Path(tmp) / "agree_20260101_000000"
        existing.mkdir()
        paths = run_distributed(_resume_latest_agrees_across_ranks, world_size=2, args=(tmp,))
    assert paths == [str(existing), str(existing)]


@pytest.mark.cpu_dist
def test_prep_failure_reaches_every_rank(run_distributed):
    # Before this was broadcast, rank 0 raised alone and the others waited out the collective
    # timeout — a bad config cost ten minutes instead of failing at once.
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        directory = Path(tmp) / "hasconfig_20260101_000000"
        directory.mkdir()
        (directory / "resolved_config.json").write_text(
            json.dumps({"model": {"kwargs": {"d": 1}}}), encoding="utf-8"
        )
        (Path(tmp) / "config.toml").write_text("x = 1\n", encoding="utf-8")
        assert all(run_distributed(_prep_failure_reaches_every_rank, world_size=2, args=(tmp,)))
