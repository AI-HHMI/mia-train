"""Trainer checkpointing at the end of a run.

`Trainer.train()` saves a checkpoint when the loop ends on a step that is *not* a multiple of
`checkpoint_every`, so finishing a run cannot silently discard up to `checkpoint_every - 1` steps
of training. It is guarded by `step > resumed_from`, so re-running an already finished run does
not rewrite the same checkpoint.

These run single-process with no mesh (`dims`/`mesh` left as `None`), so no process group is
needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.utils.data as data

from algorithms.base import BaseAlgorithm
from data.base import BaseDataset
from engine.checkpoint import CheckpointManager, coordination_group
from engine.config import TrainerConfig
from engine.trainer import Trainer
from models.base import BaseModel


class _TinyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0


class _Reconstruct(BaseAlgorithm):
    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": (self.model(batch) - batch).pow(2).mean()}

    def validation_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": (self.model(batch) - batch).pow(2).mean()}


class _Vectors(data.Dataset):
    def __init__(self, n: int = 16) -> None:
        generator = torch.Generator().manual_seed(0)
        self._items = [torch.randn(4, generator=generator) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._items[index]


class _Synthetic(BaseDataset):
    def build_dataset(self) -> data.Dataset:
        return _Vectors()


def _trainer(output_dir: Path, **overrides: Any) -> Trainer:
    # Annotated because `dict(...)` of mixed int/float values infers `dict[str, float]`, which
    # mypy then rejects against `TrainerConfig`'s int fields when splatted.
    settings: dict[str, Any] = dict(
        max_steps=7, batch_size=2, lr=1e-2, log_every=1000, checkpoint_every=5, seed=0
    )
    settings.update(overrides)
    return Trainer(
        algorithm=_Reconstruct(_TinyModel()),
        train_dataset=_Synthetic(),
        config=TrainerConfig(**settings),
        output_dir=output_dir,
    )


def _recorded_saves(trainer: Trainer, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The steps this trainer asks `CheckpointManager` to write, filled in as it runs.

    Needed because the checkpoint directory cannot show a rewrite: DCP overwrites `step_N` in
    place, leaving exactly the same names behind.
    """
    calls: list[int] = []
    real_save = trainer.checkpoints.save

    def _save(step: int) -> Path:
        calls.append(step)
        return real_save(step)

    monkeypatch.setattr(trainer.checkpoints, "save", _save)
    return calls


def _saved_steps(output_dir: Path) -> list[int]:
    checkpoints = output_dir / "checkpoints"
    if not checkpoints.is_dir():
        return []
    return sorted(int(p.name.removeprefix("step_")) for p in checkpoints.iterdir())


@pytest.mark.unit
def test_saves_the_final_step_when_it_misses_the_cadence(tmp_path):
    # 7 steps at every-5 would otherwise leave step_5 as the newest, losing steps 6 and 7.
    _trainer(tmp_path).train()
    assert _saved_steps(tmp_path) == [5, 7]


@pytest.mark.unit
def test_does_not_duplicate_when_the_final_step_is_on_the_cadence(tmp_path):
    _trainer(tmp_path, max_steps=10, checkpoint_every=5).train()
    assert _saved_steps(tmp_path) == [5, 10]


@pytest.mark.unit
def test_writes_nothing_when_checkpointing_is_disabled(tmp_path):
    _trainer(tmp_path, checkpoint_every=0).train()
    assert _saved_steps(tmp_path) == []


@pytest.mark.unit
def test_rerunning_a_finished_run_adds_no_checkpoint(tmp_path, monkeypatch):
    _trainer(tmp_path).train()
    before = _saved_steps(tmp_path)

    # Resuming a completed run trains nothing, so it must not rewrite the final checkpoint — which
    # only the save calls can show, since a rewrite leaves the directory names untouched.
    resumed = _trainer(tmp_path)
    saved = _recorded_saves(resumed, monkeypatch)

    assert resumed.train() == 7
    assert saved == []
    assert _saved_steps(tmp_path) == before


@pytest.mark.unit
def test_resuming_continues_and_saves_the_new_final_step(tmp_path):
    _trainer(tmp_path).train()
    assert _trainer(tmp_path, max_steps=12).train() == 12
    assert _saved_steps(tmp_path) == [5, 7, 10, 12]


# ------------------------------------------------------- DCP plan coordination (multi-node)


@pytest.mark.unit
def test_no_coordination_group_outside_a_process_group():
    # Single-process runs have nothing to coordinate; DCP takes its own non-distributed path.
    assert coordination_group() is None


@pytest.mark.unit
def test_no_coordination_group_when_the_default_backend_is_already_gloo(monkeypatch):
    # Nothing to route around, and building a second Gloo group would be a pointless collective.
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    created = False

    def _new_group(**kwargs):
        nonlocal created
        created = True

    monkeypatch.setattr(dist, "new_group", _new_group)

    assert coordination_group() is None
    assert not created


@pytest.mark.unit
def test_a_gloo_group_is_built_when_the_default_backend_is_nccl(monkeypatch):
    # The regression this guards: DCP scatters its save plan as a pickled *object*, which over
    # NCCL is staged through a GPU buffer. On a job spanning hosts that buffer crosses InfiniBand
    # and faults the queue pair, killing a 16-rank run at its first checkpoint while every
    # training collective before it succeeded. Intra-node it never shows, so only a multi-node
    # run finds it -- hence a test on the decision rather than on the symptom.
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    requested: dict[str, object] = {}

    def _new_group(**kwargs):
        requested.update(kwargs)
        return "the-gloo-group"

    monkeypatch.setattr(dist, "new_group", _new_group)

    assert coordination_group() == "the-gloo-group"
    assert requested == {"backend": "gloo"}


@pytest.mark.unit
def test_save_and_load_pass_the_coordination_group_to_dcp(tmp_path, monkeypatch):
    # A group that is built but not handed to dcp.save would fix nothing.
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(model, optimizer, tmp_path)
    monkeypatch.setattr(manager, "_process_group", "the-gloo-group")

    seen = {}
    monkeypatch.setattr(
        dcp, "save", lambda state, **kw: seen.update(save=kw.get("process_group")) or None
    )
    monkeypatch.setattr(
        dcp, "load", lambda state, **kw: seen.update(load=kw.get("process_group")) or None
    )

    manager.save(step=1)
    (tmp_path / "step_1").mkdir(exist_ok=True)
    manager.load_latest()

    assert seen == {"save": "the-gloo-group", "load": "the-gloo-group"}
