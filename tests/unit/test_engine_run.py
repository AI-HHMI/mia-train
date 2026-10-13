"""Unit tests for how `engine.run` wires a run together from its config."""

from __future__ import annotations

from typing import Any

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
from engine.run import build_trainer
from models.base import BaseModel
from models.registry import ModelRegistry
from utils.config import ComponentConfig, RunConfig


class _Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0


class _Algorithm(BaseAlgorithm):
    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": self.model(batch).pow(2).mean()}

    def validation_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": self.model(batch).pow(2).mean()}


class _Data(BaseDataset):
    """Declares the layout its `axes` kwarg names, as miao's `output_axes` does."""

    def __init__(self, axes: str | None = None) -> None:
        super().__init__()
        self._axes = axes

    @property
    def sample_axes(self) -> str | None:
        return self._axes

    def build_dataset(self) -> data.Dataset:
        return data.TensorDataset(torch.zeros(8, 4))


def _build(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, train_axes: Any, val_axes: Any):
    # Through `monkeypatch.setitem`, not `register`, which has no inverse: tests that sweep the
    # whole registry would trip over entries left behind.
    monkeypatch.setitem(ModelRegistry._registry, "_run_probe_model", _Model)
    monkeypatch.setitem(AlgorithmRegistry._registry, "_run_probe_algorithm", _Algorithm)
    monkeypatch.setitem(DataRegistry._registry, "_run_probe_data", _Data)
    config = RunConfig(
        experiment_name="run_probe",
        model=ComponentConfig(name="_run_probe_model"),
        algorithm=ComponentConfig(name="_run_probe_algorithm"),
        data=ComponentConfig(name="_run_probe_data", kwargs={"axes": train_axes}),
        val_data=ComponentConfig(name="_run_probe_data", kwargs={"axes": val_axes}),
        trainer=TrainerConfig(max_steps=1, batch_size=2, measure_mfu=False),
        parallelism=ParallelDims(),
    )
    return build_trainer(config, tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize("val_axes", ["lczyx", "lcyxz", "lxyz"])
def test_val_data_laid_out_unlike_the_training_data_is_refused(tmp_path, monkeypatch, val_axes):
    """The algorithm takes `[data]`'s layout and reads validation batches in it too, so a
    `[val_data]` in z, y, x -- as mia-evals' task YAMLs are -- would be validated transposed, the
    superposition depth gate and the inflated stem acting along x, with no error."""
    with pytest.raises(ValueError, match=r"\[val_data\] gives samples as"):
        _build(tmp_path, monkeypatch, "lcxyz", val_axes)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("train_axes", "val_axes"), [("lcxyz", "lcxyz"), ("lcxyz", None), (None, "lczyx")]
)
def test_val_data_in_the_same_or_an_undeclared_layout_builds(
    tmp_path, monkeypatch, train_axes, val_axes
):
    """A dataset that declares no layout makes no promise to check against."""
    trainer = _build(tmp_path, monkeypatch, train_axes, val_axes)
    assert trainer.val_loader is not None
