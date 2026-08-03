from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from algorithms.base import BaseAlgorithm
from algorithms.registry import AlgorithmRegistry


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _DummyAlgorithm(BaseAlgorithm):
    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self.model(batch)
        return {"loss": pred.pow(2).mean()}

    def validation_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self.model(batch)
        return {"val_loss": pred.pow(2).mean()}


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(AlgorithmRegistry, "_registry", {})


@pytest.mark.unit
def test_base_algorithm_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseAlgorithm(_DummyModel())


@pytest.mark.unit
def test_training_and_validation_step():
    algorithm = _DummyAlgorithm(_DummyModel())
    batch = torch.randn(2, 4)

    train_out = algorithm.training_step(batch)
    assert "loss" in train_out
    assert train_out["loss"].requires_grad

    val_out = algorithm.validation_step(batch)
    assert "val_loss" in val_out


@pytest.mark.unit
def test_registry_register_and_build():
    AlgorithmRegistry.register("dummy")(_DummyAlgorithm)

    assert AlgorithmRegistry.available() == ["dummy"]
    algorithm = AlgorithmRegistry.build("dummy", _DummyModel())
    assert isinstance(algorithm, _DummyAlgorithm)


@pytest.mark.unit
def test_registry_rejects_duplicate_name():
    AlgorithmRegistry.register("dummy")(_DummyAlgorithm)
    with pytest.raises(ValueError):
        AlgorithmRegistry.register("dummy")(_DummyAlgorithm)


@pytest.mark.unit
def test_registry_rejects_non_algorithm_subclass():
    with pytest.raises(TypeError):
        AlgorithmRegistry.register("not-an-algorithm")(object)


@pytest.mark.unit
def test_registry_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        AlgorithmRegistry.get("does-not-exist")
