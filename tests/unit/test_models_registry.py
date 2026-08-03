from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.base import BaseModel
from models.registry import ModelRegistry


class _DummyModel(BaseModel):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(ModelRegistry, "_registry", {})


@pytest.mark.unit
def test_register_and_build_with_kwargs():
    ModelRegistry.register("dummy")(_DummyModel)

    assert ModelRegistry.available() == ["dummy"]
    model = ModelRegistry.build("dummy", width=8)
    assert isinstance(model, _DummyModel)
    assert model.linear.in_features == 8


@pytest.mark.unit
def test_rejects_duplicate_name():
    ModelRegistry.register("dummy")(_DummyModel)
    with pytest.raises(ValueError):
        ModelRegistry.register("dummy")(_DummyModel)


@pytest.mark.unit
def test_rejects_non_model_subclass():
    with pytest.raises(TypeError):
        ModelRegistry.register("not-a-model")(object)


@pytest.mark.unit
def test_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        ModelRegistry.get("does-not-exist")
