from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data

from evals.base import BaseEvalTask
from evals.registry import EvalRegistry


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _DummyEvalTask(BaseEvalTask):
    def evaluate(self, model: nn.Module, dataloader: data.DataLoader) -> dict[str, float]:
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in dataloader:
                total += model(batch).sum().item()
                count += batch.shape[0]
        return {"mean_output": total / count}


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(EvalRegistry, "_registry", {})


@pytest.mark.unit
def test_base_eval_task_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseEvalTask()


@pytest.mark.unit
def test_evaluate_returns_metrics():
    model = _DummyModel()
    task = _DummyEvalTask()
    dataloader = data.DataLoader(torch.randn(8, 4), batch_size=4)

    metrics = task.evaluate(model, dataloader)
    assert "mean_output" in metrics
    assert isinstance(metrics["mean_output"], float)


@pytest.mark.unit
def test_registry_register_and_build():
    EvalRegistry.register("dummy")(_DummyEvalTask)

    assert EvalRegistry.available() == ["dummy"]
    task = EvalRegistry.build("dummy")
    assert isinstance(task, _DummyEvalTask)


@pytest.mark.unit
def test_registry_rejects_duplicate_name():
    EvalRegistry.register("dummy")(_DummyEvalTask)
    with pytest.raises(ValueError):
        EvalRegistry.register("dummy")(_DummyEvalTask)


@pytest.mark.unit
def test_registry_rejects_non_eval_task_subclass():
    with pytest.raises(TypeError):
        EvalRegistry.register("not-an-eval-task")(object)


@pytest.mark.unit
def test_registry_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        EvalRegistry.get("does-not-exist")
