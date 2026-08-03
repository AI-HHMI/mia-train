from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.base import BaseModel


class _DummyModel(BaseModel):
    def __init__(self, in_features: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 2 * self.linear.in_features * self.linear.out_features


@pytest.mark.unit
def test_base_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseModel()


@pytest.mark.unit
def test_forward_and_flops():
    model = _DummyModel()
    out = model(torch.randn(2, 4))
    assert out.shape == (2, 4)
    assert model.flops((4,)) == 32


@pytest.mark.unit
def test_num_parameters_counts_all_and_trainable_only():
    model = _DummyModel()
    total = sum(p.numel() for p in model.parameters())
    assert model.num_parameters() == total

    model.linear.bias.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert model.num_parameters(trainable_only=True) == trainable
