from __future__ import annotations

import pytest

from engine.config import TrainerConfig


@pytest.mark.unit
def test_minimal_config_has_expected_defaults():
    config = TrainerConfig(max_steps=100, batch_size=4)
    assert config.precision == "fp32"
    assert config.grad_clip_norm == 1.0
    assert config.warmup_steps == 0
    assert config.checkpoint_every == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"batch_size": 0},
        {"lr": 0.0},
        {"lr": -1.0},
        {"warmup_steps": -1},
        {"warmup_steps": 100},
        {"warmup_steps": 200},
        {"min_lr_ratio": -0.1},
        {"min_lr_ratio": 1.5},
        {"grad_clip_norm": 0.0},
        {"precision": "fp16"},
        {"precision": "float32"},
        {"log_every": 0},
    ],
)
def test_rejects_invalid_values(kwargs):
    base = {"max_steps": 100, "batch_size": 4}
    with pytest.raises(ValueError):
        TrainerConfig(**{**base, **kwargs})


@pytest.mark.unit
def test_grad_clip_norm_may_be_disabled():
    assert TrainerConfig(max_steps=10, batch_size=1, grad_clip_norm=None).grad_clip_norm is None
