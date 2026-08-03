from __future__ import annotations

import pytest
import torch.nn as nn

from engine.config import TrainerConfig
from engine.optimizer import build_lr_scheduler, build_optimizer, lr_multiplier


@pytest.mark.unit
def test_build_optimizer_uses_configured_hyperparameters():
    model = nn.Linear(4, 4)
    config = TrainerConfig(max_steps=10, batch_size=1, lr=3e-4, weight_decay=0.05, beta2=0.99)
    optimizer = build_optimizer(model, config)

    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(3e-4)
    assert group["weight_decay"] == pytest.approx(0.05)
    assert group["betas"] == (0.9, 0.99)


@pytest.mark.unit
def test_warmup_ramps_linearly_to_one():
    config = TrainerConfig(max_steps=100, batch_size=1, warmup_steps=10)
    assert lr_multiplier(0, config) == pytest.approx(0.1)
    assert lr_multiplier(4, config) == pytest.approx(0.5)
    assert lr_multiplier(9, config) == pytest.approx(1.0)


@pytest.mark.unit
def test_cosine_decays_from_one_to_min_lr_ratio():
    config = TrainerConfig(max_steps=100, batch_size=1, warmup_steps=10, min_lr_ratio=0.1)
    assert lr_multiplier(10, config) == pytest.approx(1.0)
    assert lr_multiplier(100, config) == pytest.approx(0.1)
    midpoint = lr_multiplier(55, config)
    assert 0.1 < midpoint < 1.0


@pytest.mark.unit
def test_schedule_is_monotonically_non_increasing_after_warmup():
    config = TrainerConfig(max_steps=50, batch_size=1, warmup_steps=5)
    values = [lr_multiplier(step, config) for step in range(5, 51)]
    pairs = zip(values, values[1:], strict=False)  # values[1:] is one shorter by construction
    assert all(later <= earlier + 1e-12 for earlier, later in pairs)


@pytest.mark.unit
def test_no_warmup_starts_at_full_lr():
    config = TrainerConfig(max_steps=100, batch_size=1, warmup_steps=0)
    assert lr_multiplier(0, config) == pytest.approx(1.0)


@pytest.mark.unit
def test_scheduler_applies_multiplier_to_optimizer_lr():
    model = nn.Linear(4, 4)
    config = TrainerConfig(max_steps=100, batch_size=1, lr=1e-2, warmup_steps=10)
    optimizer = build_optimizer(model, config)
    scheduler = build_lr_scheduler(optimizer, config)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    for _ in range(9):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)
