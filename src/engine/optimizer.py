from __future__ import annotations

import math

import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .config import TrainerConfig


def build_optimizer(model: nn.Module, config: TrainerConfig) -> Optimizer:
    return AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


def lr_multiplier(step: int, config: TrainerConfig) -> float:
    """Linear warmup then cosine decay to `min_lr_ratio`, as a multiple of the base LR.

    `step` is 0-based, matching LambdaLR's `last_epoch`. Returns 1.0 at the end of warmup and
    exactly `min_lr_ratio` at `max_steps`.
    """
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine


def build_lr_scheduler(optimizer: Optimizer, config: TrainerConfig) -> LambdaLR:
    return LambdaLR(optimizer, lr_lambda=lambda step: lr_multiplier(step, config))
