from __future__ import annotations

from dataclasses import dataclass

PRECISIONS = ("fp32", "bf16")


@dataclass(frozen=True)
class TrainerConfig:
    """Hyperparameters for one training run, normally populated from a .toml config."""

    max_steps: int
    batch_size: int
    lr: float = 1e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    grad_clip_norm: float | None = 1.0
    precision: str = "fp32"
    log_every: int = 10
    checkpoint_every: int = 0
    val_every: int = 0
    num_workers: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < max_steps ({self.max_steps})"
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError(f"grad_clip_norm must be > 0 or None, got {self.grad_clip_norm}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
