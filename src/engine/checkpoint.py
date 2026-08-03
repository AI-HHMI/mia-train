from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

_STEP_PREFIX = "step_"


class TrainState(Stateful):
    """Loop position, checkpointed alongside model and optimizer so runs resume exactly."""

    def __init__(self, step: int = 0) -> None:
        self.step = step

    def state_dict(self) -> dict[str, Any]:
        return {"step": torch.tensor(self.step, dtype=torch.int64)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = int(state_dict["step"].item())


class CheckpointManager:
    """PyTorch Distributed Checkpoint (DCP) save/resume for one training run.

    DCP writes sharded state directly from DTensors, so a run can resume under a different
    parallelism layout than it was saved with.
    """

    def __init__(self, module: nn.Module, optimizer: Optimizer, checkpoint_dir: Path) -> None:
        self._module = module
        self._optimizer = optimizer
        self._dir = checkpoint_dir

    def _state_dict(self, train_state: TrainState) -> dict[str, Any]:
        return {
            "model": get_model_state_dict(self._module),
            "optim": get_optimizer_state_dict(self._module, self._optimizer),
            "train_state": train_state,
        }

    def save(self, step: int) -> Path:
        path = self._dir / f"{_STEP_PREFIX}{step}"
        dcp.save(self._state_dict(TrainState(step)), checkpoint_id=str(path))
        return path

    def latest_checkpoint(self) -> Path | None:
        if not self._dir.is_dir():
            return None
        candidates = [p for p in self._dir.iterdir() if p.name.startswith(_STEP_PREFIX)]
        if not candidates:
            return None
        return max(candidates, key=lambda p: int(p.name[len(_STEP_PREFIX) :]))

    def load_latest(self) -> int:
        """Restore the newest checkpoint in place. Returns its step, or 0 if there is none."""
        path = self.latest_checkpoint()
        if path is None:
            return 0
        train_state = TrainState()
        state_dict = self._state_dict(train_state)
        dcp.load(state_dict, checkpoint_id=str(path))
        set_model_state_dict(self._module, state_dict["model"])
        set_optimizer_state_dict(self._module, self._optimizer, state_dict["optim"])
        return train_state.step
