from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
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

logger = logging.getLogger(__name__)

_STEP_PREFIX = "step_"


def coordination_group() -> dist.ProcessGroup | None:
    """A Gloo group for DCP's planning collectives, or None when the default one will do.

    Before anything is written, DCP agrees on *what* each rank will write by scattering a pickled
    save plan -- an object collective, not a tensor one. Over NCCL that pickle is staged through a
    GPU buffer, and on a multi-node job the buffer crosses InfiniBand, where a ~400 KB plan
    faulted the queue pair outright:

        NET/IB : mlx5_5:1 async fatal event on QP: local access violation work queue error

    which killed a 16-rank run at its first checkpoint, after training and validation had run
    correctly for every step before it. Intra-node the same collective stays on NVLink and never
    shows the problem, which is why this only appears once a job spans hosts.

    Only the *metadata* goes through this group; each rank writes its own tensor shards to storage
    directly, so moving the coordination to Gloo costs nothing measurable at checkpoint sizes this
    repo produces. Returns None when there is no process group at all (single-process runs, where
    DCP takes its own non-distributed path) or when the default one is already Gloo, since then
    there is nothing to route around.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return None
    if dist.get_backend() == "gloo":
        return None
    logger.info("checkpointing: coordinating save/load plans over a Gloo group")
    return dist.new_group(backend="gloo")


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
        # Built once, here: `new_group` is itself a collective, so it has to be reached by every
        # rank at the same point, not lazily on the first rank that happens to save.
        self._process_group = coordination_group()

    def _state_dict(self, train_state: TrainState) -> dict[str, Any]:
        return {
            "model": get_model_state_dict(self._module),
            "optim": get_optimizer_state_dict(self._module, self._optimizer),
            "train_state": train_state,
        }

    def save(self, step: int) -> Path:
        path = self._dir / f"{_STEP_PREFIX}{step}"
        dcp.save(
            self._state_dict(TrainState(step)),
            checkpoint_id=str(path),
            process_group=self._process_group,
        )
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
        return self._load(path)

    def load_step(self, step: int) -> int:
        """Restore one specific checkpoint in place. Returns its step.

        Resuming always wants the newest, but evaluation often wants a named one -- comparing two
        points of the same run says whether it is still improving, which the newest alone cannot.
        """
        path = self._dir / f"{_STEP_PREFIX}{step}"
        if not path.is_dir():
            available = sorted(
                int(p.name[len(_STEP_PREFIX) :])
                for p in self._dir.iterdir()
                if p.name.startswith(_STEP_PREFIX)
            ) if self._dir.is_dir() else []
            raise ValueError(f"no checkpoint at step {step} in {self._dir}; have {available}")
        return self._load(path)

    def _load(self, path: Path) -> int:
        train_state = TrainState()
        state_dict = self._state_dict(train_state)
        dcp.load(state_dict, checkpoint_id=str(path), process_group=self._process_group)
        set_model_state_dict(self._module, state_dict["model"])
        set_optimizer_state_dict(self._module, self._optimizer, state_dict["optim"])
        return train_state.step
