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

from distributed.tensor_parallel import strided_tensor_parallel_tensors

logger = logging.getLogger(__name__)

_STEP_PREFIX = "step_"
# The file DCP writes last, after every shard: its presence is what makes a save complete.
_METADATA = ".metadata"


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
    """Loop position, checkpointed alongside model and optimizer.

    Model, optimizer and step resume exactly; the random streams do not, because no generator
    state is saved. `Trainer._seed_streams` reseeds them from the restored step instead, so a
    resumed run continues on fresh, reproducible draws rather than replaying its first ones.
    """

    def __init__(self, step: int = 0) -> None:
        self.step = step

    def state_dict(self) -> dict[str, Any]:
        return {"step": torch.tensor(self.step, dtype=torch.int64)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = int(state_dict["step"].item())


class CheckpointManager:
    """PyTorch Distributed Checkpoint (DCP) save/resume for one training run.

    DCP writes sharded state directly from DTensors, so a run can resume under a different
    parallelism layout than it was saved with -- except tensor parallelism over the fused qkv,
    which DCP cannot describe and `check_supported` refuses.
    """

    def __init__(self, module: nn.Module, optimizer: Optimizer, checkpoint_dir: Path) -> None:
        self._module = module
        self._optimizer = optimizer
        self._dir = checkpoint_dir
        # Built once, here: `new_group` is itself a collective, so it has to be reached by every
        # rank at the same point, not lazily on the first rank that happens to save.
        self._process_group = coordination_group()
        self._unsupported = strided_tensor_parallel_tensors(module)

    def check_supported(self) -> None:
        """Raise if DCP cannot checkpoint this module's layout: the fused qkv under tp > 1.

        A save would fail on every rank, and a load -- resuming a run saved without tensor
        parallelism, which a changed `[parallelism]` otherwise allows -- would silently scramble
        q, k and v (see `strided_tensor_parallel_tensors`).
        """
        if self._unsupported:
            raise ValueError(
                "DCP cannot checkpoint the tensor-parallel fused qkv projection "
                f"({self._unsupported[0]} and {len(self._unsupported) - 1} more): its shards are "
                "three disjoint row blocks per rank, so a save fails with 'overlapping chunks' and "
                "a load fills k and v with other heads' rows. Under tp > 1, set "
                "checkpoint_every = 0 and start from no checkpoint; use tp = 1 to save or resume."
            )

    def _state_dict(self, train_state: TrainState) -> dict[str, Any]:
        return {
            "model": get_model_state_dict(self._module),
            "optim": get_optimizer_state_dict(self._module, self._optimizer),
            "train_state": train_state,
        }

    def save(self, step: int) -> Path:
        self.check_supported()
        path = self._dir / f"{_STEP_PREFIX}{step}"
        dcp.save(
            self._state_dict(TrainState(step)),
            checkpoint_id=str(path),
            process_group=self._process_group,
        )
        return path

    def latest_checkpoint(self) -> Path | None:
        """The newest *complete* checkpoint, or None.

        DCP creates `step_N/` when it starts planning and writes `.metadata` last, once every
        shard is on disk, so a save interrupted in between -- a bkill, a node failure, a full
        quota -- leaves a directory without it. Picking that one would fail every later resume at
        startup until someone deleted it by hand; the newest complete one is used instead.
        """
        if not self._dir.is_dir():
            return None
        candidates = [p for p in self._dir.iterdir() if p.name.startswith(_STEP_PREFIX)]
        complete = [p for p in candidates if (p / _METADATA).is_file()]
        unfinished = sorted(p.name for p in candidates if p not in complete)
        if unfinished and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[checkpoint] ignoring unfinished save(s) without {_METADATA}: "
                  f"{', '.join(unfinished)}", flush=True)
        if not complete:
            return None
        return max(complete, key=checkpoint_step)

    def load_latest(self, *, model_only: bool = False) -> int:
        """Restore the newest checkpoint in place. Returns its step, or 0 if there is none."""
        path = self.latest_checkpoint()
        if path is None:
            return 0
        return self.load(path, model_only=model_only)

    def load_step(self, step: int, *, model_only: bool = False) -> int:
        """Restore one specific checkpoint in place. Returns its step.

        Resuming always wants the newest, but evaluation often wants a named one -- comparing two
        points of the same run says whether it is still improving, which the newest alone cannot.
        """
        path = self._dir / f"{_STEP_PREFIX}{step}"
        if not (path / _METADATA).is_file():
            available = sorted(
                checkpoint_step(p)
                for p in self._dir.iterdir()
                if p.name.startswith(_STEP_PREFIX) and (p / _METADATA).is_file()
            ) if self._dir.is_dir() else []
            state = "an unfinished save" if path.is_dir() else "no checkpoint"
            raise ValueError(f"{state} at step {step} in {self._dir}; complete ones: {available}")
        return self.load(path, model_only=model_only)

    def load(self, path: Path, *, model_only: bool = False) -> int:
        """Restore the checkpoint at `path` in place. Returns its step.

        `model_only` reads the weights and the step and leaves the optimizer alone, which is all
        prediction needs -- and the only load that works for a checkpoint whose optimizer state
        does not match a fresh optimizer's, like one saved during a frozen-backbone warm-up.
        """
        self.check_supported()
        train_state = TrainState()
        if model_only:
            state_dict = {"model": get_model_state_dict(self._module), "train_state": train_state}
            dcp.load(state_dict, checkpoint_id=str(path), process_group=self._process_group)
            set_model_state_dict(self._module, state_dict["model"])
            return train_state.step
        state_dict = self._state_dict(train_state)
        dcp.load(state_dict, checkpoint_id=str(path), process_group=self._process_group)
        set_model_state_dict(self._module, state_dict["model"])
        set_optimizer_state_dict(self._module, self._optimizer, state_dict["optim"])
        return train_state.step


def checkpoint_step(path: Path) -> int:
    """The step a `step_N` checkpoint directory was saved at."""
    return int(path.name[len(_STEP_PREFIX) :])
