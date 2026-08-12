from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from algorithms.registry import AlgorithmRegistry
from data.augment import VolumeAugmentation
from data.registry import DataRegistry
from distributed.setup import destroy_distributed, device_type, init_distributed
from models.registry import ModelRegistry
from utils.cluster import checkpoint_dir
from utils.config import RunConfig, as_plain_dict, diff_resolved, load_run_config
from utils.pretrained import load_pretrained
from utils.provenance import write_run_artifacts

from .trainer import Trainer

_REPO_ROOT = Path(__file__).resolve().parents[2]


RESUME_LATEST = "latest"


def _latest_run_dir(output_root: Path, experiment_name: str) -> Path | None:
    """Newest existing run directory for this experiment, if there is one.

    Names end in a `%Y%m%d_%H%M%S` stamp, so sorting them lexicographically orders them by time.
    """
    if not output_root.is_dir():
        return None
    prefix = f"{experiment_name}_"
    existing = sorted(p for p in output_root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    return existing[-1] if existing else None


def resolve_output_dir(
    output_root: Path, experiment_name: str, resume: str | None = None
) -> Path:
    """Pick this run's artifact directory.

    Without `resume`, a fresh `output_root/<experiment_name>_<timestamp>`. With
    `resume="latest"`, the newest existing directory for this experiment, falling back to a fresh
    one when there is none — which is what lets a single submission script serve both the first
    launch and every resubmission after a wall-time limit or node failure. With an explicit path,
    exactly that directory.

    Rank 0 decides and the rest are told. Both halves of the choice are rank-dependent: a
    timestamp can straddle a second boundary, and a directory scan over NFS can disagree between
    ranks. Either would silently split one run across two directories.
    """
    if resume is not None and resume != RESUME_LATEST:
        chosen = Path(resume)
        if not chosen.is_dir():
            raise ValueError(
                f"--resume {resume!r} is not an existing directory; omit --resume to start a "
                f"fresh run, or pass --resume {RESUME_LATEST} to continue the most recent one"
            )
        return chosen

    name = [""]
    if not dist.is_initialized() or dist.get_rank() == 0:
        existing = _latest_run_dir(output_root, experiment_name) if resume else None
        name[0] = (
            existing.name
            if existing is not None
            else f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    if dist.is_initialized():
        dist.broadcast_object_list(name, src=0)
    return output_root / name[0]


# A different architecture cannot load the checkpoint's parameters at all, so say so up front
# instead of failing deep inside DCP with a shape mismatch. Everything else — a new learning
# rate, more steps, a different mask ratio — is a legitimate thing to change on a restart.
_INCOMPATIBLE_ON_RESUME = ("model.", "algorithm.name")


def _report_resumed_config(output_dir: Path, resolved: dict[str, Any]) -> None:
    """Compare this attempt's settings against the ones the run was started with."""
    stored_path = output_dir / "resolved_config.json"
    if not stored_path.is_file():
        return

    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    changes = diff_resolved(stored, resolved)
    if not changes:
        return

    incompatible = {
        path: change
        for path, change in changes.items()
        if any(path.startswith(prefix) for prefix in _INCOMPATIBLE_ON_RESUME)
    }
    if incompatible:
        detail = "; ".join(f"{p}: {old!r} -> {new!r}" for p, (old, new) in incompatible.items())
        raise ValueError(
            f"cannot resume {output_dir.name}: the architecture changed since it was started "
            f"({detail}). Its checkpoint holds the old parameters. Start a fresh run under a new "
            "experiment_name instead."
        )

    detail = "; ".join(f"{p}: {old!r} -> {new!r}" for p, (old, new) in changes.items())
    print(f"[resume] continuing {output_dir.name} with changed settings: {detail}", flush=True)


def resolved_settings(config: RunConfig) -> dict[str, Any]:
    """The run's settings with every dataset section expanded to what it actually amounts to.

    A `[data]` section may describe itself by reference — miao's `config_path` points at a YAML —
    and a record that kept only the reference would stop being true the moment that file was
    edited. Asking the dataset *class* keeps this generic: no registry entry is constructed, so
    the record is complete before any data is touched.
    """
    resolved = as_plain_dict(config)
    for section in ("data", "val_data"):
        component = getattr(config, section)
        if component is not None:
            resolved[section]["kwargs"] = DataRegistry.get(component.name).resolve_settings(
                **component.kwargs
            )
    return resolved


def referenced_config_files(config: RunConfig) -> tuple[Path, ...]:
    """Files the config points at, to be copied into the run directory beside it.

    `resolved_settings` already captures every value; these preserve the files themselves, whose
    comments explain why the values are what they are. Deduplicated, since a train and a
    validation section commonly name the same dataset file.
    """
    seen: dict[Path, None] = {}
    for section in ("data", "val_data"):
        component = getattr(config, section)
        if component is not None:
            for path in DataRegistry.get(component.name).referenced_files(**component.kwargs):
                seen[path] = None
    return tuple(seen)


def _prepare_run_dir(
    output_dir: Path,
    config_path: Path,
    resolved: dict[str, Any],
    rank: int,
    referenced: tuple[Path, ...] = (),
) -> None:
    """Have rank 0 create the run directory and its artifacts, then let everyone proceed.

    Only rank 0 writes, since the alternative is every rank racing on the same files. But a
    failure there — an incompatible resume, a full disk, a permission problem — must reach the
    other ranks, or they sit in the barrier below until the collective times out and a config
    mistake costs ten minutes instead of failing immediately. So the outcome is broadcast and
    every rank raises together.
    """
    failure: list[str | None] = [None]
    if rank == 0:
        try:
            # While the previous attempt's copy is still on disk, before it is overwritten.
            _report_resumed_config(output_dir, resolved)
            write_run_artifacts(output_dir, config_path, resolved, _REPO_ROOT, referenced)
        except Exception as error:  # forwarded to the other ranks below
            failure[0] = f"{type(error).__name__}: {error}"

    if dist.is_initialized():
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(f"could not prepare {output_dir}: {failure[0]}")

    if dist.is_initialized():
        dist.barrier()  # no rank may write checkpoints before rank 0 has created the directory


def build_trainer(
    config: RunConfig,
    output_dir: Path,
    mesh: DeviceMesh | None = None,
    device: torch.device | None = None,
) -> Trainer:
    """Instantiate model, algorithm, and datasets from the registries named in the config.

    The dataset is built first and handed to the algorithm, so a strategy can adopt the data's
    layout instead of having it restated in its own config section where the two could drift
    (see `BaseAlgorithm`).
    """
    train_dataset = DataRegistry.build(config.data.name, **config.data.kwargs)
    if config.augment.enabled():
        # The training set only. `val_data` is deliberately never wrapped: validation has to
        # measure the model on the data as it is, and there is no config key that can change that.
        train_dataset.attach_transform(
            VolumeAugmentation(**dataclasses.asdict(config.augment))
        )
        print(f"[augment] training data: {config.augment}", flush=True)
    val_dataset = (
        DataRegistry.build(config.val_data.name, **config.val_data.kwargs)
        if config.val_data is not None
        else None
    )
    model = ModelRegistry.build(config.model.name, **config.model.kwargs)
    if config.init.path:
        # Before the algorithm wraps it and before any parallelism is applied, so the load sees
        # plain unsharded tensors and a strategy's own parameters (a decoder, say) keep the
        # initialisation they were built with.
        load_pretrained(
            model,
            config.init.path,
            prefix=config.init.prefix,
            inflate=config.init.inflate_2d_to_3d,
            skip=config.init.skip,
            strict=config.init.strict,
            allow_unused=config.init.allow_unused,
        )
    algorithm = AlgorithmRegistry.build(
        config.algorithm.name, model, train_dataset, **config.algorithm.kwargs
    )
    return Trainer(
        algorithm=algorithm,
        train_dataset=train_dataset,
        config=config.trainer,
        output_dir=output_dir,
        dims=config.parallelism,
        mesh=mesh,
        val_dataset=val_dataset,
        device=device,
    )


def run(config_path: Path, output_root: Path | None = None, resume: str | None = None) -> Path:
    """Execute one training run end to end. Returns the run's artifact directory.

    `output_root` defaults to [environment].checkpoint_dir from configs/cluster/active.toml, an
    absolute path, so every run lands in the same place regardless of the launching directory.

    `resume` is None for a fresh run, "latest" to continue the newest run of this experiment, or
    a path to continue that exact directory. "latest" makes a submission script idempotent: the
    same command starts the run and, after a wall-time limit or a node failure, continues it.
    """
    config = load_run_config(config_path)
    root = output_root or checkpoint_dir()
    rank, _world_size, local_rank = init_distributed()
    try:
        mesh = config.parallelism.build_mesh(device_type())
        device = (
            torch.device("cuda", local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        output_dir = resolve_output_dir(root, config.experiment_name, resume)
        _prepare_run_dir(
            output_dir,
            config_path,
            resolved_settings(config),
            rank,
            referenced_config_files(config),
        )

        build_trainer(config, output_dir, mesh=mesh, device=device).train()
    finally:
        destroy_distributed()
    return output_dir
